#!/usr/bin/env python3
"""
Automated ICP Correction on Testing Poses - 35 Corners Version (All in METERS)

This script:
1. Loads all images and robot poses from testing_poses directory
2. Detects ChArUco board (14x9 board with 40mm squares) in each image
3. Computes predicted corner positions using initial calibration
4. Uses ICP to compute corrected camera-to-gripper transformation
5. Saves the corrected extrinsics

This version uses 35 corners from ICPNEW.py (not 40).
Board specs: 14x9 ChArUco, 40mm squares, 30mm markers, DICT_5X5_1000
ALL UNITS IN METERS (ground truth converted from mm to m)

COORDINATE FRAME TRANSFORMATIONS:
---------------------------------
Q (Ground Truth): 
  - Measured directly in robot base frame at 35 specific physical locations
  - These are the true 3D positions we want to match
  
P (Predicted):
  - Theoretical corner positions in board frame (7x5 grid pattern)
  - Transformed to robot base frame via the transformation chain:
    
    P_base = T_base2gripper @ X0 @ T_board2cam @ P_board
    
    where:
    - P_board: Corner position in board frame (from theoretical grid)
    - T_board2cam: Board-to-camera transform (from solvePnP detection)
    - X0: Camera-to-gripper extrinsics (what we're calibrating)
    - T_base2gripper: Robot forward kinematics (known from robot)
    - P_base: Predicted corner position in robot base frame
    
ICP finds the correction deltaX such that:
  Q ≈ deltaX @ P
  
This deltaX is then used to compute corrected extrinsics X1.
"""

import numpy as np
import cv2
import argparse
import os
import glob
import sys
from typing import Optional, Tuple, Dict, List
from scipy.optimize import linear_sum_assignment

# Add parent directory to path to import ICP module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ICP functions from ICPNEW
from ICPNEW import (
    compute_icp_transformation,
    compute_corrected_extrinsics,
    analyze_point_quality,
    to_homogeneous,
    from_homogeneous
)

# Default marker parameters
CHARUCO_SQUARES_X = 14
CHARUCO_SQUARES_Y = 9
SQUARE_LEN_MM = 40.0  # 40mm squares (physical dimension)
MARKER_LEN_MM = 30.0  # 30mm markers (physical dimension)
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_1000

# Convert to meters for all computations
SQUARE_LEN_M = SQUARE_LEN_MM / 1000.0
MARKER_LEN_M = MARKER_LEN_MM / 1000.0


def default_detector_params():
    """Create detector parameters optimized for ChArUco detection with subpixel refinement"""
    p = cv2.aruco.DetectorParameters()
    # Use subpixel corner refinement for better accuracy
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.01
    return p


def estimate_charuco_pose_solvepnp(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                                    board, aruco_dict, min_corners: int = 6, debug: bool = False) -> Optional[Dict]:
    """Estimate ChArUco board pose using solvePnP and return detected corners with subpixel refinement"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ChArUco corners using CharucoDetector (which handles ArUco detection internally)
    params = default_detector_params()
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_detector.setDetectorParameters(params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        if debug:
            n = 0 if charuco_ids is None else len(charuco_ids)
            if marker_ids is not None:
                print(f"    Detected {len(marker_ids)} ArUco markers but only {n} ChArUco corners (need {min_corners})")
            else:
                print(f"    No markers detected")
        return None

    if debug:
        print(f"    Detected {len(marker_ids) if marker_ids is not None else 0} ArUco markers, {len(charuco_ids)} ChArUco corners")

    # Additional subpixel refinement using cv2.cornerSubPix for even better accuracy
    # Define criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    winSize = (5, 5)
    zeroZone = (-1, -1)
    
    # cornerSubPix modifies in place and returns None, so copy first
    charuco_corners_refined = charuco_corners.copy()
    cv2.cornerSubPix(
        gray, 
        charuco_corners_refined, 
        winSize, 
        zeroZone, 
        criteria
    )

    # Use matchImagePoints to get 3D-2D correspondences
    obj_points, img_points = board.matchImagePoints(charuco_corners_refined, charuco_ids)

    if obj_points is None or len(obj_points) == 0:
        if debug:
            print(f"    matchImagePoints failed")
        return None

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, D,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        if debug:
            print(f"    solvePnP failed")
        return None

    # Refine pose estimate
    rvec, tvec = cv2.solvePnPRefineLM(obj_points, img_points, K, D, rvec, tvec)

    # Convert to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    return {
        'rvec': rvec,
        'tvec': tvec.flatten(),
        'R_matrix': R_matrix,
        'num_corners': len(charuco_ids),
        'charuco_corners_img': charuco_corners_refined,  # 2D image points (refined)
        'charuco_ids': charuco_ids,  # Corner IDs
        'charuco_corners_obj': obj_points,  # 3D object points in board frame
    }


def generate_charuco_corners_board_frame(squares_x: int, squares_y: int, square_len: float, num_corners: int = 35) -> np.ndarray:
    """
    Generate ChArUco corner positions in board frame.
    For 5x7 board:
    - Internal corners: (5-1) × (7-1) = 4 × 6 = 24 corners
    - All grid intersections: 5 × 7 = 35 corners
    - Extended grid: 6 × 8 = 48 corners

    Returns array in meters.
    """
    corners = []
    # Generate grid corners based on requested number
    if num_corners <= 24:
        # Internal corners only: (squares_x - 1) × (squares_y - 1)
        for row in range(squares_y - 1):
            for col in range(squares_x - 1):
                x = col * square_len
                y = row * square_len
                corners.append([x, y, 0])
    elif num_corners <= 35:
        # All grid intersections: squares_x × squares_y
        for row in range(squares_y):
            for col in range(squares_x):
                x = col * square_len
                y = row * square_len
                corners.append([x, y, 0])
    else:
        # Extended grid: (squares_x + 1) × (squares_y + 1)
        for row in range(squares_y + 1):
            for col in range(squares_x + 1):
                x = col * square_len
                y = row * square_len
                corners.append([x, y, 0])
    return np.array(corners[:num_corners], dtype=float)


def load_test_data(testing_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load all images and robot poses from testing directory (poses in METERS)"""
    pose_files = sorted(glob.glob(os.path.join(testing_dir, "pose*.npy")))

    images = []
    T_base2gripper_list = []
    filenames = []

    for pose_file in pose_files:
        # Load robot pose
        pose_data = np.load(pose_file, allow_pickle=True).item()

        # Build transformation matrix
        if 'T_base2gripper' in pose_data:
            T_bt = pose_data['T_base2gripper']
        elif 'R' in pose_data and 't' in pose_data:
            R = pose_data['R']
            t = pose_data['t']  # t is in meters from pose files
            T_bt = to_homogeneous(R, t)
        else:
            print(f"⚠️  Skipping {os.path.basename(pose_file)}: missing pose data")
            continue

        # Load corresponding image
        img_file = pose_file.replace('.npy', '.jpg')
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
                T_base2gripper_list.append(T_bt)
                filenames.append(os.path.basename(pose_file).replace('.npy', ''))

    return images, T_base2gripper_list, filenames


def main():
    parser = argparse.ArgumentParser(description="Automated ICP correction on testing poses (35 corners)")
    parser.add_argument("--testing-dir", default="../testing_poses",
                       help="Directory with test images and poses")
    parser.add_argument("--extrinsics", default="../test_extrinsics.npy",
                       help="Initial camera-to-gripper extrinsics file (.npy or .npz)")
    parser.add_argument("--intrinsics", default="../gemini_intrinsics/gemini_355_rgb_intrinsics_20250930_181519.json",
                       help="Camera intrinsics file (.npz or .json)")
    parser.add_argument("--num-corners", type=int, default=35,
                       help="Number of corners to use (max 35 from ICPNEW.py)")
    parser.add_argument("--output", default="icp_corrected_extrinsics_35corners.npy",
                       help="Output corrected extrinsics file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results for each image")

    args = parser.parse_args()

    print("="*70)
    print("Automated ICP Correction on Testing Poses (35 Corners Version)")
    print("="*70)
    print()

    # Load extrinsics (in METERS)
    print(f"Loading initial extrinsics from: {args.extrinsics}")
    try:
        if args.extrinsics.endswith('.npy'):
            X0 = np.load(args.extrinsics)
        elif args.extrinsics.endswith('.npz'):
            calib = np.load(args.extrinsics, allow_pickle=True)
            if 'T_cam2gripper' in calib:
                X0 = calib['T_cam2gripper']
            else:
                R = calib['R_cam2gripper']
                t = calib['t_cam2gripper']
                X0 = to_homogeneous(R, t)
        
        print(f"✓ Loaded initial camera-to-gripper extrinsics")
        print(f"  Translation (m): [{X0[0,3]:.6f}, {X0[1,3]:.6f}, {X0[2,3]:.6f}]")
    except Exception as e:
        print(f"❌ Failed to load extrinsics: {e}")
        return

    # Load intrinsics
    print(f"\nLoading intrinsics from: {args.intrinsics}")
    try:
        if args.intrinsics.endswith('.npz'):
            intr = np.load(args.intrinsics)
            K = intr['camera_matrix']
            D = intr['dist_coeffs']
        elif args.intrinsics.endswith('.json'):
            import json
            with open(args.intrinsics, 'r') as f:
                intr = json.load(f)
            if 'camera_matrix' in intr:
                K = np.array(intr['camera_matrix'])
                D = np.array(intr['dist_coeffs']).flatten()
            else:
                for value in intr.values():
                    if isinstance(value, dict) and 'fx' in value:
                        fx, fy = value['fx'], value['fy']
                        cx, cy = value['cx'], value['cy']
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                        D = np.array(value.get('distortion', [0]*5)).flatten()
                        break
        print("✓ Loaded camera intrinsics")
    except Exception as e:
        print(f"❌ Failed to load intrinsics: {e}")
        return

    # Load test data
    print(f"\nLoading test data from: {args.testing_dir}")
    images, T_base2gripper_list, filenames = load_test_data(args.testing_dir)
    print(f"✓ Loaded {len(images)} images with robot poses")

    # Setup ChArUco board (5x7)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        aruco_dict
    )

    # Ground truth corners from ICPNEW.py (35 corners in millimeters, convert to meters)
    Q_ground_truth_mm = np.array([
        [355.5, -288.8, -11.1],      # corner 1
        [356.5, -209.1, -11.3],      # corner 2
        [357.6, -129.5, -11.5],      # corner 3
        [358.5, -49.5, -12.0],       # corner 4
        [360.1, 30.3, -12.1],        # corner 5
        [361.4, 110.4, -12.0],       # corner 6
        [362.8, 190.6, -11.9],       # corner 7
        [395.3, -289.4, -10.3],      # corner 8
        [396.5, -209.8, -10.5],      # corner 9
        [397.6, -129.8, -10.5],      # corner 10
        [398.9, -49.9, -10.5],       # corner 11
        [400.2, 30, -11.2],          # corner 12
        [401.5, 110, -11.2],         # corner 13
        [402.8, 190.1, -11.0],       # corner 14
        [435.8, -289.9, -9.5],       # corner 15
        [436.7, -210.1, -9.6],       # corner 16
        [437.8, -130.3, -9.8],       # corner 17
        [438.9, -50.3, -10.2],       # corner 18
        [440.2, 29.6, -10.4],        # corner 19
        [441.3, 109.5, -10.3],       # corner 20
        [442.6, 189.8, -10.0],       # corner 21
        [475.6, -290.1, -9.0],       # corner 22
        [476.5, -210.8, -9.0],       # corner 23
        [478, -130.7, -9.0],         # corner 24
        [479.1, -50.9, -9.3],        # corner 25
        [480.1, 29.2, -9.5],         # corner 26
        [481.4, 109.1, -9.4],        # corner 27
        [482.4, 189.2, -9.2],        # corner 28
        [515.7, -290.8, -8.6],       # corner 29
        [516.7, -210.9, -8.5],       # corner 30
        [517.7, -131.1, -8.2],       # corner 31
        [519.1, -51.2, -8.3],        # corner 32
        [520.2, 29, -8.4],           # corner 33
        [521.3, 108.7, -8.4],        # corner 34
        [522.5, 188.8, -8.1]         # corner 35
    ])

    # Use only the requested number of corners and convert to meters
    num_corners = min(args.num_corners, len(Q_ground_truth_mm))
    Q_ground_truth_m = Q_ground_truth_mm[:num_corners] / 1000.0  # Convert mm to meters
    
    print(f"\n✓ Using {num_corners} measured ground truth positions (robot base frame)")
    print(f"  Point 1: [{Q_ground_truth_m[0][0]*1000:.1f}, {Q_ground_truth_m[0][1]*1000:.1f}, {Q_ground_truth_m[0][2]*1000:.1f}] mm")
    print(f"  Point 8: [{Q_ground_truth_m[7][0]*1000:.1f}, {Q_ground_truth_m[7][1]*1000:.1f}, {Q_ground_truth_m[7][2]*1000:.1f}] mm")
    print(f"  Point 35: [{Q_ground_truth_m[-1][0]*1000:.1f}, {Q_ground_truth_m[-1][1]*1000:.1f}, {Q_ground_truth_m[-1][2]*1000:.1f}] mm")
    print(f"\nMatching detected ChArUco corners to ground truth within 10mm threshold...")

    # Process all images and collect predicted vs ground truth corners
    print("\n" + "="*70)
    print("Processing images and collecting corner data...")
    print("="*70)

    P_list = []  # Predicted corner positions in base frame (using initial calibration)
    Q_list = []  # Ground truth corner positions in base frame
    successful_count = 0
    failed_count = 0

    for i, (img, T_base2gripper, filename) in enumerate(zip(images, T_base2gripper_list, filenames)):
        # Detect board
        debug_detection = args.verbose or (i < 5)  # Debug first 5 or if verbose
        pose_result = estimate_charuco_pose_solvepnp(img, K, D, board, aruco_dict, debug=debug_detection)

        if pose_result is None:
            failed_count += 1
            if debug_detection:
                print(f"[{i+1}/{len(images)}] {filename}: ❌ Detection failed")
            continue

        # solvePnP returns R,t such that: P_camera = R @ P_board + t
        # This directly gives us board-to-camera transformation
        R_board2cam = pose_result['R_matrix']
        t_board2cam_m = pose_result['tvec']
        T_board2cam_m = to_homogeneous(R_board2cam, t_board2cam_m)

        # Get the detected ChArUco corners in board frame (3D object points)
        detected_corners_board = pose_result['charuco_corners_obj']  # Nx3 array in meters
        
        # Build full transformation chain: board -> camera -> gripper -> base
        T_board2base_predicted_m = T_base2gripper @ X0 @ T_board2cam_m

        # Transform detected corners to robot base frame
        detected_corners_base = []
        for corner_board in detected_corners_board:
            corner_hom = np.append(corner_board, 1.0)
            corner_base = (T_board2base_predicted_m @ corner_hom)[:3]
            detected_corners_base.append(corner_base)
        detected_corners_base = np.array(detected_corners_base)
        
        # Use Hungarian algorithm for optimal one-to-one matching
        # Build cost matrix: distance between each detected corner and each ground truth corner
        match_threshold = 0.010  # 10mm in meters
        n_detected = len(detected_corners_base)
        n_ground_truth = len(Q_ground_truth_m)
        
        # Vectorized computation of cost matrix (much faster than nested loops)
        # Shape: (n_detected, n_ground_truth)
        # Broadcasting: detected_corners_base[:, None, :] - Q_ground_truth_m[None, :, :]
        diff = detected_corners_base[:, np.newaxis, :] - Q_ground_truth_m[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        # Set very high cost for distances beyond threshold
        cost_matrix = np.where(distances < match_threshold, distances, 1e6)
        
        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter out matches that exceed threshold
        matched_pairs = []
        for det_idx, gt_idx in zip(row_ind, col_ind):
            dist = cost_matrix[det_idx, gt_idx]
            if dist < match_threshold:
                p_corner_m = detected_corners_base[det_idx]
                q_corner_m = Q_ground_truth_m[gt_idx]
                matched_pairs.append((p_corner_m, q_corner_m, dist))
        
        # Add matched pairs to P and Q lists
        for p_corner, q_corner, dist in matched_pairs:
            P_list.append(p_corner)
            Q_list.append(q_corner)

        successful_count += 1
        if args.verbose:
            print(f"[{i+1}/{len(images)}] {filename}: ✓ {pose_result['num_corners']} detected, {len(matched_pairs)} matched")
        
        # Print detailed transformation info for first image
        if i == 0 and args.verbose:
            print(f"\n{'='*70}")
            print(f"First Image Matching Details:")
            print(f"{'='*70}")
            print(f"Detected {len(detected_corners_board)} ChArUco corners")
            print(f"Matched {len(matched_pairs)} corners to ground truth")
            print(f"\nFirst 3 matched corners:")
            for j in range(min(3, len(matched_pairs))):
                p, q, dist = matched_pairs[j]
                print(f"  Match {j+1}:")
                print(f"    P (detected): [{p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}] mm")
                print(f"    Q (ground truth): [{q[0]*1000:.1f}, {q[1]*1000:.1f}, {q[2]*1000:.1f}] mm")
                print(f"    Distance: {dist*1000:.1f} mm")
            print(f"{'='*70}\n")

    print(f"\n✓ Successfully processed: {successful_count}/{len(images)} images")
    print(f"✗ Failed detections: {failed_count}/{len(images)} images")
    print(f"✓ Total corner correspondences: {len(P_list)}")

    if len(P_list) < 4:
        print("\n❌ Not enough data for ICP! Need at least 4 point correspondences.")
        return

    # Convert to numpy arrays (in meters)
    P_m = np.array(P_list)  # meters
    Q_m = np.array(Q_list)  # meters

    # Analyze point quality (in meters)
    print("\n" + "="*70)
    print("Analyzing point quality...")
    print("="*70)
    analyze_point_quality(P_m, Q_m)

    # Compute ICP transformation (in base frame, in meters)
    print("\n" + "="*70)
    print("Computing ICP correction in BASE frame...")
    print("="*70)

    # Both P_m and Q_m are already in base frame (in meters)
    # P was computed as: T_base2gripper @ X0 @ T_board2cam @ P_board
    # Q is the ground truth in base frame
    deltaX, Rdelta, tdelta = compute_icp_transformation(P_m, Q_m)

    print(f"\n✓ ICP Correction Transform (deltaX) in BASE frame:")
    print(f"  Rotation (Rdelta):\n{Rdelta}")
    print(f"  Translation (tdelta) in m: [{tdelta[0]:.6f}, {tdelta[1]:.6f}, {tdelta[2]:.6f}]")
    
    # Check rotation angle
    trace = np.trace(Rdelta)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"  Rotation angle: {angle_deg:.2f} degrees")

    # Compute corrected extrinsics using proper kinematic correction
    # For multiple poses, average the corrections
    def rotation_mean(Rs: list) -> np.ndarray:
        M = np.zeros((3, 3), dtype=float)
        for R in Rs:
            M += R
        U, _, Vt = np.linalg.svd(M)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    X1_list = []
    for T_bg in T_base2gripper_list:
        X1_i = compute_corrected_extrinsics(X0, deltaX, T_bg)
        X1_list.append(X1_i)

    # Average translations and rotations separately
    translations = np.array([X[:3, 3] for X in X1_list])
    t_mean = translations.mean(axis=0)
    t_std = translations.std(axis=0)
    rotations = [X[:3, :3] for X in X1_list]
    R_mean = rotation_mean(rotations)

    X1 = np.eye(4)
    X1[:3, :3] = R_mean
    X1[:3, 3] = t_mean

    print(f"\n✓ Corrected Extrinsics (X1) [averaged across {len(X1_list)} poses]:")
    print(X1)
    print(f"\nConsistency check - translation std dev across poses: [{t_std[0]:.6f}, {t_std[1]:.6f}, {t_std[2]:.6f}] m")
    if t_std.max() > 0.01:
        print(f"⚠️  Warning: High std dev indicates pose-dependent correction (may indicate wrong approach)")

    # Compare before and after
    print("\n" + "="*70)
    print("Comparison: Initial vs Corrected")
    print("="*70)
    print(f"\nInitial translation (m): [{X0[0,3]:.6f}, {X0[1,3]:.6f}, {X0[2,3]:.6f}]")
    print(f"Corrected translation (m): [{X1[0,3]:.6f}, {X1[1,3]:.6f}, {X1[2,3]:.6f}]")
    print(f"Change (m): [{X1[0,3]-X0[0,3]:.6f}, {X1[1,3]-X0[1,3]:.6f}, {X1[2,3]-X0[2,3]:.6f}]")

    # Compute error reduction
    errors_before = np.linalg.norm(P_m - Q_m, axis=1)

    # Apply deltaX to P (in base frame) to get corrected P
    P_m_hom = np.hstack([P_m, np.ones((P_m.shape[0], 1))])
    P_corrected_m = (deltaX @ P_m_hom.T).T[:, :3]
    errors_after = np.linalg.norm(P_corrected_m - Q_m, axis=1)

    print(f"\nError Statistics (m):")
    print(f"  Before - Mean: {np.mean(errors_before):.6f}, Std: {np.std(errors_before):.6f}, Max: {np.max(errors_before):.6f}")
    print(f"  After  - Mean: {np.mean(errors_after):.6f}, Std: {np.std(errors_after):.6f}, Max: {np.max(errors_after):.6f}")
    print(f"  Improvement: {(1 - np.mean(errors_after)/np.mean(errors_before))*100:.1f}% reduction in mean error")

    # Save corrected extrinsics
    np.save(args.output, X1)
    print(f"\n✓ Saved corrected extrinsics to: {args.output}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
