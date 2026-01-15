#!/usr/bin/env python3
"""
Automated ICP Correction on Testing Poses

This script:
1. Loads all images and robot poses from testing_poses directory
2. Detects ChArUco board in each image
3. Computes predicted corner positions using initial calibration
4. Uses ICP to compute corrected camera-to-gripper transformation
5. Saves the corrected extrinsics
"""

import numpy as np
import cv2
import argparse
import os
import glob
import sys
from typing import Optional, Tuple, Dict, List

# Add parent directory to path to import ICP module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ICP functions
from ICP import (
    compute_icp_transformation,
    compute_corrected_extrinsics,
    analyze_point_quality,
    to_homogeneous,
    from_homogeneous
)

# Default marker parameters
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
SQUARE_LEN_M = 0.03718
MARKER_LEN_M = SQUARE_LEN_M * 0.80
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250


def default_detector_params():
    """Create detector parameters optimized for ChArUco detection"""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return p


def estimate_charuco_pose_solvepnp(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                                    board, aruco_dict, min_corners: int = 6) -> Optional[Dict]:
    """Estimate ChArUco board pose using solvePnP"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    params = default_detector_params()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Detect ChArUco corners
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_detector.setDetectorParameters(params)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    # Use matchImagePoints
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    if obj_points is None or len(obj_points) == 0:
        return None

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, D,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    # Refine
    rvec, tvec = cv2.solvePnPRefineLM(obj_points, img_points, K, D, rvec, tvec)

    # Convert to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    return {
        'rvec': rvec,
        'tvec': tvec.flatten(),
        'R_matrix': R_matrix,
        'num_corners': len(charuco_ids),
    }


def generate_charuco_corners_board_frame(squares_x: int, squares_y: int, square_len: float) -> np.ndarray:
    """
    Generate 24 internal ChArUco corner positions in board frame.
    Returns (24, 3) array in meters.
    """
    corners = []
    # Internal corners: (squares_x - 1) × (squares_y - 1) = 4 × 6 = 24
    for row in range(squares_y - 1):  # 6 rows
        for col in range(squares_x - 1):  # 4 columns
            x = col * square_len
            y = row * square_len
            corners.append([x, y, 0])
    return np.array(corners, dtype=float)


def load_test_data(testing_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load all images and robot poses from testing directory"""
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
            t = pose_data['t']
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
    parser = argparse.ArgumentParser(description="Automated ICP correction on testing poses")
    parser.add_argument("--testing-dir", default="../testing_poses",
                       help="Directory with test images and poses")
    parser.add_argument("--extrinsics", default="../test_extrinsics.npy",
                       help="Initial camera-to-gripper extrinsics file (.npy or .npz)")
    parser.add_argument("--intrinsics", default="../output/orbbec_calibration.npz",
                       help="Camera intrinsics file (.npz or .json)")
    parser.add_argument("--output", default="icp_corrected_extrinsics.npy",
                       help="Output corrected extrinsics file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results for each image")

    args = parser.parse_args()

    print("="*70)
    print("Automated ICP Correction on Testing Poses")
    print("="*70)
    print()

    # Load extrinsics (in meters)
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
        print("✓ Loaded initial camera-to-gripper extrinsics")
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

    # Setup ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        aruco_dict
    )

    # Ground truth corners (24 measured positions in mm) - from ICP.py
    Q_ground_truth_mm = np.array([
        [49.3, 526, -1.4],          # corner 1
        [48.8, 489.3, -2.5],        # corner 2
        [47.5, 452.3, -3.2],        # corner 3
        [47, 415.9, -3.8],          # corner 4
        [11.1, 527.40, -2.2],       # corner 5
        [11.3, 490.2, -2.6],        # corner 6
        [10.9, 453, -3],            # corner 7
        [9.3, 416.8, -4.1],         # corner 8
        [-25.6, 527.3, -2.6],       # corner 9
        [-25.8, 491.1, -3],         # corner 10
        [-26.1, 453.6, -3.4],       # corner 11
        [-27.2, 416.6, -4],         # corner 12
        [-62.5, 528.7, -2.4],       # corner 13
        [-62.8, 491.8, -2.9],       # corner 14
        [-63.2, 454.6, -3.6],       # corner 15
        [-63.9, 418, -4.2],         # corner 16
        [-99.1, 529.8, -2.5],       # corner 17
        [-99.8, 492.8, -3.2],       # corner 18
        [-100.6, 456.4, -4.3],      # corner 19
        [-101.4, 418.3, -4.5],      # corner 20
        [-136.8, 530.6, -2.5],      # corner 21
        [-137.2, 494.1, -3.2],      # corner 22
        [-137.7, 456.5, -3.8],      # corner 23
        [-138, 420.1, -4.5]         # corner 24
    ])

    # Generate board corners for detection/matching (24 internal corners)
    corners_board_frame = generate_charuco_corners_board_frame(CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y, SQUARE_LEN_M)

    print(f"\n✓ Using {len(Q_ground_truth_mm)} measured ground truth corner positions (in mm)")
    print(f"✓ Ground truth positions from ICP.py")

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
        pose_result = estimate_charuco_pose_solvepnp(img, K, D, board, aruco_dict)

        if pose_result is None:
            failed_count += 1
            if args.verbose:
                print(f"[{i+1}/{len(images)}] {filename}: ❌ Detection failed")
            continue

        # Board pose in camera frame (from detection)
        R_board2cam = pose_result['R_matrix']
        t_board2cam = pose_result['tvec']
        T_board2cam = to_homogeneous(R_board2cam, t_board2cam)

        # Predicted: Transform corners using INITIAL calibration X0
        # P = T_base2gripper @ X0 @ T_board2cam @ corners_board_frame
        T_board2base_predicted = T_base2gripper @ X0 @ T_board2cam

        # Transform all 24 corners to get predicted positions P
        for corner_board in corners_board_frame:
            corner_hom = np.append(corner_board, 1.0)

            # Predicted position in base frame (using initial X0)
            corner_predicted = (T_board2base_predicted @ corner_hom)[:3]
            P_list.append(corner_predicted)

        # Ground truth Q: use the 24 measured corner positions (same for all images)
        # Convert from mm to meters
        for q_corner_mm in Q_ground_truth_mm:
            Q_list.append(q_corner_mm / 1000.0)

        successful_count += 1
        if args.verbose:
            print(f"[{i+1}/{len(images)}] {filename}: ✓ {pose_result['num_corners']} corners")

    print(f"\n✓ Successfully processed: {successful_count}/{len(images)} images")
    print(f"✗ Failed detections: {failed_count}/{len(images)} images")
    print(f"✓ Total corner correspondences: {len(P_list)}")

    if len(P_list) < 4:
        print("\n❌ Not enough data for ICP! Need at least 4 point correspondences.")
        return

    # Convert to numpy arrays (in meters)
    P = np.array(P_list)  # meters
    Q = np.array(Q_list)  # meters
    
    # Convert to millimeters for display
    P_mm = P * 1000
    Q_mm = Q * 1000

    # Analyze point quality (convert to mm for display)
    print("\n" + "="*70)
    print("Analyzing point quality...")
    print("="*70)
    analyze_point_quality(P_mm, Q_mm)

    # Compute ICP transformation (in millimeters for consistency with ICP.py)
    print("\n" + "="*70)
    print("Computing ICP correction...")
    print("="*70)
    
    # Convert X0 to millimeters for ICP
    X0_mm = X0.copy()
    X0_mm[:3, 3] *= 1000
    
    deltaX, Rdelta, tdelta = compute_icp_transformation(P_mm, Q_mm)

    print(f"\n✓ ICP Correction Transform (deltaX):")
    print(f"  Rotation (Rdelta):\n{Rdelta}")
    print(f"  Translation (tdelta) in mm: [{tdelta[0]:.3f}, {tdelta[1]:.3f}, {tdelta[2]:.3f}]")


    # Compute corrected extrinsics per pose and average (more robust than single-pose)
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

    X1_list_mm = []
    for T_bg in T_base2gripper_list:
        T_bg_mm = T_bg.copy()
        T_bg_mm[:3, 3] *= 1000
        X1_i_mm = compute_corrected_extrinsics(X0_mm, deltaX, T_bg_mm)
        X1_list_mm.append(X1_i_mm)

    # Average translations and rotations separately
    translations = np.array([X[:3, 3] for X in X1_list_mm])
    t_mean_mm = translations.mean(axis=0)
    rotations = [X[:3, :3] for X in X1_list_mm]
    R_mean = rotation_mean(rotations)

    X1_mm = np.eye(4)
    X1_mm[:3, :3] = R_mean
    X1_mm[:3, 3] = t_mean_mm

    # Convert back to meters for robot use
    X1 = X1_mm.copy()
    X1[:3, 3] /= 1000

    print(f"\n✓ Corrected Extrinsics (X1) [averaged across {len(X1_list_mm)} poses]:")
    print(X1)

    # Compare before and after
    print("\n" + "="*70)
    print("Comparison: Initial vs Corrected")
    print("="*70)
    print(f"\nInitial translation (mm): [{X0[0,3]*1000:.3f}, {X0[1,3]*1000:.3f}, {X0[2,3]*1000:.3f}]")
    print(f"Corrected translation (mm): [{X1[0,3]*1000:.3f}, {X1[1,3]*1000:.3f}, {X1[2,3]*1000:.3f}]")
    print(f"Change (mm): [{(X1[0,3]-X0[0,3])*1000:.3f}, {(X1[1,3]-X0[1,3])*1000:.3f}, {(X1[2,3]-X0[2,3])*1000:.3f}]")

    # Compute error reduction
    errors_before = np.linalg.norm(P_mm - Q_mm, axis=1)
    
    # Apply deltaX to P (in mm, already in base frame)
    P_mm_hom = np.hstack([P_mm, np.ones((P_mm.shape[0], 1))])
    P_corrected_mm = (deltaX @ P_mm_hom.T).T[:, :3]
    errors_after = np.linalg.norm(P_corrected_mm - Q_mm, axis=1)

    print(f"\nError Statistics (mm):")
    print(f"  Before - Mean: {np.mean(errors_before):.2f}, Std: {np.std(errors_before):.2f}, Max: {np.max(errors_before):.2f}")
    print(f"  After  - Mean: {np.mean(errors_after):.2f}, Std: {np.std(errors_after):.2f}, Max: {np.max(errors_after):.2f}")
    print(f"  Improvement: {(1 - np.mean(errors_after)/np.mean(errors_before))*100:.1f}% reduction in mean error")

    # Save corrected extrinsics
    np.save(args.output, X1)
    print(f"\n✓ Saved corrected extrinsics to: {args.output}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
