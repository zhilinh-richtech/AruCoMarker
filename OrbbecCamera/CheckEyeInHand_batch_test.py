#!/usr/bin/env python3
"""
Batch Test Eye-in-Hand Calibration on Saved Images

This script:
1. Loads all images and robot poses from testing_poses directory
2. Detects ChArUco board in each image
3. Computes board position in base frame using eye-in-hand calibration
4. Compares against ground truth position
5. Reports average XYZ errors and distance errors
"""

import numpy as np
import cv2
import argparse
import os
import glob
from typing import Optional, Tuple, Dict, List


# Default marker parameters
CHARUCO_SQUARES_X = 14       # columns (X across)
CHARUCO_SQUARES_Y = 9       # rows    (Y down)
SQUARE_LEN_M = 0.040      # square side length in meters
MARKER_LEN_M = 0.030  # marker side length in meters (80% of square)
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_1000

# Ground truth board origin position in base frame (mm)
GROUND_TRUTH_MM = np.array([49.3, 526.0, -1.4])


def to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform T from rotation R (3x3) and translation t (3,) or (3,1)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)

    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"t must have 3 elements, got {t.shape}")

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def from_homogeneous(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract R, t from 4x4 homogeneous transformation matrix"""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def default_detector_params():
    """Create detector parameters optimized for ChArUco detection"""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return p


def estimate_charuco_pose_solvepnp(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                                    board, aruco_dict, min_corners: int = 6) -> Optional[Dict]:
    """
    Estimate ChArUco board pose using solvePnP with official matchImagePoints

    Returns:
        Dictionary with: rvec, tvec, R_matrix, num_corners, reprojection_error
    """
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

    # Use matchImagePoints to ensure correct correspondence
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    if obj_points is None or len(obj_points) == 0:
        return None

    # IPPE for initial pose (best for planar targets)
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, D,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    # Levenberg-Marquardt refinement
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points, img_points, K, D, rvec, tvec
    )

    # Convert to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    # Calculate reprojection error
    proj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    proj_points = proj_points.reshape(-1, 2)
    reproj_error = np.linalg.norm(img_points - proj_points, axis=1).mean()

    return {
        'rvec': rvec,
        'tvec': tvec.flatten(),
        'R_matrix': R_matrix,
        'num_corners': len(charuco_ids),
        'reprojection_error': reproj_error,
    }


def load_test_data(testing_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load all images and robot poses from testing directory

    Returns:
        images: List of images
        T_base2gripper_list: List of 4x4 transformation matrices
        filenames: List of base filenames (without extension)
    """
    pose_files = sorted(glob.glob(os.path.join(testing_dir, "pose*.npy")))

    images = []
    T_base2gripper_list = []
    filenames = []

    for pose_file in pose_files:
        # Load robot pose
        pose_data = np.load(pose_file, allow_pickle=True).item()

        # Build transformation matrix from R and t
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
    parser = argparse.ArgumentParser(description="Batch test eye-in-hand calibration on saved images")
    parser.add_argument("--testing-dir", default="./imagesJONATHAN",
                       help="Directory with test images and poses")
    parser.add_argument("--extrinsics", default="./test_extrinsics.npy",
                       help="Camera-to-gripper extrinsics file (.npy)")
    parser.add_argument("--intrinsics", default="./gemini_intrinsics/gemini_355_rgb_intrinsics_20250930_181519.json",
                       help="Camera intrinsics file (.npz)")
    parser.add_argument("--ground-truth", type=float, nargs=3,
                       default=[355.8, -288.6, -12.5],
                       help="Ground truth XYZ position in mm (default: 49.3 526.0 -1.4)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results for each image")

    args = parser.parse_args()

    print("="*70)
    print("Eye-in-Hand Calibration Batch Test")
    print("="*70)
    print()

    # Load extrinsics
    print(f"Loading extrinsics from: {args.extrinsics}")
    try:
        if args.extrinsics.endswith('.npy'):
            # Single .npy file containing 4x4 matrix
            T_cam2gripper = np.load(args.extrinsics)
            if T_cam2gripper.shape != (4, 4):
                print(f"❌ Extrinsics must be 4x4, got {T_cam2gripper.shape}")
                return
        elif args.extrinsics.endswith('.npz'):
            # .npz calibration file
            calib = np.load(args.extrinsics, allow_pickle=True)
            if 'T_cam2gripper' in calib:
                T_cam2gripper = calib['T_cam2gripper']
            else:
                # Build from R and t
                R = calib['R_cam2gripper']
                t = calib['t_cam2gripper']
                T_cam2gripper = to_homogeneous(R, t)
        else:
            print(f"❌ Unsupported extrinsics file format: {args.extrinsics}")
            return

        print("✓ Loaded camera-to-gripper extrinsics")
        if args.verbose:
            print(f"  T_cam2gripper:\n{T_cam2gripper}")
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

            # Try direct format first
            if 'camera_matrix' in intr:
                K = np.array(intr['camera_matrix'])
                D = np.array(intr['dist_coeffs']).flatten()
            else:
                # Try nested format (look for fx, fy, cx, cy)
                found = False
                for value in intr.values():
                    if isinstance(value, dict) and 'fx' in value:
                        fx, fy = value['fx'], value['fy']
                        cx, cy = value['cx'], value['cy']
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                        D = np.array(value.get('distortion', [0]*5)).flatten()
                        found = True
                        break
                if not found:
                    raise ValueError("Could not find intrinsics in JSON file")
        else:
            raise ValueError(f"Unsupported intrinsics file format: {args.intrinsics}")

        print("✓ Loaded camera intrinsics")
        if args.verbose:
            print(f"  K:\n{K}")
            print(f"  D: {D}")
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

    # Ground truth
    gt_mm = np.array(args.ground_truth)
    print(f"\n✓ Ground truth position: [{gt_mm[0]:.1f}, {gt_mm[1]:.1f}, {gt_mm[2]:.1f}] mm")

    # Process all images
    print("\n" + "="*70)
    print("Processing images...")
    print("="*70)

    errors_xyz = []  # List of [error_x, error_y, error_z] in mm
    errors_dist = []  # List of Euclidean distance errors in mm
    successful_detections = []
    failed_detections = []

    for i, (img, T_base2gripper, filename) in enumerate(zip(images, T_base2gripper_list, filenames)):
        # Detect ChArUco board
        pose_result = estimate_charuco_pose_solvepnp(img, K, D, board, aruco_dict)

        if pose_result is None:
            failed_detections.append(filename)
            if args.verbose:
                print(f"\n[{i+1}/{len(images)}] {filename}: ❌ Detection failed")
            continue

        # Get board pose in camera frame
        R_board2cam = pose_result['R_matrix']
        t_board2cam = pose_result['tvec']
        T_board2cam = to_homogeneous(R_board2cam, t_board2cam)

        # Transform to base frame: board_in_base = base2gripper @ cam2gripper @ board2cam
        T_board2base = T_base2gripper @ T_cam2gripper @ T_board2cam
        _, t_board2base = from_homogeneous(T_board2base)

        # Convert to mm
        predicted_mm = t_board2base * 1000.0

        # Calculate errors
        error_xyz = predicted_mm - gt_mm  # [dx, dy, dz] in mm
        error_dist = np.linalg.norm(error_xyz)  # Euclidean distance in mm

        errors_xyz.append(error_xyz)
        errors_dist.append(error_dist)
        successful_detections.append(filename)

        if args.verbose:
            print(f"\n[{i+1}/{len(images)}] {filename}:")
            print(f"  Corners detected: {pose_result['num_corners']}")
            print(f"  Reprojection error: {pose_result['reprojection_error']:.2f} px")
            print(f"  Predicted: [{predicted_mm[0]:7.2f}, {predicted_mm[1]:7.2f}, {predicted_mm[2]:7.2f}] mm")
            print(f"  Error XYZ: [{error_xyz[0]:7.2f}, {error_xyz[1]:7.2f}, {error_xyz[2]:7.2f}] mm")
            print(f"  Distance error: {error_dist:.2f} mm")

    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\nTotal images: {len(images)}")
    print(f"Successful detections: {len(successful_detections)}")
    print(f"Failed detections: {len(failed_detections)}")

    if failed_detections:
        print(f"\nFailed images: {', '.join(failed_detections)}")

    if len(errors_xyz) > 0:
        errors_xyz = np.array(errors_xyz)
        errors_dist = np.array(errors_dist)

        # Calculate statistics
        mean_xyz_error = np.mean(errors_xyz, axis=0)
        std_xyz_error = np.std(errors_xyz, axis=0)
        mean_dist_error = np.mean(errors_dist)
        std_dist_error = np.std(errors_dist)
        max_dist_error = np.max(errors_dist)
        min_dist_error = np.min(errors_dist)

        print(f"\nGround Truth: [{gt_mm[0]:.1f}, {gt_mm[1]:.1f}, {gt_mm[2]:.1f}] mm")
        print(f"\nAverage XYZ Error (mm):")
        print(f"  X: {mean_xyz_error[0]:7.2f} ± {std_xyz_error[0]:.2f}")
        print(f"  Y: {mean_xyz_error[1]:7.2f} ± {std_xyz_error[1]:.2f}")
        print(f"  Z: {mean_xyz_error[2]:7.2f} ± {std_xyz_error[2]:.2f}")

        print(f"\nDistance Error (mm):")
        print(f"  Mean:   {mean_dist_error:.2f} ± {std_dist_error:.2f}")
        print(f"  Min:    {min_dist_error:.2f}")
        print(f"  Max:    {max_dist_error:.2f}")

        print(f"\nRMS Error:")
        print(f"  XYZ RMS: [{np.sqrt(np.mean(errors_xyz[:, 0]**2)):.2f}, "
              f"{np.sqrt(np.mean(errors_xyz[:, 1]**2)):.2f}, "
              f"{np.sqrt(np.mean(errors_xyz[:, 2]**2)):.2f}] mm")
        print(f"  Distance RMS: {np.sqrt(np.mean(errors_dist**2)):.2f} mm")

    else:
        print("\n❌ No successful detections to analyze!")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
