#!/usr/bin/env python3
"""
Advanced Hand-Eye Calibration with Data Filtering and Parameter Optimization

Based on the research paper:
"A Method to Improve the Accuracy of Hand-eye Calibration Based on Data Filtering and Parameters Optimization"
by Xining Cui and Junchao Shangguan (2024)

This method implements:
1. RANSAC-based data filtering to remove equations with large rotation axis errors
2. Closed-form two-stage initial calibration (Sarabandi method)
3. 3D spatial distance error minimization through iterative optimization

The key improvement over traditional methods is using 3D spatial error instead of
2D reprojection error for optimization, which better reflects real-world accuracy.

Usage:
    python CalibrateEyeInHandOptimized.py --poses-dir ./new_poses_folder \\
        --calibration ../output/orbbec_calibration.npz \\
        --ground-truth-corners ./ground_truth_corners.json \\
        --ransac-iterations 500 \\
        --optimization-iterations 200
"""

import cv2
import numpy as np
import os
import glob
import json
import argparse
import random
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as Rsc
from scipy.optimize import minimize
import time

# ChArUco board parameters
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
SQUARE_LEN_M = 0.03705
MARKER_LEN_M = SQUARE_LEN_M * 0.8
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250


def load_calibration(calib_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Load camera calibration from .npz or .json file"""
    if calib_path.endswith('.json'):
        with open(calib_path, 'r') as f:
            calib = json.load(f)

        if 'camera_matrix' in calib and 'dist_coeffs' in calib:
            K = np.array(calib['camera_matrix'], dtype=np.float64)
            D = np.array(calib['dist_coeffs'], dtype=np.float64).flatten()
        else:
            cam_node = None
            for v in calib.values():
                if isinstance(v, dict) and all(k in v for k in ('fx','fy','cx','cy')):
                    cam_node = v
                    break
            if cam_node is None:
                print(f"❌ Unsupported calibration JSON structure")
                return None, None, {}
            fx = float(cam_node['fx']); fy = float(cam_node['fy'])
            cx = float(cam_node['cx']); cy = float(cam_node['cy'])
            dist = cam_node.get('distortion', [0,0,0,0,0])
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            D = np.array(dist, dtype=np.float64).flatten()

        print(f"✓ Loaded calibration from: {calib_path}")
        return K, D, calib

    elif calib_path.endswith('.npz'):
        calib = np.load(calib_path)
        K = calib['camera_matrix']
        D = calib['dist_coeffs'].flatten()
        print(f"✓ Loaded calibration from: {calib_path}")
        return K, D, dict(calib)

    else:
        print(f"❌ Unsupported calibration file format")
        return None, None, {}


def make_charuco_board():
    """Create ChArUco board"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        aruco_dict
    )
    return aruco_dict, board


def make_aruco_detector():
    """Create ArucoDetector with optimized parameters"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(aruco_dict, params), aruco_dict


def estimate_charuco_pose(image_bgr: np.ndarray, K: np.ndarray, D: np.ndarray,
                          board, detector, min_charuco: int = 6) -> Tuple[Optional[Dict], str]:
    """Estimate ChArUco board pose"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, "No markers detected"

    if K is not None and D is not None:
        try:
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                image=gray, board=board, detectedCorners=corners,
                detectedIds=ids, rejectedCorners=rejected,
                cameraMatrix=K, distCoeffs=D
            )
        except Exception:
            pass

    response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners, markerIds=ids, image=gray, board=board
    )

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < max(min_charuco, 4):
        return None, f"Too few ChArUco corners: {0 if charuco_corners is None else len(charuco_corners)}"

    # Sub-pixel refinement
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), term_crit)

    # Estimate pose
    try:
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=charuco_corners, charucoIds=charuco_ids,
            board=board, cameraMatrix=K, distCoeffs=D,
            rvec=None, tvec=None
        )
    except TypeError:
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, D, None, None
        )

    if not pose_ok:
        return None, "estimatePoseCharucoBoard failed"

    # Use matchImagePoints for robust correspondence
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)

    # Refine with Levenberg-Marquardt
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
    except Exception:
        pass

    R_matrix, _ = cv2.Rodrigues(rvec)

    # Compute reprojection error
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    img_pts_2d = img_pts.reshape(-1, 2)
    reproj_err = float(cv2.norm(img_pts_2d, proj, cv2.NORM_L2) / len(img_pts_2d))

    result = {
        "rvec": rvec,
        "tvec": tvec.flatten(),
        "R_matrix": R_matrix,
        "reprojection_error_px": reproj_err,
        "num_charuco": int(len(charuco_ids))
    }

    return result, ""


def load_robot_pose(npy_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load robot pose from .npy file"""
    try:
        data = np.load(npy_path, allow_pickle=True).item()

        if "R_gripper2base" in data:
            R = data["R_gripper2base"]
            t = data["t_gripper2base"]
        elif "R" in data:
            R = data["R"]
            t = data["t"]
        else:
            print(f"⚠️  Unknown pose format in {npy_path}")
            return None

        return R, t

    except Exception as e:
        print(f"❌ Failed to load robot pose: {e}")
        return None


def to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform"""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_rotation_axis(R: np.ndarray) -> np.ndarray:
    """
    Compute equivalent rotation axis from rotation matrix using Rodrigues
    More stable than extracting from skew-symmetric part
    Returns normalized axis (or zero vector for identity)
    """
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()

    angle = np.linalg.norm(rvec)
    if angle > 1e-6:
        return rvec / angle  # Normalized axis
    else:
        return np.zeros(3)  # Identity rotation


def sarabandi_initial_solution(R_list_A: List[np.ndarray], R_list_B: List[np.ndarray],
                                t_list_A: List[np.ndarray], t_list_B: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form two-stage method (Sarabandi et al. 2022)
    Implements the axis-angle approach from the paper

    For n equations A_i @ X = X @ B_i:
    - Compute rotation axes a_i, b_i for all pairs
    - Solve: a_i = RX @ b_i using least squares
    - Project to SO(3)
    - Solve for translation
    """
    n = len(R_list_A)

    # Collect rotation axes
    axes_A = []
    axes_B = []

    for i in range(n):
        ai = compute_rotation_axis(R_list_A[i])
        bi = compute_rotation_axis(R_list_B[i])

        # Only use if rotation is significant
        if np.linalg.norm(ai) > 1e-6 and np.linalg.norm(bi) > 1e-6:
            axes_A.append(ai)
            axes_B.append(bi)

    if len(axes_A) < 2:
        # Fallback to OpenCV if insufficient data
        RX, tX = cv2.calibrateHandEye(
            R_gripper2base=R_list_A,
            t_gripper2base=[t.reshape(3, 1) for t in t_list_A],
            R_target2cam=R_list_B,
            t_target2cam=[t.reshape(3, 1) for t in t_list_B],
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        return RX, tX.flatten()

    # Build overconstrained system: A = RX @ B (Equation 11 from paper)
    A_matrix = np.column_stack(axes_A)  # 3 x m
    B_matrix = np.column_stack(axes_B)  # 3 x m

    # Least squares solution: RX = A @ B^+ (Equation 14)
    B_pinv = np.linalg.pinv(B_matrix)
    RX_approx = A_matrix @ B_pinv

    # Project to SO(3) using SVD
    U, _, Vt = np.linalg.svd(RX_approx)
    RX = U @ Vt

    # Ensure det(RX) = +1
    if np.linalg.det(RX) < 0:
        U[:, -1] *= -1
        RX = U @ Vt

    # Solve for translation using Equation 4: (R_A - I)tX = RX @ tB - tA
    A_t = []
    b_t = []

    for i in range(n):
        A_t.append(R_list_A[i] - np.eye(3))
        b_t.append(RX @ t_list_B[i] - t_list_A[i])

    A_t = np.vstack(A_t)
    b_t = np.hstack(b_t)

    tX, _, _, _ = np.linalg.lstsq(A_t, b_t, rcond=None)

    return RX, tX


def ransac_filter_calibration_data(R_list_A: List[np.ndarray], R_list_B: List[np.ndarray],
                                    t_list_A: List[np.ndarray], t_list_B: List[np.ndarray],
                                    max_iterations: int = 500,
                                    error_threshold: float = 0.008) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    RANSAC-based data filtering to remove outlier equations
    Returns: (inlier_indices, best_RX, best_tX)
    """
    n = len(R_list_A)
    best_inliers = []
    best_RX = None
    best_tX = None

    print(f"\n🔍 RANSAC filtering ({max_iterations} iterations, threshold={error_threshold:.4f})...")

    for iteration in range(max_iterations):
        # Randomly select 3 pairs
        if n < 3:
            print("⚠️  Need at least 3 samples for RANSAC")
            return list(range(n)), None, None

        sample_indices = random.sample(range(n), min(3, n))

        R_sample_A = [R_list_A[i] for i in sample_indices]
        R_sample_B = [R_list_B[i] for i in sample_indices]
        t_sample_A = [t_list_A[i] for i in sample_indices]
        t_sample_B = [t_list_B[i] for i in sample_indices]

        # Compute initial solution from sample
        try:
            temp_RX, temp_tX = sarabandi_initial_solution(R_sample_A, R_sample_B,
                                                          t_sample_A, t_sample_B)
        except Exception:
            continue

        # Compute rotation axis error for all relative motion equations
        # According to paper: for AX=XB, we have a_i = RX @ b_i (Equation 7)
        inliers = []
        for i in range(n):
            # Compute equivalent rotation axes for relative motions
            ai = compute_rotation_axis(R_list_A[i])
            bi = compute_rotation_axis(R_list_B[i])

            # Skip near-zero rotations (already normalized by compute_rotation_axis)
            if np.linalg.norm(ai) < 1e-6 or np.linalg.norm(bi) < 1e-6:
                continue

            # Predicted axis: ai_pred = RX @ bi
            ai_pred = temp_RX @ bi
            ai_pred_norm = np.linalg.norm(ai_pred)

            if ai_pred_norm < 1e-10:
                continue

            ai_pred = ai_pred / ai_pred_norm

            # Compute true angular error (in radians) using arccos(dot product)
            dot_product = np.clip(np.dot(ai, ai_pred), -1.0, 1.0)
            axis_error = np.arccos(dot_product)  # Radians

            if axis_error < error_threshold:
                inliers.append(i)

        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_RX = temp_RX
            best_tX = temp_tX

    print(f"  RANSAC found {len(best_inliers)}/{n} inliers")

    # Recompute solution using all inliers
    if len(best_inliers) >= 3:
        R_inlier_A = [R_list_A[i] for i in best_inliers]
        R_inlier_B = [R_list_B[i] for i in best_inliers]
        t_inlier_A = [t_list_A[i] for i in best_inliers]
        t_inlier_B = [t_list_B[i] for i in best_inliers]

        best_RX, best_tX = sarabandi_initial_solution(R_inlier_A, R_inlier_B,
                                                       t_inlier_A, t_inlier_B)

    return best_inliers, best_RX, best_tX


def optimize_spatial_distance_error(R_gripper2base_list: List[np.ndarray],
                                     t_gripper2base_list: List[np.ndarray],
                                     R_target2cam_list: List[np.ndarray],
                                     t_target2cam_list: List[np.ndarray],
                                     initial_RX: np.ndarray, initial_tX: np.ndarray,
                                     board_corners_3d: np.ndarray,
                                     known_board_pos_base: np.ndarray,
                                     max_iterations: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimize hand-eye calibration using 3D spatial distance error (Paper Section III.C)

    This is the key innovation: instead of 2D reprojection error, we minimize 3D spatial
    distance in the robot base frame using the full transformation chain.

    For each frame i and each board corner k:
        p_ik_base(X) = T_base_gripper_i @ T_gripper_cam(X) @ T_cam_board_i @ p_k_board

    Compare against ground truth:
        p_ik_base_true = T_base_board_true @ p_k_board

    Args:
        R_gripper2base_list: Robot poses (gripper→base) for each frame
        t_gripper2base_list: Robot translations for each frame
        R_target2cam_list: Detected board poses (board→cam) for each frame
        t_target2cam_list: Detected board translations for each frame
        initial_RX: Initial camera→gripper rotation
        initial_tX: Initial camera→gripper translation
        board_corners_3d: Nx3 array of board corner coordinates in board frame
        known_board_pos_base: 4x4 ground-truth board pose in base frame
        max_iterations: Maximum optimization iterations

    Returns:
        (RX_optimized, tX_optimized, final_error_meters)
    """
    n_frames = len(R_gripper2base_list)
    n_corners = len(board_corners_3d)

    print(f"\n🎯 Optimizing calibration parameters (max {max_iterations} iterations)...")
    print(f"  Using {n_frames} frames × {n_corners} corners = {n_frames * n_corners} 3D points")

    # Convert rotation to Rodrigues for optimization
    rvec_init, _ = cv2.Rodrigues(initial_RX)

    # Parameters: [rx, ry, rz, tx, ty, tz]
    x0 = np.hstack([rvec_init.flatten(), initial_tX])

    # Pre-compute ground-truth corner positions in base frame
    board_corners_hom = np.hstack([board_corners_3d, np.ones((n_corners, 1))])  # Nx4
    corners_true_base = (known_board_pos_base @ board_corners_hom.T).T[:, :3]  # Nx3

    def spatial_error_objective(x):
        """
        Compute mean 3D spatial error across all frames and corners
        """
        rvec = x[:3]
        tvec = x[3:6]

        R_cam2gripper, _ = cv2.Rodrigues(rvec)
        t_cam2gripper = tvec

        # Build T_cam2gripper
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper

        total_error = 0.0

        for i in range(n_frames):
            # Build T_gripper2base
            T_gripper2base = np.eye(4)
            T_gripper2base[:3, :3] = R_gripper2base_list[i]
            T_gripper2base[:3, 3] = t_gripper2base_list[i]

            # Build T_target2cam
            T_target2cam = np.eye(4)
            T_target2cam[:3, :3] = R_target2cam_list[i]
            T_target2cam[:3, 3] = t_target2cam_list[i]

            # Full chain: base ← gripper ← cam ← board
            T_board2base = T_gripper2base @ T_cam2gripper @ T_target2cam

            # Transform all corners
            corners_computed = (T_board2base @ board_corners_hom.T).T[:, :3]  # Nx3

            # Sum squared distances
            total_error += np.sum(np.linalg.norm(corners_computed - corners_true_base, axis=1)**2)

        # Return RMS error
        return np.sqrt(total_error / (n_frames * n_corners))

    # Set bounds: ±5 degrees on rotation (in radians), ±2cm on translation
    angle_bound = np.deg2rad(5)
    trans_bound = 0.02  # meters

    rvec_init_flat = rvec_init.flatten()
    bounds = [
        (rvec_init_flat[0] - angle_bound, rvec_init_flat[0] + angle_bound),
        (rvec_init_flat[1] - angle_bound, rvec_init_flat[1] + angle_bound),
        (rvec_init_flat[2] - angle_bound, rvec_init_flat[2] + angle_bound),
        (initial_tX[0] - trans_bound, initial_tX[0] + trans_bound),
        (initial_tX[1] - trans_bound, initial_tX[1] + trans_bound),
        (initial_tX[2] - trans_bound, initial_tX[2] + trans_bound),
    ]

    # Optimize
    start_time = time.time()
    result = minimize(
        spatial_error_objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': 1e-9}
    )
    optimization_time = time.time() - start_time

    # Extract result
    rvec_opt = result.x[:3]
    tvec_opt = result.x[3:6]

    RX_opt, _ = cv2.Rodrigues(rvec_opt)
    tX_opt = tvec_opt

    final_error = result.fun

    print(f"  ✓ Optimization completed in {optimization_time:.3f}s")
    print(f"  Initial RMS error: {spatial_error_objective(x0)*1000:.3f} mm")
    print(f"  Final RMS error: {final_error*1000:.3f} mm")
    print(f"  Iterations: {result.nit}")

    return RX_opt, tX_opt, final_error


def main():
    parser = argparse.ArgumentParser(
        description="Advanced hand-eye calibration with RANSAC filtering and 3D optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--poses-dir", nargs='+', default=["./new_poses_folder"],
                       help="One or more directories containing pose images and .npy files")
    parser.add_argument("--calibration", default="../output/allimage_calibration.npz",
                       help="Camera calibration file (.npz or .json)")
    parser.add_argument("--output", default="./calibrate_result/EyeInHand_optimized.npz",
                       help="Output file for optimized calibration")
    parser.add_argument("--min-charuco", type=int, default=20,
                       help="Minimum ChArUco corners required")
    parser.add_argument("--ransac-iterations", type=int, default=500,
                       help="RANSAC iterations for data filtering")
    parser.add_argument("--ransac-threshold", type=float, default=0.008,
                       help="RANSAC rotation axis error threshold")
    parser.add_argument("--optimization-iterations", type=int, default=200,
                       help="Maximum optimization iterations")
    parser.add_argument("--ground-truth-corners", type=str, default=None,
                       help="JSON file with ground-truth corner positions for optimization")

    args = parser.parse_args()

    print("="*80)
    print("Advanced Hand-Eye Calibration with Data Filtering & Parameter Optimization")
    print("Based on Cui & Shangguan (2024) research methodology")
    print("="*80)

    # Load camera calibration
    K, D, _ = load_calibration(args.calibration)
    if K is None:
        return

    print()

    # Create board and detector
    _, board = make_charuco_board()
    detector, _ = make_aruco_detector()

    # Handle multiple directories
    poses_dirs = args.poses_dir if isinstance(args.poses_dir, list) else [args.poses_dir]

    # Collect all pose pairs
    print(f"📁 Collecting calibration data from {len(poses_dirs)} director{'y' if len(poses_dirs) == 1 else 'ies'}:")

    all_image_files = []
    for poses_dir in poses_dirs:
        dir_images = sorted(glob.glob(os.path.join(poses_dir, "pose*.jpg")))
        print(f"  {poses_dir}: {len(dir_images)} images")
        all_image_files.extend([(img, poses_dir) for img in dir_images])

    print(f"\n📊 Total: {len(all_image_files)} pose images")

    # Process images and collect calibration equations
    R_list_A = []  # Relative robot motions
    R_list_B = []  # Relative camera motions
    t_list_A = []
    t_list_B = []

    R_gripper_list = []
    t_gripper_list = []
    R_target_list = []
    t_target_list = []

    print("\n🔍 Processing images and building calibration equations...")

    for i, (img_path, source_dir) in enumerate(all_image_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        npy_path = os.path.join(source_dir, f"{base_name}.npy")

        if not os.path.exists(npy_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        robot_pose = load_robot_pose(npy_path)
        if robot_pose is None:
            continue

        R_gripper, t_gripper = robot_pose

        result, error_msg = estimate_charuco_pose(img, K, D, board, detector, args.min_charuco)

        if result is None:
            continue

        R_target = result["R_matrix"]
        t_target = result["tvec"]

        R_gripper_list.append(R_gripper)
        t_gripper_list.append(t_gripper)
        R_target_list.append(R_target)
        t_target_list.append(t_target)

    print(f"✓ Collected {len(R_gripper_list)} valid pose pairs")

    if len(R_gripper_list) < 3:
        print("❌ Need at least 3 valid poses")
        return

    # Build relative motion equations: A_i @ X = X @ B_i
    for i in range(len(R_gripper_list)):
        for j in range(i + 1, len(R_gripper_list)):
            # A_ij = K_j @ K_i^-1
            R_A = R_gripper_list[j] @ R_gripper_list[i].T
            t_A = t_gripper_list[j] - R_A @ t_gripper_list[i]

            # B_ij = C_i^-1 @ C_j
            R_B = R_target_list[i].T @ R_target_list[j]
            t_B = R_target_list[i].T @ (t_target_list[j] - t_target_list[i])

            R_list_A.append(R_A)
            t_list_A.append(t_A)
            R_list_B.append(R_B)
            t_list_B.append(t_B)

    print(f"✓ Built {len(R_list_A)} calibration equations")

    # Step 1: RANSAC filtering
    inlier_indices, RX_ransac, tX_ransac = ransac_filter_calibration_data(
        R_list_A, R_list_B, t_list_A, t_list_B,
        max_iterations=args.ransac_iterations,
        error_threshold=args.ransac_threshold
    )

    if RX_ransac is None or len(inlier_indices) < 3:
        print("❌ RANSAC failed to find sufficient inliers")
        return

    # Map inlier relative-equation indices back to original absolute pose pairs
    # inlier_indices refers to equations in R_list_A/B (relative motions)
    # We need to find which ORIGINAL pose pairs (i,j) generated those equations

    # Build mapping: equation_idx -> (i, j) pair
    equation_to_pair = []
    for i in range(len(R_gripper_list)):
        for j in range(i + 1, len(R_gripper_list)):
            equation_to_pair.append((i, j))

    # Collect unique absolute pose indices that contributed to inlier equations
    used_pose_indices = set()
    for eq_idx in inlier_indices:
        i, j = equation_to_pair[eq_idx]
        used_pose_indices.add(i)
        used_pose_indices.add(j)

    # Use those absolute poses for OpenCV calibration
    pose_indices_sorted = sorted(list(used_pose_indices))

    print(f"\n🔧 Refining with OpenCV using {len(pose_indices_sorted)} absolute poses (from {len(inlier_indices)} inlier equations)...")

    R_filt_gripper = [R_gripper_list[i] for i in pose_indices_sorted]
    t_filt_gripper = [t_gripper_list[i] for i in pose_indices_sorted]
    R_filt_target = [R_target_list[i] for i in pose_indices_sorted]
    t_filt_target = [t_target_list[i] for i in pose_indices_sorted]

    RX_ransac, tX_ransac = cv2.calibrateHandEye(
        R_gripper2base=R_filt_gripper,
        t_gripper2base=[t.reshape(3,1) for t in t_filt_gripper],
        R_target2cam=R_filt_target,
        t_target2cam=[t.reshape(3,1) for t in t_filt_target],
        method=cv2.CALIB_HAND_EYE_PARK
    )
    tX_ransac = tX_ransac.flatten()

    # Step 2: Parameter optimization (if ground truth provided)
    if args.ground_truth_corners and os.path.exists(args.ground_truth_corners):
        print(f"\n📍 Loading ground-truth data from: {args.ground_truth_corners}")

        with open(args.ground_truth_corners, 'r') as f:
            gt_data = json.load(f)

        # Expected JSON format:
        # {
        #   "board_corners_3d": [[x1, y1, z1], [x2, y2, z2], ...],  # Nx3 in board frame
        #   "board_pose_base": [[r11, r12, r13, tx],                 # 4x4 matrix
        #                       [r21, r22, r23, ty],
        #                       [r31, r32, r33, tz],
        #                       [0, 0, 0, 1]]
        # }

        board_corners_3d = np.array(gt_data['board_corners_3d'])  # Nx3
        board_pose_base = np.array(gt_data['board_pose_base'])     # 4x4

        RX_final, tX_final, final_error = optimize_spatial_distance_error(
            R_gripper_list,  # Use ORIGINAL absolute poses, not relative equations
            t_gripper_list,
            R_target_list,
            t_target_list,
            RX_ransac, tX_ransac,
            board_corners_3d,
            board_pose_base,
            max_iterations=args.optimization_iterations
        )
    else:
        print("\n⚠️  No ground-truth data provided, using RANSAC result only")
        RX_final = RX_ransac
        tX_final = tX_ransac
        final_error = 0.0

    # Compute derived representations
    r = Rsc.from_matrix(RX_final)
    quat_xyzw = r.as_quat()
    rpy_zyx_deg = r.as_euler('zyx', degrees=True)

    R_gc = RX_final.T
    t_gc = -R_gc @ tX_final
    r_gc = Rsc.from_matrix(R_gc)
    quat_gc = r_gc.as_quat()
    rpy_gc_deg = r_gc.as_euler('zyx', degrees=True)

    print("\n" + "="*80)
    print("📐 Final Calibration Results:")
    print("="*80)
    print(f"\nCamera→Gripper Transformation:")
    print(f"  Translation (m): [{tX_final[0]:.6f}, {tX_final[1]:.6f}, {tX_final[2]:.6f}]")
    print(f"  Quaternion (xyzw): [{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, {quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]")
    print(f"  RPY ZYX (deg): [{rpy_zyx_deg[0]:.3f}, {rpy_zyx_deg[1]:.3f}, {rpy_zyx_deg[2]:.3f}]")

    print(f"\nGripper→Camera (inverse, for URDF):")
    print(f"  Translation (m): [{t_gc[0]:.6f}, {t_gc[1]:.6f}, {t_gc[2]:.6f}]")
    print(f"  Quaternion (xyzw): [{quat_gc[0]:.6f}, {quat_gc[1]:.6f}, {quat_gc[2]:.6f}, {quat_gc[3]:.6f}]")
    print(f"  RPY ZYX (deg): [{rpy_gc_deg[0]:.3f}, {rpy_gc_deg[1]:.3f}, {rpy_gc_deg[2]:.3f}]")

    if final_error > 0:
        print(f"\n✅ Final 3D spatial error: {final_error*1000:.3f} mm")

    # Save results
    output_data = {
        'R_cam2gripper': RX_final,
        't_cam2gripper': tX_final,
        'T_cam2gripper': to_homogeneous(RX_final, tX_final),
        'quat_cg_xyzw': quat_xyzw,
        'rpy_cg_zyx_deg': rpy_zyx_deg,
        'R_gc': R_gc,
        't_gc': t_gc,
        'T_gc': to_homogeneous(R_gc, t_gc),
        'quat_gc_xyzw': quat_gc,
        'rpy_gc_zyx_deg': rpy_gc_deg,
        'selected_method': 'OPTIMIZED_RANSAC_3D',
        'num_poses': len(inlier_indices),
        'spatial_error_mm': final_error * 1000,
        'camera_matrix': K,
        'dist_coeffs': D
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **output_data)

    print(f"\n✅ Saved optimized calibration to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
