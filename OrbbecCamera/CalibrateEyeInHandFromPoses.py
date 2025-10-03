#!/usr/bin/env python3
"""
Calibrate Eye-in-Hand from Pre-captured Poses

This script uses pre-captured image-pose pairs to perform eye-in-hand calibration.
It loads the camera intrinsics and estimates the ChArUco board pose for each image,
then performs hand-eye calibration.

Usage:
    python CalibrateEyeInHandFromPoses.py --poses-dir ../output/poses_orbbec --calibration ../output/orbbec_calibration.npz
"""

import cv2
import numpy as np
import os
import glob
import json
import argparse
from typing import Optional, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as Rsc


# ChArUco board parameters (must match your physical board!)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M = 0.03718      # square side length in meters
MARKER_LEN_M = SQUARE_LEN_M * 0.8  # marker side length in meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250


def load_calibration(calib_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Load camera calibration from .npz or .json file"""
    if calib_path.endswith('.json'):
        with open(calib_path, 'r') as f:
            calib = json.load(f)

        # Support two JSON formats:
        # 1) {"camera_matrix": [[...]], "dist_coeffs": [...], ...}
        # 2) {"<device>": {"fx":..., "fy":..., "cx":..., "cy":..., "width":..., "height":..., "distortion":[...]}}
        if 'camera_matrix' in calib and 'dist_coeffs' in calib:
            K = np.array(calib['camera_matrix'], dtype=np.float64)
            D = np.array(calib['dist_coeffs'], dtype=np.float64).flatten()
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
        else:
            # Try nested device dict
            cam_node = None
            for v in calib.values():
                if isinstance(v, dict) and all(k in v for k in ('fx','fy','cx','cy')):
                    cam_node = v
                    break
            if cam_node is None:
                print(f"❌ Unsupported calibration JSON structure: {calib_path}")
                return None, None, {}
            fx = float(cam_node['fx']); fy = float(cam_node['fy'])
            cx = float(cam_node['cx']); cy = float(cam_node['cy'])
            dist = cam_node.get('distortion', [0,0,0,0,0])
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            D = np.array(dist, dtype=np.float64).flatten()

        print(f"✓ Loaded calibration from: {calib_path}")
        print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
        if isinstance(calib, dict) and 'reprojection_error' in calib:
            print(f"  Reprojection error: {calib.get('reprojection_error')} pixels")

        return K, D, calib

    elif calib_path.endswith('.npz'):
        calib = np.load(calib_path)
        K = calib['camera_matrix']
        D = calib['dist_coeffs'].flatten()

        print(f"✓ Loaded calibration from: {calib_path}")
        print(f"  Camera matrix:\n{K}")
        print(f"  Distortion coeffs: {D}")

        return K, D, dict(calib)

    else:
        print(f"❌ Unsupported calibration file format: {calib_path}")
        return None, None, {}


def make_charuco_board():
    """Create ChArUco board with specified parameters"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        aruco_dict
    )
    return aruco_dict, board


def make_aruco_detector():
    """Create modern ArucoDetector with optimized parameters for accuracy"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    return cv2.aruco.ArucoDetector(aruco_dict, params), aruco_dict


def estimate_charuco_pose(image_bgr: np.ndarray, K: Optional[np.ndarray], D: Optional[np.ndarray],
                          board, detector, min_charuco: int = 6,
                          draw_debug: bool = False) -> Tuple[Optional[Dict], Optional[np.ndarray], str]:
    """
    Estimate pose of ChArUco board in the image using modern ArucoDetector.

    Returns:
        (result_dict, debug_image, error_message)
        result_dict contains: rvec, tvec, R_matrix, reprojection_error_px, num_charuco
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Detect ArUco markers using modern ArucoDetector
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, None, "No markers detected"

    # 2) Refine detected markers using the board model
    if K is not None and D is not None:
        try:
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                image=gray,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected,
                cameraMatrix=K,
                distCoeffs=D
            )
        except Exception:
            pass

    # 3) Interpolate Charuco corners
    response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < max(min_charuco, 4):
        return None, None, f"Too few ChArUco corners: {0 if charuco_corners is None else len(charuco_corners)}"

    # 3a) Sub-pixel refinement pass on charuco corners (2nd pass for extra precision)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    win = (5, 5)
    cv2.cornerSubPix(gray, charuco_corners, win, (-1, -1), term_crit)

    # 4) Estimate board pose using Charuco (classic API)
    # Some OpenCV builds require rvec/tvec placeholders (positional) in Python
    try:
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
            board=board,
            cameraMatrix=K,
            distCoeffs=D,
            rvec=None,
            tvec=None
        )
    except TypeError:
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, D, None, None
        )

    if not pose_ok:
        return None, None, "estimatePoseCharucoBoard failed"

    # Use matchImagePoints for robust correspondence (recommended by OpenCV docs)
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)

    # Refine pose with Levenberg-Marquardt (solvePnPRefineLM)
    try:
        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            obj_pts, img_pts, K, D, rvec, tvec
        )
        rvec = rvec_refined
        tvec = tvec_refined
    except Exception:
        # Fall back to non-refined pose if LM refinement fails
        pass

    # Convert rvec to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    # Compute reprojection error (following OpenCV's standard method)
    # error = cv.norm(imgpoints, imgpoints2, cv.NORM_L2) / len(imgpoints2)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    img_pts_2d = img_pts.reshape(-1, 2)
    reproj_err = float(cv2.norm(img_pts_2d, proj, cv2.NORM_L2) / len(img_pts_2d))

    # Draw debug visualization if requested
    debug_img = None
    if draw_debug:
        debug_img = image_bgr.copy()
        try:
            cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids, (0, 255, 0))
            axis_len = float(board.getSquareLength()) * 2.0
            cv2.drawFrameAxes(debug_img, K, D, rvec, tvec, axis_len)
        except Exception:
            pass

        # Add text info
        cv2.putText(debug_img, f"Corners: {len(charuco_ids)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Reproj err: {reproj_err:.2f}px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    result = {
        "rvec": rvec,
        "tvec": tvec.flatten(),
        "R_matrix": R_matrix,
        "reprojection_error_px": float(reproj_err),
        "num_charuco": int(len(charuco_ids))
    }

    return result, debug_img, ""


def load_robot_pose(npy_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load robot pose from .npy file"""
    try:
        data = np.load(npy_path, allow_pickle=True).item()

        # Check which format the data is in
        if "R_gripper2base" in data:
            R = data["R_gripper2base"]
            t = data["t_gripper2base"]
            print("Loaded from R_gripper2base")
        elif "R" in data:
            R = data["R"]
            t = data["t"]
            print("Loaded from R and t")
        else:
            print(f"⚠️  Unknown pose format in {npy_path}")
            return None

        return R, t

    except Exception as e:
        print(f"❌ Failed to load robot pose from {npy_path}: {e}")
        return None



def to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform T from rotation R (3x3) and translation t (3,) or (3,1)."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)  # accepts (3,), (3,1), or (1,3)

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


def main():
    parser = argparse.ArgumentParser(description="Eye-in-hand calibration from pre-captured poses")
    parser.add_argument("--poses-dir", default="./new_poses_folder",
                       help="Directory containing pose images and .npy files")
    parser.add_argument("--calibration", default="../output/orbbec_calibration_20251003_112009.json",
                       help="Camera calibration file (.npz or .json)")
    parser.add_argument("--output", default="./calibrate_result/EyeInHand.npz",
                       help="Output file for hand-eye calibration result")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization of detected boards")
    parser.add_argument("--save-vis", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--min-charuco", type=int, default=20,
                       help="Minimum number of ChArUco corners required per frame")
    parser.add_argument("--max-reproj", type=float, default=0.1,
                       help="Maximum reprojection error in pixels (frames above this are rejected)")
    parser.add_argument("--keep-top", type=float, default=0.98,
                       help="Keep best X fraction of frames by reprojection error (0.9 = top 90%%)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}\n")

    print("="*70)
    print("Eye-in-Hand Calibration from Pre-captured Poses")
    print("="*70)

    # Load camera calibration
    K, D, calib_info = load_calibration(args.calibration)
    if K is None:
        return

    print()

    # Create ChArUco board and detector
    _, board = make_charuco_board()
    detector, _ = make_aruco_detector()
    print(f"✓ Created ChArUco board: {CHARUCO_SQUARES_X}x{CHARUCO_SQUARES_Y}")
    print(f"  Square length: {SQUARE_LEN_M}m, Marker length: {MARKER_LEN_M}m")
    print(f"✓ Created ArucoDetector with sub-pixel refinement")
    print(f"  Quality filters: min_corners={args.min_charuco}, max_reproj={args.max_reproj}px, keep_top={args.keep_top*100:.0f}%")
    print()

    # Find all pose pairs
    image_files = sorted(glob.glob(os.path.join(args.poses_dir, "pose*.jpg")))
    print(f"Found {len(image_files)} pose images")

    # Lists to store valid calibration data
    R_gripper2base_list = []  # Robot gripper poses in base frame
    t_gripper2base_list = []
    R_target2cam_list = []    # ChArUco board poses in camera frame
    t_target2cam_list = []
    valid_pairs = []
    frame_quality = []  # Store (reproj_err, num_corners, idx) for quality filtering

    print("\n🔍 Processing images...")

    for i, img_path in enumerate(image_files):
        # Get corresponding .npy file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        npy_path = os.path.join(args.poses_dir, f"{base_name}.npy")

        if not os.path.exists(npy_path):
            print(f"  [{i+1}/{len(image_files)}] ⚠️  Missing pose file: {base_name}.npy")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{i+1}/{len(image_files)}] ❌ Failed to load image: {base_name}")
            continue

        # Load robot pose
        robot_pose = load_robot_pose(npy_path)
        if robot_pose is None:
            continue

        R_gripper2base, t_gripper2base = robot_pose

        # Estimate ChArUco board pose using modern detector
        result, debug_img, error_msg = estimate_charuco_pose(
            img, K, D, board, detector, min_charuco=args.min_charuco, draw_debug=args.visualize or args.save_vis
        )

        if result is None:
            print(f"  [{i+1}/{len(image_files)}] ✗ {base_name}: {error_msg}")

            # Show failed detection if visualizing
            if args.visualize:
                failed_img = img.copy()
                cv2.putText(failed_img, f"FAILED: {error_msg}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(failed_img, "Press any key to continue", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("ChArUco Detection", failed_img)
                cv2.waitKey(0)
            continue

        # Extract board pose
        R_target2cam = result["R_matrix"]
        t_target2cam = result["tvec"]
        reproj_err = result["reprojection_error_px"]
        num_corners = result["num_charuco"]

        # Store valid pair
        R_gripper2base_list.append(R_gripper2base)
        t_gripper2base_list.append(t_gripper2base)
        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(t_target2cam)
        valid_pairs.append(base_name)
        frame_quality.append((reproj_err, num_corners, len(valid_pairs) - 1))

        print(f"  [{i+1}/{len(image_files)}] ✓ {base_name}: {num_corners} corners, "
              f"reproj={reproj_err:.2f}px")

        # Visualize if requested (interactive mode - press key to advance)
        if args.visualize and debug_img is not None:
            cv2.putText(debug_img, f"[{i+1}/{len(image_files)}] Press any key to continue, 'q' to skip rest",
                       (10, debug_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("ChArUco Detection", debug_img)
            key = cv2.waitKey(0)  # Wait for key press
            if key == ord('q'):
                print("  Skipping remaining visualizations...")
                args.visualize = False  # Disable further visualization

        # Save visualization if requested
        if args.save_vis and debug_img is not None:
            vis_path = os.path.join(args.poses_dir, f"vis_{base_name}.jpg")
            cv2.imwrite(vis_path, debug_img)

    if args.visualize:
        cv2.destroyAllWindows()

    print(f"\n✓ Successfully processed {len(valid_pairs)} pose pairs")

    if len(valid_pairs) < 3:
        print("❌ Not enough valid pose pairs for calibration (need at least 3)")
        return

    # Apply quality filtering to keep only the best frames
    print("\n🔍 Applying quality filtering...")
    print(f"  Initial frames: {len(frame_quality)}")

    # Filter by max reprojection error and min corner count
    keep_idx = [i for err, ncorners, i in frame_quality
                if err <= args.max_reproj and ncorners >= args.min_charuco]

    # Sort by reprojection error (lower is better) and keep top percentage
    quality_sorted = [(frame_quality[i][0], frame_quality[i][1], i) for i in keep_idx]
    quality_sorted.sort(key=lambda x: (x[0], -x[1]))  # Sort by error asc, corners desc

    keep_count = max(3, int(args.keep_top * len(quality_sorted)))
    keep_idx = [idx for _, _, idx in quality_sorted[:keep_count]]
    keep_idx.sort()  # Restore original order

    if len(keep_idx) < 3:
        print(f"❌ After quality filtering, only {len(keep_idx)} frames remain (need at least 3)")
        print(f"   Try relaxing --max-reproj, --min-charuco, or --keep-top parameters")
        return

    # Filter all lists
    R_gripper2base_list = [R_gripper2base_list[i] for i in keep_idx]
    t_gripper2base_list = [t_gripper2base_list[i] for i in keep_idx]
    R_target2cam_list = [R_target2cam_list[i] for i in keep_idx]
    t_target2cam_list = [t_target2cam_list[i] for i in keep_idx]
    valid_pairs_filtered = [valid_pairs[i] for i in keep_idx]

    removed_count = len(valid_pairs) - len(valid_pairs_filtered)
    if removed_count > 0:
        print(f"  Removed {removed_count} low-quality frames")
        print(f"  Kept {len(valid_pairs_filtered)} high-quality frames for calibration")
        # Show quality stats
        kept_quality = [frame_quality[i] for i in keep_idx]
        reproj_errs = [err for err, _, _ in kept_quality]
        corner_counts = [nc for _, nc, _ in kept_quality]
        print(f"  Reproj error range: {min(reproj_errs):.2f} - {max(reproj_errs):.2f}px (median: {np.median(reproj_errs):.2f}px)")
        print(f"  Corner count range: {min(corner_counts)} - {max(corner_counts)} (median: {int(np.median(corner_counts))})")
    else:
        print(f"  All frames passed quality filter")
        valid_pairs_filtered = valid_pairs

    # Prepare data for calibrateHandEye
    print("\n🎯 Performing hand-eye calibration...")

    # Convert to list of 3x3 and 3x1 arrays as required by OpenCV
    Rg = [R for R in R_gripper2base_list]
    tg = [t.reshape(3, 1) for t in t_gripper2base_list]
    Rt = [R for R in R_target2cam_list]
    tt = [t.reshape(3, 1) for t in t_target2cam_list]

    # Try different calibration methods
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
    ]

    results = {}

    for method, name in methods:
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=Rg,
                t_gripper2base=tg,
                R_target2cam=Rt,
                t_target2cam=tt,
                method=method
            )

            results[name] = {
                'R_cam2gripper': R_cam2gripper,
                't_cam2gripper': t_cam2gripper.flatten(),
                'T_cam2gripper': to_homogeneous(R_cam2gripper, t_cam2gripper.flatten())
            }

            print(f"  Method {name}:")
            print(f"    Translation: {t_cam2gripper.flatten()}")

        except Exception as e:
            print(f"  Method {name}: FAILED ({e})")

    if not results:
        print("❌ All calibration methods failed!")
        return

    # Use PARK as default (or first available)
    selected_method = "PARK" if "PARK" in results else list(results.keys())[0]
    selected_result = results[selected_method]

    print(f"\n✅ Using {selected_method} method as default")
    print(f"\nCamera-to-Gripper Transformation:")
    print(f"  Translation (m): {selected_result['t_cam2gripper']}")
    print(f"  Rotation matrix:\n{selected_result['R_cam2gripper']}")

    def _to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return T

    # Compute derived representations (quaternion, RPY, inverse)
    R_cg = selected_result['R_cam2gripper']
    t_cg = selected_result['t_cam2gripper']

    # Quaternion (x, y, z, w) and ZYX RPY (deg)
    r = Rsc.from_matrix(R_cg)
    quat_xyzw = r.as_quat()  # x, y, z, w
    rpy_zyx_deg = r.as_euler('zyx', degrees=True)

    # Inverse (Gripper → Camera)
    R_gc = R_cg.T
    t_gc = -R_gc @ t_cg
    r_gc = Rsc.from_matrix(R_gc)
    quat_gc = r_gc.as_quat()
    rpy_gc_zyx_deg = r_gc.as_euler('zyx', degrees=True)

    print("\n📐 Derived representations:")
    print(f"  Camera→Gripper:")
    print(f"    Quaternion (x,y,z,w): [{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, {quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]")
    print(f"    RPY ZYX (deg): [{rpy_zyx_deg[0]:.3f}, {rpy_zyx_deg[1]:.3f}, {rpy_zyx_deg[2]:.3f}]")
    print(f"  Gripper→Camera (inverse, for URDF):")
    print(f"    Translation: [{t_gc[0]:.6f}, {t_gc[1]:.6f}, {t_gc[2]:.6f}]")
    print(f"    Quaternion (x,y,z,w): [{quat_gc[0]:.6f}, {quat_gc[1]:.6f}, {quat_gc[2]:.6f}, {quat_gc[3]:.6f}]")
    print(f"    RPY ZYX (deg): [{rpy_gc_zyx_deg[0]:.3f}, {rpy_gc_zyx_deg[1]:.3f}, {rpy_gc_zyx_deg[2]:.3f}]")

    # Store in result dict
    selected_result['quat_cg_xyzw'] = quat_xyzw
    selected_result['rpy_cg_zyx_deg'] = rpy_zyx_deg
    selected_result['R_gc'] = R_gc
    selected_result['t_gc'] = t_gc
    selected_result['T_gc'] = to_homogeneous(R_gc, t_gc)
    selected_result['quat_gc_xyzw'] = quat_gc
    selected_result['rpy_gc_zyx_deg'] = rpy_gc_zyx_deg

    # =========================
    # Board-constancy verification
    # =========================
    def _rotation_mean(rots: list) -> np.ndarray:
        # Robust mean rotation via SVD of sum of rotation matrices
        M = np.zeros((3, 3), dtype=np.float64)
        for R in rots:
            M += R
        U, _, Vt = np.linalg.svd(M)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
        R = Ra @ Rb.T
        tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(tr)))

    print("\n🔎 Board-constancy test (board static across samples):")
    T_cam2gripper = selected_result['T_cam2gripper']

    T_board2base_list = []
    for Rgb, tgb, Rtc, ttc in zip(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list):
        T_gb = _to_T(Rgb, tgb)
        T_tc = _to_T(Rtc, ttc)
        T_bb = T_gb @ T_cam2gripper @ T_tc
        T_board2base_list.append(T_bb)

    # Translation spread
    translations = np.array([T[:3, 3] for T in T_board2base_list])
    t_mean = translations.mean(axis=0)
    t_dev = translations - t_mean
    dists = np.linalg.norm(t_dev, axis=1)
    t_rms = float(np.sqrt(np.mean(dists ** 2)))
    t_max = float(np.max(dists))
    t_std_axis = translations.std(axis=0)

    # Rotation spread
    rotations = [T[:3, :3] for T in T_board2base_list]
    R_mean = _rotation_mean(rotations)
    angs = np.array([_angle_deg(R, R_mean) for R in rotations])
    ang_mean = float(np.mean(angs))
    ang_max = float(np.max(angs))

    print(f"  Samples: {len(T_board2base_list)}")
    print(f"  Mean position (m): [{t_mean[0]:.4f}, {t_mean[1]:.4f}, {t_mean[2]:.4f}]")
    print(f"  Position spread: RMS={t_rms*1000:.1f} mm, Max={t_max*1000:.1f} mm")
    print(f"  Axis std (mm): x={t_std_axis[0]*1000:.1f}, y={t_std_axis[1]*1000:.1f}, z={t_std_axis[2]*1000:.1f}")
    print(f"  Orientation spread: mean={ang_mean:.3f} deg, max={ang_max:.3f} deg")

    # =========================
    # Per-sample residual computation
    # =========================
    print("\n📊 Per-sample residuals (predicted vs measured board pose):")

    def _pose_error(T_pred: np.ndarray, T_meas: np.ndarray) -> Tuple[float, float]:
        """Compute position (mm) and orientation (deg) error between two poses"""
        Rp, tp = T_pred[:3, :3], T_pred[:3, 3]
        Rm, tm = T_meas[:3, :3], T_meas[:3, 3]
        pos_mm = float(np.linalg.norm(tp - tm) * 1000.0)
        dR = Rp @ Rm.T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        return pos_mm, ang

    res_pos = []
    res_ang = []

    # For each sample, compare predicted board→base (via kinematics) against the mean board position
    for i, (Rgb, tgb, Rtc, ttc) in enumerate(zip(R_gripper2base_list, t_gripper2base_list,
                                                    R_target2cam_list, t_target2cam_list)):
        T_gb = _to_T(Rgb, tgb)
        T_tc = _to_T(Rtc, ttc)
        T_pred = T_gb @ T_cam2gripper @ T_tc

        # Use mean board position as "ground truth"
        T_mean_board = np.eye(4, dtype=np.float64)
        T_mean_board[:3, :3] = R_mean
        T_mean_board[:3, 3] = t_mean

        pos_mm, ang_deg = _pose_error(T_pred, T_mean_board)
        res_pos.append(pos_mm)
        res_ang.append(ang_deg)

    if res_pos:
        res_pos_arr = np.array(res_pos)
        res_ang_arr = np.array(res_ang)

        print(f"  Position residuals (mm):")
        print(f"    Mean: {np.mean(res_pos_arr):.2f}, Median: {np.median(res_pos_arr):.2f}, "
              f"RMS: {np.sqrt(np.mean(res_pos_arr**2)):.2f}")
        print(f"    Min: {np.min(res_pos_arr):.2f}, Max: {np.max(res_pos_arr):.2f}, "
              f"P95: {np.percentile(res_pos_arr, 95):.2f}")

        print(f"  Orientation residuals (deg):")
        print(f"    Mean: {np.mean(res_ang_arr):.3f}, Median: {np.median(res_ang_arr):.3f}, "
              f"RMS: {np.sqrt(np.mean(res_ang_arr**2)):.3f}")
        print(f"    Min: {np.min(res_ang_arr):.3f}, Max: {np.max(res_ang_arr):.3f}, "
              f"P95: {np.percentile(res_ang_arr, 95):.3f}")

        # Store residuals
        selected_result['residuals_mm'] = res_pos_arr
        selected_result['residuals_deg'] = res_ang_arr

    # Save results (including all derived representations)
    output_data = {
        'R_cam2gripper': selected_result['R_cam2gripper'],
        't_cam2gripper': selected_result['t_cam2gripper'],
        'T_cam2gripper': selected_result['T_cam2gripper'],
        'quat_cg_xyzw': selected_result['quat_cg_xyzw'],
        'rpy_cg_zyx_deg': selected_result['rpy_cg_zyx_deg'],
        'R_gc': selected_result['R_gc'],
        't_gc': selected_result['t_gc'],
        'T_gc': selected_result['T_gc'],
        'quat_gc_xyzw': selected_result['quat_gc_xyzw'],
        'rpy_gc_zyx_deg': selected_result['rpy_gc_zyx_deg'],
        'residuals_mm': selected_result.get('residuals_mm', []),
        'residuals_deg': selected_result.get('residuals_deg', []),
        'selected_method': selected_method,
        'all_methods': results,
        'num_poses': len(valid_pairs_filtered),
        'valid_pairs': valid_pairs_filtered,
        'camera_matrix': K,
        'dist_coeffs': D,
        'charuco_board': {
            'squares_x': CHARUCO_SQUARES_X,
            'squares_y': CHARUCO_SQUARES_Y,
            'square_len_m': SQUARE_LEN_M,
            'marker_len_m': MARKER_LEN_M
        }
    }

    np.savez(args.output, **output_data)
    print(f"\n✓ Saved calibration result to: {args.output}")

    # Also save individual method results
    for method_name, result in results.items():
        method_output = args.output.replace('.npz', f'_{method_name.lower()}.npz')
        np.savez(method_output, **result)
        print(f"✓ Saved {method_name} result to: {method_output}")

    print("\n" + "="*70)
    print("✅ Eye-in-hand calibration complete!")
    print("="*70)


if __name__ == "__main__":
    main()
