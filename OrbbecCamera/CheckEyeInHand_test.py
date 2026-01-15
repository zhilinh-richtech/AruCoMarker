#!/usr/bin/env python3
"""
Check Eye-in-Hand Calibration with Comprehensive Reprojection Error Analysis

This script verifies the eye-in-hand calibration by:
1. Capturing multiple images with the Orbbec camera at different robot poses
2. Detecting ArUco marker or ChArUco board and estimating its pose using solvePnP
3. Computing comprehensive reprojection error across all collected samples
4. Visualizing the results

Usage:
    # ChArUco mode
    python CheckEyeInHand_test.py --calibration ./calibrate_result/EyeInHand.npz --mode charuco

    # ArUco mode
    python CheckEyeInHand_test.py --calibration ./calibrate_result/EyeInHand.npz --mode aruco --aruco-size 0.05
"""

import time
import cv2
import numpy as np
import argparse
import sys
import os
from typing import Optional, Tuple, Dict, List
from xarm.wrapper import XArmAPI
from pyorbbecsdk import Pipeline, Context, Config, OBStreamType, OBFormat, OBSensorType


# Default marker parameters
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M = 0.03714      # square side length in meters
MARKER_LEN_M = SQUARE_LEN_M * 0.80  # marker side length in meters (80% of square)
ARUCO_SIZE_M = 0.0500         # single ArUco marker size in meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250

# Robot and camera settings
XARM_IP = "192.168.10.202"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30


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


def load_eye_in_hand_calibration(calib_path: str) -> Optional[Dict]:
    """Load eye-in-hand calibration results"""
    try:
        calib = np.load(calib_path, allow_pickle=True)

        result = {
            'R_cam2gripper': calib['R_cam2gripper'],
            't_cam2gripper': calib['t_cam2gripper'],
            'T_cam2gripper': calib['T_cam2gripper'],
            'camera_matrix': calib['camera_matrix'],
            'dist_coeffs': calib['dist_coeffs'],
            'selected_method': str(calib.get('selected_method', 'unknown'))
        }

        # Optionally load all methods if present
        if 'all_methods' in calib:
            try:
                # np.load returns object array for dict; convert to native dict
                all_methods_obj = calib['all_methods'].item() if hasattr(calib['all_methods'], 'item') else calib['all_methods']
                result['all_methods'] = all_methods_obj
            except Exception:
                pass

        print(f" Loaded eye-in-hand calibration from: {calib_path}")
        print(f"  Method: {result['selected_method']}")
        print(f"  Camera-to-Gripper translation: {result['t_cam2gripper']}")
        print(f"  Camera-to-Gripper rotation:\n{result['R_cam2gripper']}")

        return result

    except Exception as e:
        print(f"L Failed to load calibration: {e}")
        return None


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


def default_detector_params():
    """Create detector parameters optimized for accuracy"""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.0100
    return p


class OrbbecCamera:
    """Orbbec camera interface using pyorbbecsdk (matches OrbbecTakePicture.py style)."""
    def __init__(self, width=1920, height=1080, fps=30, camera_kind: str = "orbbec"):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None

    def initialize(self) -> bool:
        try:
            print("Initializing Orbbec camera...")
            self.pipeline = Pipeline()

            # Ensure a device is present
            device_list = Context().query_devices()
            if len(device_list) == 0:
                print("L No Orbbec devices found!")
                return False

            # Get available color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile_count = profile_list.get_count()

            # Prefer exact 1920x1080 MJPG; fall back to first available MJPG, else any profile
            selected_profile = None
            fallback_profile = None

            for i in range(profile_count):
                profile = profile_list.get_stream_profile_by_index(i)
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    w = vp.get_width(); h = vp.get_height(); fmt = vp.get_format(); f = vp.get_fps()

                    # Exact match MJPG first
                    if (w == self.width and h == self.height and fmt == OBFormat.MJPG):
                        selected_profile = profile
                        break

                    # Track best MJPG as fallback
                    if fmt == OBFormat.MJPG and fallback_profile is None:
                        fallback_profile = profile

            if selected_profile is None and fallback_profile is not None:
                selected_profile = fallback_profile

            if selected_profile is None:
                # Final fallback to first available profile
                selected_profile = profile_list.get_stream_profile_by_index(0)

            self.config = Config()
            self.config.enable_stream(selected_profile)
            self.pipeline.start(self.config)

            print(" Camera initialized successfully!")
            return True

        except Exception as e:
            print(f"L Failed to initialize Orbbec camera: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            frames = self.pipeline.wait_for_frames(3000)
            if frames is None:
                return None
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None

            color_data = np.asanyarray(color_frame.get_data())
            if color_data is None or color_data.size == 0:
                return None

            width = color_frame.get_width()
            height = color_frame.get_height()

            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if color_image is None:
                    return None
            else:
                # Raw formats RGB/BGR
                if len(color_data.shape) == 1:
                    expected_size = width * height * 3
                    if color_data.size == expected_size:
                        color_image = color_data.reshape((height, width, 3))
                    else:
                        return None
                else:
                    color_image = color_data

                # Convert RGB to BGR for OpenCV visualization
                if color_frame.get_format() == OBFormat.RGB:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            return color_image

        except Exception:
            return None

    def close(self):
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass


def estimate_aruco_pose_solvepnp(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                                 aruco_dict, marker_size_m: float) -> Optional[Dict]:
    """
    Estimate single ArUco marker pose using solvePnP with IPPE and refinement

    Args:
        image: Input image
        K: Camera intrinsic matrix
        D: Distortion coefficients
        aruco_dict: ArUco dictionary
        marker_size_m: Physical size of the marker in meters

    Returns:
        Dictionary with: rvec, tvec, R_matrix, marker_id, reprojection_error, obj_points, img_points
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers with subpixel refinement
    params = default_detector_params()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Use the first detected marker
    marker_corners = corners[0].reshape(-1, 2)
    marker_id = ids[0][0]

    # Define 3D object points for a square marker (origin at center)
    half_size = marker_size_m / 2.0
    obj_points = np.array([
        [-half_size, half_size, 0],   # Top-left
        [half_size, half_size, 0],    # Top-right
        [half_size, -half_size, 0],   # Bottom-right
        [-half_size, -half_size, 0]   # Bottom-left
    ], dtype=np.float64)

    # Solve PnP with IPPE
    success, rvec, tvec = cv2.solvePnP(
        obj_points, marker_corners, K, D,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    # Iterative refinement using Levenberg-Marquardt
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points, marker_corners, K, D, rvec, tvec
    )

    # Convert to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    # Calculate reprojection error
    proj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    proj_points = proj_points.reshape(-1, 2)
    reproj_error = cv2.norm(marker_corners, proj_points, cv2.NORM_L2) / len(proj_points)

    return {
        'rvec': rvec,
        'tvec': tvec.flatten(),
        'R_matrix': R_matrix,
        'marker_id': marker_id,
        'marker_size': marker_size_m,
        'reprojection_error': reproj_error,
        'marker_corners': corners,
        'marker_ids': ids,
        'obj_points': obj_points,
        'img_points': marker_corners,
        'mode': 'aruco'
    }


def estimate_charuco_pose_solvepnp(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                                    board, aruco_dict, min_corners: int = 6) -> Optional[Dict]:
    """
    Estimate ChArUco board pose using solvePnP with official matchImagePoints

    Returns:
        Dictionary with: rvec, tvec, R_matrix, num_corners, reprojection_error, obj_points, img_points
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

    # OFFICIAL PATH: Use matchImagePoints to ensure correct correspondence
    # This avoids subtle corner-indexing mismatches
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

    # Levenberg-Marquardt refinement (same as ArUco)
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points, img_points, K, D, rvec, tvec
    )

    # Convert to rotation matrix
    R_matrix, _ = cv2.Rodrigues(rvec)

    # Calculate reprojection error
    proj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    proj_points = proj_points.reshape(-1, 2)
    reproj_error = cv2.norm(img_points, proj_points, cv2.NORM_L2) / len(proj_points)

    return {
        'rvec': rvec,
        'tvec': tvec.flatten(),
        'R_matrix': R_matrix,
        'num_corners': len(charuco_ids),
        'reprojection_error': reproj_error,
        'charuco_corners': charuco_corners,
        'charuco_ids': charuco_ids,
        'marker_corners': corners,
        'marker_ids': ids,
        'obj_points': obj_points,
        'img_points': img_points,
        'mode': 'charuco'
    }


def get_robot_pose(xarm: XArmAPI) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get current robot gripper pose in base frame"""
    try:
        code, pos = xarm.get_position(is_radian=False)

        if code != 0:
            print(f"   xArm get_position failed, code={code}")
            return None

        x, y, z, roll, pitch, yaw = pos

        # Convert to meters
        t = np.array([x, y, z], dtype=np.float64) / 1000.0

        # Convert Euler angles to rotation matrix (degrees)
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)

        # ZYX Euler
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])

        R = Rz @ Ry @ Rx
        return R, t

    except Exception as e:
        print(f"L Error getting robot pose: {e}")
        return None


def compute_comprehensive_reprojection_error(samples: List[Dict], K: np.ndarray, D: np.ndarray) -> Dict:
    """
    Compute comprehensive reprojection error across all samples
    Using the method: mean_error = sum(error_i) / len(samples)
    where error_i = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)

    Args:
        samples: List of sample dictionaries containing 'obj_points', 'img_points', 'rvec', 'tvec'
        K: Camera matrix
        D: Distortion coefficients

    Returns:
        Dictionary with mean_error, per_sample_errors, total_points
    """
    mean_error = 0.0
    per_sample_errors = []
    total_points = 0

    for i, sample in enumerate(samples):
        obj_points = sample['obj_points']
        img_points = sample['img_points']
        rvec = sample['rvec']
        tvec = sample['tvec']

        # Project 3D points to image plane
        imgpoints2, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        imgpoints2 = imgpoints2.reshape(-1, 2)

        # Compute error for this sample using cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        error = cv2.norm(img_points, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_sample_errors.append(error)
        mean_error += error
        total_points += len(img_points)

    if len(samples) > 0:
        mean_error = mean_error / len(samples)

    return {
        'mean_error': mean_error,
        'per_sample_errors': per_sample_errors,
        'total_points': total_points,
        'num_samples': len(samples)
    }


def visualize_result(image: np.ndarray, pose_result: Dict, K: np.ndarray, D: np.ndarray,
                    marker_in_base: Optional[np.ndarray] = None, axis_len: float = 0.0500,
                    sample_idx: Optional[int] = None) -> np.ndarray:
    """Draw visualization on image"""
    vis = image.copy()
    mode = pose_result.get('mode', 'charuco')

    # Draw detected markers
    if pose_result['marker_corners'] is not None:
        cv2.aruco.drawDetectedMarkers(vis, pose_result['marker_corners'],
                                      pose_result['marker_ids'])

    # Draw ChArUco corners if present
    if 'charuco_corners' in pose_result and pose_result['charuco_corners'] is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, pose_result['charuco_corners'],
                                            pose_result['charuco_ids'], (0, 255, 0))

    # Draw coordinate axes
    cv2.drawFrameAxes(vis, K, D, pose_result['rvec'],
                     pose_result['tvec'].reshape(3, 1), axis_len, 3)

    # Add text information
    y_offset = 30

    if sample_idx is not None:
        cv2.putText(vis, f"Sample #{sample_idx + 1}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 35

    if mode == 'aruco':
        cv2.putText(vis, f"ArUco ID: {pose_result['marker_id']} | Size: {pose_result['marker_size']*1000:.1f}mm",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        y_offset += 35
    else:
        cv2.putText(vis, f"ChArUco corners: {pose_result['num_corners']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 35

    cv2.putText(vis, f"Reproj error: {pose_result['reprojection_error']:.4f}px", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_offset += 35

    t = pose_result['tvec']
    label = "Marker" if mode == 'aruco' else "Board"
    cv2.putText(vis, f"{label} in camera: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 30

    if marker_in_base is not None:
        cv2.putText(vis, f"{label} in base: [{marker_in_base[0]:.3f}, {marker_in_base[1]:.3f}, {marker_in_base[2]:.3f}]m",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Check eye-in-hand calibration with reprojection error analysis")
    parser.add_argument("--calibration", default="./calibrate_result/EyeInHand.npz",
                       help="Eye-in-hand calibration file")
    parser.add_argument("--xarm-ip", default=XARM_IP,
                       help="xArm IP address")
    parser.add_argument("--camera-kind", default="orbbec",
                       choices=["auto", "orbbec", "realsense", "opencv", "usb", "uvc"],
                       help="Camera backend to use")
    parser.add_argument("--mode", type=str, choices=['aruco', 'charuco'], default='charuco',
                       help="Detection mode: aruco (single marker) or charuco (board)")
    parser.add_argument("--aruco-size", type=float, default=ARUCO_SIZE_M,
                       help="ArUco marker size in meters (for aruco mode)")
    parser.add_argument("--squares-x", type=int, default=CHARUCO_SQUARES_X,
                       help="ChArUco board squares in X (for charuco mode)")
    parser.add_argument("--squares-y", type=int, default=CHARUCO_SQUARES_Y,
                       help="ChArUco board squares in Y (for charuco mode)")
    parser.add_argument("--square-len", type=float, default=SQUARE_LEN_M,
                       help="ChArUco square length in meters (for charuco mode)")
    parser.add_argument("--marker-len", type=float, default=MARKER_LEN_M,
                       help="ChArUco marker length in meters (for charuco mode)")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum number of samples to collect before computing comprehensive error")

    args = parser.parse_args()

    print("="*70)
    print("Check Eye-in-Hand Calibration with Reprojection Error Analysis")
    print("="*70)
    print()

    # Load calibration
    calib = load_eye_in_hand_calibration(args.calibration)
    if calib is None:
        return

    # Extract calibration data
    K = calib['camera_matrix']
    D = calib['dist_coeffs']
    print("\nCamera Matrix:")
    print(K)
    print("\nDistortion Coefficients:")
    print(D)

    # Collect extrinsics to evaluate: selected plus any others available
    extrinsics_list = []
    extrinsics_list.append((calib.get('selected_method', 'selected'), calib['T_cam2gripper']))
    if 'all_methods' in calib:
        try:
            for name, res in calib['all_methods'].items():
                if isinstance(res, dict) and 'T_cam2gripper' in res:
                    extrinsics_list.append((name, res['T_cam2gripper']))
        except Exception:
            pass

    # Create ArUco dictionary and board based on mode
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = None

    if args.mode == 'charuco':
        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_len,
            args.marker_len,
            aruco_dict
        )
        print(f"\n Mode: ChArUco Board {args.squares_x}x{args.squares_y}")
        print(f"  Square: {args.square_len*1000:.1f}mm, Marker: {args.marker_len*1000:.1f}mm")
    else:
        print(f"\n Mode: Single ArUco Marker")
        print(f"  Marker size: {args.aruco_size*1000:.1f}mm")
    print()

    # Initialize camera
    camera = OrbbecCamera(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, camera_kind=args.camera_kind)
    if not camera.initialize():
        return
    print()

    # Defer xArm connection until needed to avoid blocking window creation
    xarm = None

    # Storage for collected samples
    collected_samples = []

    print("="*70)
    print(f"Live View - Press 's' to capture sample (need {args.min_samples} minimum)")
    print("Press 'c' to compute reprojection error, 'q' to quit")
    print("="*70)

    try:
        # Create a resizable window before the loop to ensure it shows up
        cv2.namedWindow("Check Eye-in-Hand", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Check Eye-in-Hand", 1280, 720)

        while True:
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                # Show a waiting screen so the window is still visible
                preview = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(preview, "Waiting for camera...", (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(preview, "Press 'q' to quit", (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Check Eye-in-Hand", preview)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                time.sleep(0.05)
                continue

            # Show preview
            preview = frame.copy()

            # Show sample count
            cv2.putText(preview, f"Samples collected: {len(collected_samples)}/{args.min_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(preview, "Press 's' to capture, 'c' to compute error, 'q' to quit",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Check Eye-in-Hand", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "-"*70)
                print(f"Capturing sample #{len(collected_samples) + 1}...")

                # Estimate marker/board pose in camera frame using solvePnP
                if args.mode == 'aruco':
                    pose_result = estimate_aruco_pose_solvepnp(frame, K, D, aruco_dict, args.aruco_size)
                else:
                    pose_result = estimate_charuco_pose_solvepnp(frame, K, D, board, aruco_dict)

                if pose_result is None:
                    error_msg = "Failed to detect ArUco marker" if args.mode == 'aruco' else "Failed to detect ChArUco board"
                    print(f"L {error_msg}")
                    cv2.putText(frame, f"FAILED: No {args.mode} detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Check Eye-in-Hand", frame)
                    cv2.waitKey(2000)
                    continue

                if args.mode == 'aruco':
                    print(f" Detected ArUco marker ID {pose_result['marker_id']}")
                else:
                    print(f" Detected ChArUco board with {pose_result['num_corners']} corners")
                print(f"  Reprojection error: {pose_result['reprojection_error']:.4f} pixels")

                # Connect to robot on-demand (first capture)
                if xarm is None:
                    try:
                        print(f"Connecting to xArm at {args.xarm_ip}...")
                        xarm = XArmAPI(args.xarm_ip)
                        xarm.connect()
                        print(" xArm connected")
                    except Exception as e:
                        print(f"L Failed to connect to xArm: {e}")
                        continue

                # Get robot pose
                robot_pose = get_robot_pose(xarm)
                if robot_pose is None:
                    print("L Failed to get robot pose")
                    continue

                R_gripper2base, t_gripper2base = robot_pose
                T_gripper2base = to_homogeneous(R_gripper2base, t_gripper2base)

                print(f" Got robot pose")
                print(f"  Gripper in base: {t_gripper2base}")

                # Store sample
                sample = {
                    'obj_points': pose_result['obj_points'],
                    'img_points': pose_result['img_points'],
                    'rvec': pose_result['rvec'],
                    'tvec': pose_result['tvec'],
                    'R_matrix': pose_result['R_matrix'],
                    'T_gripper2base': T_gripper2base,
                    'image': frame.copy(),
                    'pose_result': pose_result
                }
                collected_samples.append(sample)

                # Visualize
                R_marker2cam = pose_result['R_matrix']
                t_marker2cam = pose_result['tvec']
                T_marker2cam = to_homogeneous(R_marker2cam, t_marker2cam)

                T_marker2base = T_gripper2base @ extrinsics_list[0][1] @ T_marker2cam
                _, t_marker2base = from_homogeneous(T_marker2base)

                axis_len = args.aruco_size if args.mode == 'aruco' else float(board.getSquareLength()) * 2.0
                vis = visualize_result(frame, pose_result, K, D, t_marker2base, axis_len, len(collected_samples) - 1)
                cv2.putText(vis, "Press any key to continue", (10, vis.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Check Eye-in-Hand", vis)
                cv2.waitKey(0)

                print(f" Sample #{len(collected_samples)} collected")
                print("-"*70)

            elif key == ord('c'):
                if len(collected_samples) < args.min_samples:
                    print(f"\n   Need at least {args.min_samples} samples, currently have {len(collected_samples)}")
                    continue

                print("\n" + "="*70)
                print("Computing comprehensive reprojection error...")
                print("="*70)

                # Compute comprehensive reprojection error
                error_stats = compute_comprehensive_reprojection_error(collected_samples, K, D)

                print(f"\n=Ê Comprehensive Reprojection Error Analysis:")
                print(f"  Total samples: {error_stats['num_samples']}")
                print(f"  Total points: {error_stats['total_points']}")
                print(f"  Mean error across all samples: {error_stats['mean_error']:.6f} pixels")
                print(f"\n  Per-sample errors:")
                for i, err in enumerate(error_stats['per_sample_errors']):
                    print(f"    Sample #{i+1}: {err:.6f} pixels")

                print("\n" + "="*70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        camera.close()
        try:
            if xarm is not None:
                xarm.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("\n Done")


if __name__ == "__main__":
    main()
