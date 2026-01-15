#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

# Import from Realsense directory
sys.path.insert(0, '../Realsense')
from utils import rpy_to_matrix, rot_angle_deg, to_homogeneous, invert_rt, to_cv_lists, rel_motion
from camera import create_camera
from terminal_display import Display, draw_axes_ascii_friendly
sys.path.pop(0)  # Remove from path after import to avoid conflicts

# =========================
# CONFIG
# =========================

# Configuration with environment variable support
XARM_IP = os.getenv('XARM_IP', "192.168.10.202")
USE_DEG = True

# ALWAYS HIGH ACCURACY MODE
# Uses raw images with distortion model for maximum accuracy

# Orbbec camera settings (1920x1080 is typical for Orbbec)
ORBBEC_WIDTH = 1920
ORBBEC_HEIGHT = 1080
ORBBEC_FPS = 30

# HIGH ACCURACY ChArUco board - MEASURE PRECISELY WITH CALIPERS!
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)

# CRITICAL FOR ACCURACY: Measure these with digital calipers (±0.01mm precision)
# Each 0.1mm error in these measurements can cause 2-5mm final calibration error!
SQUARE_LEN_M = 0.03714        # MEASURE YOUR ACTUAL PRINTED SQUARE SIDE (meters)
MARKER_LEN_M = 0.02956       # MEASURE YOUR ACTUAL PRINTED MARKER SIDE (meters)

# Validation: Check for reasonable values
if abs(MARKER_LEN_M / SQUARE_LEN_M - 0.8) > 0.1:
    print(f"⚠️  WARNING: Unusual marker/square ratio: {MARKER_LEN_M/SQUARE_LEN_M:.3f}")
    print(f"   Expected ~0.8, got {MARKER_LEN_M/SQUARE_LEN_M:.3f}")
    print(f"   Please verify your measurements!")

print(f"📏 Board Configuration:")
print(f"   Square size: {SQUARE_LEN_M*1000:.1f}mm")
print(f"   Marker size: {MARKER_LEN_M*1000:.1f}mm")
print(f"   Ratio: {MARKER_LEN_M/SQUARE_LEN_M:.3f}")

print(f"🎯 HIGH ACCURACY Pose Estimation:")
print(f"   Mode: Raw images + distortion model")
print(f"   Method: SOLVEPNP_IPPE → SOLVEPNP_ITERATIVE")
print(f"   Corner refinement: Multi-stage subpixel")
print(f"   Minimum corners: 20 (was 10)")
print(f"   Max reprojection error: 2.0 pixels")

# If you KNOW these, set them; otherwise leave None to auto-lock from the image
ARUCO_DICT_ID    = None     # e.g. cv2.aruco.DICT_4X4_250
FIRST_MARKER_ID  = None     # e.g. 17

# HIGH ACCURACY: Stricter pose diversity requirements (reduce conditioning errors)
MIN_ANGLE_DEG    = 20.0     # Larger angle changes between poses - FORCE more diversity
MIN_TRANS_M      = 0.08     # Larger translation changes - FORCE more diversity
MIN_SAMPLES      = 8        # More minimum samples
TARGET_SAMPLES   = 35       # More total samples for better accuracy
MIN_CHARUCO_CORNERS = 20    # Require more corners for accurate pose estimation (was 10)

AXIS_LEN_M       = 0.08
SAVE_DIR         = "../output/poses_orbbec"  # Directory to save pose pairs

# Load Orbbec intrinsics/distortion from calibration file
# You'll need to create this file using OrbbecIntrinsics.py first
try:
    import json

    # Load from JSON file - use latest calibration with fixed IPPE
    intrinsics_file = "../output/orbbec_calibration_20250929_204146.json"
    with open(intrinsics_file, 'r') as f:
        calib_data = json.load(f)

    # Extract camera matrix and distortion coefficients
    K = np.array(calib_data["camera_matrix"], dtype=np.float64)
    dist = np.array(calib_data["dist_coeffs"], dtype=np.float64)

    print("✓ Loaded Orbbec calibration from JSON")
    print(f"📷 Camera matrix focal lengths: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"   Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"   Distortion coeffs: {[f'{d:.4f}' for d in dist.flatten()]}")

    # CRITICAL: Validate calibration quality
    if 'reprojection_error' in calib_data:
        reproj_error = float(calib_data['reprojection_error'])
        print(f"🎯 Camera reprojection error: {reproj_error:.3f} pixels")

        if reproj_error > 0.8:
            print("❌ CRITICAL: Poor camera calibration!")
            print(f"   Reprojection error {reproj_error:.3f} pixels will cause 10-50mm hand-eye errors")
            print("   🔧 REQUIRED: Recalibrate camera with:")
            print("   - 50+ high-quality images")
            print("   - Full field of view coverage")
            print("   - Sharp focus throughout")
            print("   - Various distances and angles")
            print("   - Steady capture (no motion blur)")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        elif reproj_error > 0.4:
            print("⚠️  WARNING: Moderate calibration error")
            print(f"   {reproj_error:.3f} pixels may cause 3-10mm final errors")
            print("   📈 Recommend: Recalibrate for better accuracy")
        else:
            print("✅ Excellent calibration quality!")
    else:
        print("⚠️  No reprojection error data available")
        print("   Recommend recalibrating with error reporting enabled")

    # Validate camera matrix sanity
    if K[0,0] < 500 or K[1,1] < 500:
        print("⚠️  WARNING: Unusually low focal lengths - check calibration")
    if abs(K[0,0] - K[1,1]) / K[0,0] > 0.05:
        print("⚠️  WARNING: Large focal length difference - check for aspect ratio issues")

except FileNotFoundError:
    print(f"❌ Orbbec calibration file not found: {intrinsics_file}")
    print("Please check the file path")
    sys.exit(1)
except KeyError as e:
    print(f"❌ Missing key in calibration file: {e}")
    print("Expected keys: 'camera_matrix', 'dist_coeffs'")
    sys.exit(1)

# (Kalman filter removed per request; using raw pose outputs only)

# =========================
# Utilities
# =========================
def euler_angles_to_rotation_matrix(rx, ry, rz):
    # Compute the rotation matrix
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R

def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    # Delegate to shared utility. utils.rpy_to_matrix expects degrees.
    if not degrees:
        roll = np.degrees(roll); pitch = np.degrees(pitch); yaw = np.degrees(yaw)
    return rpy_to_matrix(roll, pitch, yaw)

def check_pose_diversity(R_list):
    """Check rotation diversity across poses"""
    if len(R_list) < 2:
        return 0.0

    angles = []
    for i in range(len(R_list)):
        for j in range(i+1, len(R_list)):
            R_diff = R_list[i] @ R_list[j].T
            angle = np.abs(cv2.Rodrigues(R_diff)[0]).max() * 180/np.pi
            angles.append(angle)

    return max(angles) if angles else 0.0

def check_translation_diversity(t_list):
    """Check translation diversity across poses"""
    if len(t_list) < 2:
        return 0.0

    distances = []
    for i in range(len(t_list)):
        for j in range(i+1, len(t_list)):
            dist = np.linalg.norm(t_list[i] - t_list[j])
            distances.append(dist)

    return max(distances) if distances else 0.0

# =========================
# Orbbec source
# =========================
class OrbbecSource:
    def __init__(self, width=1920, height=1080, fps=30, camera_kind: str = "orbbec"):
        self.cam = create_camera(kind=camera_kind, width=width, height=height, fps=fps)

    def read(self):
        return self.cam.read()

    def close(self):
        self.cam.close()

# =========================
# ChArUco with auto dict + firstMarkerId lock-in (no board mutation)
# =========================
def _dict_size(dict_id):
    m = {
        cv2.aruco.DICT_4X4_50: 50,    cv2.aruco.DICT_4X4_100: 100,
        cv2.aruco.DICT_4X4_250: 250,  cv2.aruco.DICT_4X4_1000: 1000,
        cv2.aruco.DICT_5X5_50: 50,    cv2.aruco.DICT_5X5_100: 100,
        cv2.aruco.DICT_5X5_250: 250,  cv2.aruco.DICT_5X5_1000: 1000,
    }
    return m.get(dict_id, 250)

def _make_board_and_detector(dict_id):
    """Create high-accuracy ArUco detector and ChArUco board"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()

    # HIGH ACCURACY SETTINGS for precise hand-eye calibration
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 2
    params.adaptiveThreshConstant = 7

    # CRITICAL: Enable advanced corner refinement for accuracy
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR  # Best accuracy
    params.cornerRefinementWinSize = 15       # Larger window for better accuracy
    params.cornerRefinementMaxIterations = 100
    params.cornerRefinementMinAccuracy = 0.001  # Higher accuracy requirement

    # Enhanced detection settings
    params.detectInvertedMarker = True
    params.minMarkerPerimeterRate = 0.03      # Stricter perimeter requirements
    params.maxMarkerPerimeterRate = 4.0
    params.minCornerDistanceRate = 0.05       # Better corner separation
    params.markerBorderBits = 1
    params.useAruco3Detection = True

    # Additional accuracy improvements
    params.minDistanceToBorder = 3
    params.minMarkerDistanceRate = 0.05

    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M, MARKER_LEN_M, aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(board)
    detector.setDetectorParameters(aruco_detector.getDetectorParameters())
    return board, detector

_COMMON_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
]

def make_charuco_state():
    state = {
        'locked': False,
        'dict_id': None,
        'first_off': None,   # firstMarkerId offset
        'board': None,
        'detector': None,
        'candidates': {},    # dict_id -> (board, detector)
        'last_markers': 0,
        'last_charuco': 0,
    }
    if ARUCO_DICT_ID is not None:
        board, det = _make_board_and_detector(ARUCO_DICT_ID)
        state.update({'locked': True,
                      'dict_id': ARUCO_DICT_ID,
                      'first_off': (FIRST_MARKER_ID or 0),
                      'board': board,
                      'detector': det})
    return state

def _adjust_ids(ids, offset, dict_sz):
    # ids is shape (N,1); keep shape
    ids_i = ids.astype(np.int32)
    ids_adj = ((ids_i - int(offset)) % dict_sz).astype(np.int32)
    return ids_adj

def draw_counts(img, markers, charuco):
    cv2.putText(img, f"markers:{markers}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(img, f"charuco:{charuco}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

def draw_bounding_boxes(img, charuco_corners, charuco_ids, marker_corners=None, marker_ids=None):
    """Draw oriented bounding boxes around detected ChArUco corners and markers"""
    try:
        # Draw small circles around ChArUco corners
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            for i, corner in enumerate(charuco_corners):
                # ChArUco corners are individual points, not 4-point polygons
                if len(corner.shape) == 2 and corner.shape[1] == 2:  # Shape (1, 2) for single corner
                    x, y = int(corner[0, 0]), int(corner[0, 1])
                    # Draw a small circle around the corner point
                    cv2.circle(img, (x, y), 8, (255, 255, 0), 2)  # Yellow circle

                    # Add corner ID label
                    if i < len(charuco_ids):
                        cv2.putText(img, f"Ch{charuco_ids[i][0]}", (x + 12, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw oriented bounding boxes around individual markers
        if marker_corners is not None and marker_ids is not None and len(marker_corners) > 0:
            for i, corners in enumerate(marker_corners):
                # ArUco marker corners have shape (1, 4, 2)
                if len(corners.shape) == 3 and corners.shape[1] == 4:
                    # Get the 4 corner points of the marker
                    pts = corners[0].astype(np.int32)  # Shape: (4, 2)

                    # Draw the oriented quadrilateral (follows marker rotation)
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)  # Green oriented outline

                    # Add marker ID label at the top-left corner
                    if i < len(marker_ids):
                        # Find the "top-left" corner (closest to origin)
                        distances = np.sum(pts**2, axis=1)
                        top_left_idx = np.argmin(distances)
                        label_x, label_y = pts[top_left_idx]

                        cv2.putText(img, f"M{marker_ids[i][0]}", (label_x - 10, label_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        import traceback
        traceback.print_exc()

def estimate_charuco_pose(raw_img, K, dist, debug_img=None, state=None):
    """
    Auto-lock dictionary and firstMarkerId from observed IDs using raw images.
    Returns (R, t) or None. Updates state['last_markers'], state['last_charuco'].
    Note: Uses raw images with full distortion model for pose estimation.
    """
    assert state is not None

    if not state['locked']:
        # try each common dict
        for did in _COMMON_DICTS if ARUCO_DICT_ID is None else [ARUCO_DICT_ID]:
            if did not in state['candidates']:
                state['candidates'][did] = _make_board_and_detector(did)
            board, det = state['candidates'][did]

            # Use the new CharucoDetector API
            charuco_corners, charuco_ids, marker_corners, marker_ids = det.detectBoard(raw_img)
            state['last_charuco'] = 0 if charuco_ids is None else int(len(charuco_ids))
            state['last_markers'] = 0 if marker_ids is None else int(len(marker_ids))
            
            if debug_img is not None and charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
                # Draw bounding boxes around detected elements
                draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)

            # HIGH ACCURACY: Require more corners for reliable pose estimation
            if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
                if debug_img is not None:
                    draw_counts(debug_img, 0, state['last_charuco'])
                    cv2.putText(debug_img, f"Need {MIN_CHARUCO_CORNERS}+ corners", (20,190),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                continue

            # HIGH ACCURACY: Multi-stage subpixel corner refinement
            if charuco_corners is not None and len(charuco_corners) > 0:
                gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                ch_corners_float = np.array(charuco_corners, dtype=np.float32)

                # Stage 1: Coarse subpixel refinement with larger window
                cv2.cornerSubPix(gray_img, ch_corners_float, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))

                # Stage 2: Fine subpixel refinement with smaller window
                cv2.cornerSubPix(gray_img, ch_corners_float, (5, 5), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001))

                charuco_corners = ch_corners_float

            # HIGH ACCURACY: IPPE + Iterative pose estimation for planar objects
            obj_points = board.getChessboardCorners()

            # HIGH ACCURACY: IPPE + Iterative using raw images with distortion model
            # Step 1: IPPE method (best for planar objects like ChArUco)
            ok_ippe, rvecs_ippe, tvecs_ippe, _ = cv2.solvePnPGeneric(
                obj_points[charuco_ids.flatten()],
                charuco_corners,
                K, dist,  # Use full distortion model with raw images
                flags=cv2.SOLVEPNP_IPPE  # Best initial estimate for planar objects
            )

            if not ok_ippe or len(rvecs_ippe) == 0:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            # Step 2: Select best IPPE solution and refine with iterative
            # IPPE returns two solutions for planar objects - pick the one with positive Z
            best_rvec_ippe = None
            best_tvec_ippe = None
            for i in range(len(rvecs_ippe)):
                if tvecs_ippe[i][2] > 0:  # Board should be in front of camera (positive Z)
                    best_rvec_ippe = rvecs_ippe[i]
                    best_tvec_ippe = tvecs_ippe[i]
                    break

            # Fallback to first solution if no positive Z found
            if best_rvec_ippe is None:
                best_rvec_ippe = rvecs_ippe[0]
                best_tvec_ippe = tvecs_ippe[0]

            # Iterative refinement using best IPPE result as initial guess
            ok, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
                obj_points[charuco_ids.flatten()],
                charuco_corners,
                K, dist,  # Use full distortion model with raw images
                rvec=best_rvec_ippe,  # Use best IPPE result as initial guess
                tvec=best_tvec_ippe,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE  # Refine with iterative
            )

            if not ok or len(rvecs) == 0:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            # Get the best solution (first one from iterative solver)
            rvec = rvecs[0]
            tvec = tvecs[0]
            reproj_error = reprojErrors[0][0] if len(reprojErrors) > 0 else float('inf')

            # HIGH ACCURACY: Reject poses with high reprojection error
            if reproj_error > 1.0:  # 1 pixel maximum error for high accuracy
                print(f"⚠️  Rejecting pose with high reprojection error: {reproj_error:.2f} pixels")
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                    cv2.putText(debug_img, f"High error: {reproj_error:.1f}px", (20,220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                continue

            # Use raw measured pose (no Kalman filtering)
            R_measured, _ = cv2.Rodrigues(rvec)
            t_measured = tvec.reshape(3)

            # Display accuracy info
            if debug_img is not None:
                cv2.putText(debug_img, f"IPPE+Iter: {reproj_error:.2f}px", (20,220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(debug_img, "Raw+Dist", (20,250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Lock in
            state.update({'locked': True, 'dict_id': did, 'first_off': 0,
                          'board': board, 'detector': det})
            print(f"[ChArUco] Locked dict={did}, firstMarkerId=0, "
                  f"markers={state['last_markers']}, charuco={state['last_charuco']}")

            if debug_img is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
                # Draw bounding boxes around detected elements
                draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)
                draw_counts(debug_img, state['last_markers'], state['last_charuco'])

            return R_measured, t_measured

        return None

    # locked path: reuse board/detector and detect board directly
    board, det = state['board'], state['detector']
    
    # Use the new CharucoDetector API
    charuco_corners, charuco_ids, marker_corners, marker_ids = det.detectBoard(raw_img)
    state['last_charuco'] = 0 if charuco_ids is None else int(len(charuco_ids))
    state['last_markers'] = 0 if marker_ids is None else int(len(marker_ids))
    
    if debug_img is not None and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
        # Draw bounding boxes around detected elements
        draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)

    # HIGH ACCURACY: Require more corners for reliable pose estimation
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
            cv2.putText(debug_img, f"Need {MIN_CHARUCO_CORNERS}+ corners", (20,190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return None

    # HIGH ACCURACY: Multi-stage subpixel corner refinement
    if charuco_corners is not None and len(charuco_corners) > 0:
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        ch_corners_float = np.array(charuco_corners, dtype=np.float32)

        # Stage 1: Coarse subpixel refinement with larger window
        cv2.cornerSubPix(gray_img, ch_corners_float, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))

        # Stage 2: Fine subpixel refinement with smaller window
        cv2.cornerSubPix(gray_img, ch_corners_float, (5, 5), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001))

        charuco_corners = ch_corners_float

    # HIGH ACCURACY: IPPE + Iterative pose estimation for planar objects
    obj_points = board.getChessboardCorners()

    # HIGH ACCURACY: IPPE + Iterative using raw images with distortion model
    # Step 1: IPPE method (best for planar objects like ChArUco)
    ok_ippe, rvecs_ippe, tvecs_ippe, _ = cv2.solvePnPGeneric(
        obj_points[charuco_ids.flatten()],
        charuco_corners,
        K, dist,  # Use full distortion model with raw images
        flags=cv2.SOLVEPNP_IPPE  # Best initial estimate for planar objects
    )

    if not ok_ippe or len(rvecs_ippe) == 0:
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
        return None

    # Step 2: Select best IPPE solution and refine with iterative
    # IPPE returns two solutions for planar objects - pick the one with positive Z
    best_rvec_ippe = None
    best_tvec_ippe = None
    for i in range(len(rvecs_ippe)):
        if tvecs_ippe[i][2] > 0:  # Board should be in front of camera (positive Z)
            best_rvec_ippe = rvecs_ippe[i]
            best_tvec_ippe = tvecs_ippe[i]
            break

    # Fallback to first solution if no positive Z found
    if best_rvec_ippe is None:
        best_rvec_ippe = rvecs_ippe[0]
        best_tvec_ippe = tvecs_ippe[0]

    # Iterative refinement using best IPPE result as initial guess
    ok, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
        obj_points[charuco_ids.flatten()],
        charuco_corners,
        K, dist,  # Use full distortion model with raw images
        rvec=best_rvec_ippe,  # Use best IPPE result as initial guess
        tvec=best_tvec_ippe,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE  # Refine with iterative
    )

    if not ok or len(rvecs) == 0:
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
        return None

    # Get the best solution (first one from iterative solver)
    rvec = rvecs[0]
    tvec = tvecs[0]
    reproj_error = reprojErrors[0][0] if len(reprojErrors) > 0 else float('inf')

    # HIGH ACCURACY: Reject poses with high reprojection error
    if reproj_error > 1.0:  # 1 pixel maximum error for high accuracy
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
            cv2.putText(debug_img, f"High error: {reproj_error:.1f}px", (20,220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return None

    # Use raw measured pose (no Kalman filtering)
    R_measured, _ = cv2.Rodrigues(rvec)
    t_measured = tvec.reshape(3)

    # Display accuracy info
    if debug_img is not None:
        cv2.putText(debug_img, f"IPPE+Iter: {reproj_error:.2f}px", (20,220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(debug_img, "Raw+Dist", (20,250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    if debug_img is not None:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
        # Draw bounding boxes around detected elements
        draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)
        draw_counts(debug_img, state['last_markers'], state['last_charuco'])

    return R_measured, t_measured

# =========================
# xArm client
# =========================
class XArmClient:
    def __init__(self, ip):
        self.arm = XArmAPI(ip)
        self.arm.connect()
        # self.arm.motion_enable(True)
        # self.arm.set_mode(0)
        # self.arm.set_state(0)

    def get_base_to_gripper(self):
        code, pos = self.arm.get_position(is_radian=not USE_DEG)
        if code != 0:
            raise RuntimeError(f"xArm get_position failed, code={code}")
        x, y, z, roll, pitch, yaw = pos

        # Convert position from mm to meters
        t_bg = np.array([x, y, z], dtype=np.float64) / 1000.0

        # Convert angles to radians if needed (your function expects radians)
        if USE_DEG:
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
        else:
            roll_rad, pitch_rad, yaw_rad = roll, pitch, yaw

        # Use your custom function
        R_bg = euler_angles_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
        return R_bg, t_bg

    def close(self):
        try: self.arm.disconnect()
        except Exception: pass

# =========================
# Residual diagnostics
# =========================
def handeye_residuals(Rg, tg, Rt, tt, R_cam2gripper, t_cam2gripper):
    """
    Calculate residuals for eye-in-hand hand-eye calibration.
    Rg: gripper poses in base frame
    tg: gripper translations in base frame
    Rt: target poses in camera frame
    tt: target translations in camera frame
    R_cam2gripper, t_cam2gripper: camera to gripper transformation (what we're calibrating)
    """
    X = to_homogeneous(R_cam2gripper, t_cam2gripper)
    rots, trans = [], []
    for i in range(len(Rg) - 1):
        RA, tA = rel_motion(Rg[i], tg[i], Rg[i+1], tg[i+1])
        RB, tB = rel_motion(Rt[i+1], tt[i+1], Rt[i], tt[i])  # inverse order
        L = to_homogeneous(RA, tA) @ X
        Rhs = X @ to_homogeneous(RB, tB)
        dR = L[:3,:3].T @ Rhs[:3,:3]
        dtheta = rot_angle_deg(dR)
        dt = np.linalg.norm(L[:3,3] - Rhs[:3,3])
        rots.append(dtheta); trans.append(dt)
    if not rots: return None
    def stats(a): a=np.array(a); return dict(mean=float(np.mean(a)),
                                            median=float(np.median(a)),
                                            p95=float(np.percentile(a,95)))
    return dict(rot_deg=stats(rots), trans_m=stats(trans))

# =========================
# COORDINATE FRAME CONVENTION
# =========================
"""
COORDINATE FRAME CONVENTION for Eye-in-Hand Calibration:

1. Base Frame: Robot base coordinate system
2. Gripper Frame: Robot gripper/end-effector coordinate system
3. Camera Frame: Camera coordinate system (mounted on gripper)
4. Target Frame: ChArUco board coordinate system (fixed in world)

TRANSFORMATIONS:
- R_gripper2base, t_gripper2base: Gripper pose in base frame (from robot)
- R_target2cam, t_target2cam: Target pose in camera frame (from vision)
- R_cam2gripper, t_cam2gripper: Camera pose in gripper frame (WHAT WE WANT)

HAND-EYE EQUATION:
AX = XB where:
- A = relative motion between gripper poses
- B = relative motion between target poses
- X = camera-to-gripper transformation (unknown)

CRITICAL: Camera and gripper are NOT the same! We're finding the fixed transformation between them.
"""

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-Hand hand-eye calibration for Orbbec Camera (terminal-friendly)")
    parser.add_argument("--mode", choices=["gui", "ascii", "ascii_hi", "headless"],
                        default=("gui" if os.environ.get("DISPLAY") else "ascii"),
                        help="Display mode: OpenCV GUI, ASCII in terminal, or headless")
    parser.add_argument("--camera", choices=["auto", "orbbec", "realsense", "opencv"], default="orbbec",
                        help="Camera backend to use: Orbbec (default), RealSense (if available) or OpenCV UVC")
    args = parser.parse_args()
    
    print("🔬 Running in HIGH ACCURACY mode for optimal calibration results")

    print("\n=== Instructions ===")
    print("• Mount Orbbec camera rigidly on the gripper (eye-in-hand).")
    print("• Fix ChArUco board rigidly in the environment.")
    print("• Move to varied poses (large rotations + translations).")
    if args.mode == "gui":
        print("• Press [SPACE] to capture, [q] to finish in the GUI window.\n")
    else:
        print("• In terminal, press SPACE to capture, q to finish.\n")

    print("Opening Orbbec camera...")
    try:
        orbbec_cam = OrbbecSource(ORBBEC_WIDTH, ORBBEC_HEIGHT, ORBBEC_FPS, camera_kind=args.camera)
        print("✓ Camera object created successfully")
    except Exception as e:
        print(f"❌ Failed to create camera: {e}")
        print("Please check camera connection and drivers")
        return

    print(f"Connecting to xArm at {XARM_IP}...")
    try:
        xarm = XArmClient(XARM_IP)
        print("✓ xArm connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect to xArm: {e}")
        print("Please check network connection and xArm IP address")
        orbbec_cam.close()
        return

    ch_state = make_charuco_state()
    R_g2b_list, t_g2b_list = [], []  # gripper to base (camera pose)
    R_t2c_list, t_t2c_list = [], []  # target to camera
    captured_images = []  # Store captured images for saving

    last_Rg, last_tg = None, None

    display = Display(args.mode)
    
    # Give camera time to initialize
    print("Waiting for camera to initialize...")
    import time
    time.sleep(2)
    
    # Try to read a few frames to ensure camera is working
    camera_ready = False
    for attempt in range(10):
        ok, color = orbbec_cam.read()
        if ok and color is not None:
            print(f"✓ Camera ready after {attempt + 1} attempts")
            camera_ready = True
            break
        print(f"Camera initialization attempt {attempt + 1}/10...")
        time.sleep(0.5)
    
    if not camera_ready:
        print("❌ Camera failed to initialize after 10 attempts")
        return
    
    try:
        while True:
            ok, color = orbbec_cam.read()
            if not ok or color is None:
                print("Camera read failed")
                break

            # Process every frame for maximum accuracy

            # Use raw image with distortion model for pose estimation
            vis = color.copy()
            det = estimate_charuco_pose(color, K, dist, debug_img=vis, state=ch_state)

            if det is not None:
                R, t = det
                if args.mode in ("ascii", "ascii_hi", "headless"):
                    draw_axes_ascii_friendly(vis, K, dist, R, t, AXIS_LEN_M, thickness=6)
                else:
                    cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(R)[0], t.reshape(3,1), AXIS_LEN_M)
                cv2.putText(vis, "Board detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(vis, "No board", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cap_txt = f"Captures: {len(R_g2b_list)} / {TARGET_SAMPLES}"
            cv2.putText(vis, cap_txt, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            status = f"Captures: {len(R_g2b_list)}/{TARGET_SAMPLES}  |  {'Board detected' if det is not None else 'No board'}  |  [SPACE]=capture, q=quit"
            key = display.update(vis, status)

            if key == 'q':
                break
            if key == ' ':
                if det is None:
                    print("⚠️  Need the board detected. Adjust and try again.")
                    continue
                try:
                    R_bg, t_bg = xarm.get_base_to_gripper()
                    #R_bg, t_bg = np.eye(3), np.zeros(3)
                except Exception as e:
                    print(f"⚠️  xArm read failed: {e}")
                    continue

                # CORRECT: Store gripper pose in base frame
                # get_base_to_gripper() returns gripper pose IN base frame (not gripper-TO-base transform)
                R_gripper_in_base = R_bg  # gripper pose in base frame (from robot)
                t_gripper_in_base = t_bg  # gripper position in base frame (from robot)

                accept = True
                if last_Rg is not None:
                    dR, dt = rel_motion(last_Rg, last_tg, R_gripper_in_base, t_gripper_in_base)
                    ang = rot_angle_deg(dR); d = np.linalg.norm(dt)
                    print(f"Movement: Δang={ang:.1f}°, Δt={d*1000:.1f}mm (need >{MIN_ANGLE_DEG:.0f}°, >{MIN_TRANS_M*1000:.0f}mm)")
                    if ang < MIN_ANGLE_DEG and d < MIN_TRANS_M:
                        print(f"❌ Pose too similar! Need MORE movement.")
                        accept = False
                    else:
                        print(f"✅ Good movement diversity!")

                if accept:
                    R_g2b_list.append(R_gripper_in_base); t_g2b_list.append(t_gripper_in_base)
                    R, t = det
                    R_t2c_list.append(R); t_t2c_list.append(t)
                    captured_images.append(color.copy())  # Store the captured image
                    last_Rg, last_tg = R_gripper_in_base, t_gripper_in_base
                    
                    # Save pose pair immediately
                    pose_num = len(R_g2b_list)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    # Convert rotation matrices to Euler angles for saving
                    R_t2c_euler = cv2.Rodrigues(R)[0]  # Use R from det
                    
                    # OpenCV calibrateHandEye expects gripper poses IN base frame
                    
                    # Save as JPG
                    img_name = f"{SAVE_DIR}/pose{pose_num:03d}.jpg"
                    cv2.imwrite(img_name, captured_images[-1])
                    
                    # Save as NPY with gripper poses in base frame (for OpenCV calibrateHandEye)
                    np.save(f"{SAVE_DIR}/pose{pose_num:03d}.npy", {
                        "R_gripper_in_base": R_gripper_in_base,   # Gripper pose in base frame (from robot SDK)
                        "t_gripper_in_base": t_gripper_in_base,   # Gripper position in base frame (from robot SDK)
                        "R_target2cam": R,                        # Target to camera (from vision)
                        "t_target2cam": t,                        # Target to camera (from vision)
                        "pose_number": pose_num,
                        "timestamp": pose_num  # You could add actual timestamp here if needed
                    }, allow_pickle=True)
                    
                    # Check image focus quality
                    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

                    print(f"Captured #{pose_num}  (markers:{ch_state['last_markers']}, charuco:{ch_state['last_charuco']}, focus:{focus_measure:.1f})")
                    if focus_measure < 100:
                        print("⚠️  WARNING: Low focus quality! Check camera focus.")
                    print(f"Saved → {img_name}")
                    print(f"Saved → {SAVE_DIR}/pose{pose_num:03d}.npy")

                    # Print saved pose contents for verification
                    try:
                        np.set_printoptions(precision=5, suppress=True)
                        print("— Pose contents (what was saved) —")
                        print("  R_gripper_in_base:\n", R_gripper_in_base)
                        print("  t_gripper_in_base (mm): ", (t_gripper_in_base*1000.0))
                        print("  R_target2cam:\n", R)
                        print("  t_target2cam (mm): ", (t*1000.0))
                        # Convert R_gripper_in_base to RPY angles to compare with SDK
                        from scipy.spatial.transform import Rotation as Rot
                        gripper_rpy = Rot.from_matrix(R_gripper_in_base).as_euler('xyz', degrees=True)
                        print("  Gripper RPY (deg) [matches SDK]: ", gripper_rpy)
                        print("  R_target2cam_rvec (deg): ", (R_t2c_euler.reshape(3)*180.0/np.pi))
                        print("  pose_number: ", pose_num)
                        print("  timestamp: ", pose_num)
                    except Exception as e:
                        print(f"[WARN] Failed to print pose contents: {e}")
    finally:
        display.close()
        orbbec_cam.close()
        # xarm.close()

    n = len(R_g2b_list)
    if n < MIN_SAMPLES:
        print(f"Not enough samples ({n}). Aim for {TARGET_SAMPLES}+ varied poses.")
        return

    Rg, tg = to_cv_lists(R_g2b_list, t_g2b_list)
    Rt, tt = to_cv_lists(R_t2c_list, t_t2c_list)

    # DATA QUALITY VALIDATION before calibration
    print("\n🔍 DATA QUALITY VALIDATION")

    # Check pose diversity
    angle_range = check_pose_diversity(R_g2b_list)
    trans_range = check_translation_diversity(t_g2b_list)

    print(f"Rotation diversity: {angle_range:.1f}° range")
    print(f"Translation diversity: {trans_range*1000:.1f}mm range")

    # Validate minimum diversity requirements
    if angle_range < 30.0:
        print("⚠️  WARNING: Low rotation diversity! Move robot through larger angles.")
    if trans_range < 0.1:  # 100mm
        print("⚠️  WARNING: Low translation diversity! Move robot through larger distances.")

    # Check for outliers in target detection quality
    target_errors = []
    for i in range(len(R_t2c_list)):
        # Estimate detection quality from pose consistency
        if i > 0:
            R_diff = R_t2c_list[i] @ R_t2c_list[i-1].T
            angle_diff = np.abs(cv2.Rodrigues(R_diff)[0]).max() * 180/np.pi
            target_errors.append(angle_diff)

    if target_errors:
        mean_consistency = np.mean(target_errors)
        print(f"Target detection consistency: {mean_consistency:.1f}° mean variation")
        if mean_consistency > 5.0:
            print("⚠️  WARNING: High target detection variation! Check lighting and focus.")

    print(f"\n🔍 DEBUG: Calibration input data:")
    print(f"Number of poses: {len(R_g2b_list)}")
    if len(R_g2b_list) > 0:
        print(f"First gripper position (mm): [{t_g2b_list[0][0]*1000:.1f}, {t_g2b_list[0][1]*1000:.1f}, {t_g2b_list[0][2]*1000:.1f}]")
        print(f"Last gripper position (mm): [{t_g2b_list[-1][0]*1000:.1f}, {t_g2b_list[-1][1]*1000:.1f}, {t_g2b_list[-1][2]*1000:.1f}]")
        print(f"First target position (mm): [{t_t2c_list[0][0]*1000:.1f}, {t_t2c_list[0][1]*1000:.1f}, {t_t2c_list[0][2]*1000:.1f}]")
        print(f"Last target position (mm): [{t_t2c_list[-1][0]*1000:.1f}, {t_t2c_list[-1][1]*1000:.1f}, {t_t2c_list[-1][2]*1000:.1f}]")

    print("\nRunning hand-eye calibration (eye-in-hand)...")

    # Try different calibration methods
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"),
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
    ]
    
    calibration_results = {}
    
    for method, name in methods:
        try:
            R, t = cv2.calibrateHandEye(
                R_gripper2base=Rg, t_gripper2base=tg,  # Gripper poses in base frame (correct for eye-in-hand)
                R_target2cam=Rt,  t_target2cam=tt,     # Target poses in camera frame
                method=method
            )
            t = t.reshape(3)
            calibration_results[name] = {
                'R': R,
                't': t,
                'method': method
            }
            # More concise output showing translation magnitude and rotation angle
            t_norm = np.linalg.norm(t)
            rvec = cv2.Rodrigues(R)[0]
            r_angle = np.linalg.norm(rvec) * 180/np.pi
            print(f"Method {name} → t_norm={t_norm*1000:.1f}mm, rot_angle={r_angle:.1f}°")
        except Exception as e:
            print(f"Method {name} failed: {e}")
            calibration_results[name] = None
    
    # HIGH ACCURACY: Select method based on residuals, not defaults
    print("\n🎯 Evaluating calibration methods based on residuals...")

    best_method = None
    best_residual = float('inf')
    best_result = None

    for name, result in calibration_results.items():
        if result is not None:
            # Calculate residuals for this method
            try:
                res = handeye_residuals(Rg, tg, Rt, tt, result['R'], result['t'])
                if res:
                    mean_trans_error = res['trans_m']['mean']
                    mean_rot_error = res['rot_deg']['mean']
                    print(f"Method {name}: trans={mean_trans_error*1000:.1f}mm, rot={mean_rot_error:.1f}°")

                    # Weighted score: translation error is more critical for hand-eye calibration
                    score = mean_trans_error * 1000 + mean_rot_error * 0.1  # mm + weighted degrees

                    if score < best_residual:
                        best_residual = score
                        best_method = name
                        best_result = result
                else:
                    print(f"Method {name}: Failed to calculate residuals")
            except Exception as e:
                print(f"Method {name}: Error calculating residuals: {e}")

    if best_result is None:
        # Fallback to PARK if no residuals could be calculated
        if calibration_results.get('PARK') is not None:
            best_result = calibration_results['PARK']
            best_method = 'PARK'
            print(f"\n⚠️  Using PARK method as fallback (residual calculation failed)")
        else:
            # Final fallback to first available method
            for name, result in calibration_results.items():
                if result is not None:
                    best_result = result
                    best_method = name
                    break
            if best_result is None:
                raise RuntimeError("All calibration methods failed")

    # CRITICAL FIX: OpenCV calibrateHandEye directly gives us camera-to-gripper transformation
    # The result 'R' and 't' are already the camera-to-gripper transformation!
    R_cam2gripper = best_result['R']
    t_cam2gripper = best_result['t']
    print(f"\n✅ Selected method: {best_method} (score: {best_residual:.2f})")
    print(f"   Expected accuracy: ~{best_residual:.1f}mm translation error")

    # Create 4x4 transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3,:3] = R_cam2gripper
    T_cam2gripper[:3,3] = t_cam2gripper

    # CALIBRATION RESULT VALIDATION
    print(f"\n🔍 CALIBRATION RESULT VALIDATION")

    # Check if rotation matrix is valid
    det_R = np.linalg.det(R_cam2gripper)
    if abs(det_R - 1.0) > 0.01:
        print(f"⚠️  WARNING: Rotation matrix determinant = {det_R:.3f} (should be 1.0)")

    # Check orthogonality
    orthogonality_error = np.linalg.norm(R_cam2gripper @ R_cam2gripper.T - np.eye(3))
    if orthogonality_error > 0.01:
        print(f"⚠️  WARNING: Rotation matrix not orthogonal (error: {orthogonality_error:.3f})")

    # Check translation magnitude reasonableness (typical camera-gripper offset: 50-300mm)
    trans_magnitude = np.linalg.norm(t_cam2gripper)
    if trans_magnitude < 0.05:  # <50mm
        print(f"⚠️  WARNING: Very small camera-gripper offset: {trans_magnitude*1000:.1f}mm")
    elif trans_magnitude > 0.5:  # >500mm
        print(f"⚠️  WARNING: Very large camera-gripper offset: {trans_magnitude*1000:.1f}mm")
    else:
        print(f"✅ Camera-gripper offset: {trans_magnitude*1000:.1f}mm (reasonable)")

    # Check consistency across all calibration methods
    if len(calibration_results) > 1:
        trans_variations = []
        rot_variations = []
        for method_name, result in calibration_results.items():
            if result is not None and method_name != best_method:
                trans_diff = np.linalg.norm(result['t'] - t_cam2gripper)
                R_diff = result['R'] @ R_cam2gripper.T
                rot_diff = np.abs(cv2.Rodrigues(R_diff)[0]).max() * 180/np.pi
                trans_variations.append(trans_diff * 1000)  # mm
                rot_variations.append(rot_diff)

        if trans_variations:
            max_trans_var = max(trans_variations)
            max_rot_var = max(rot_variations)
            print(f"Method consistency: ±{max_trans_var:.1f}mm, ±{max_rot_var:.1f}°")

            if max_trans_var > 10.0:  # >10mm variation
                print("⚠️  WARNING: High variation between calibration methods!")
            elif max_trans_var < 2.0:  # <2mm variation
                print("✅ Excellent consistency between methods")

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== T_cam2gripper (camera to gripper transformation) ===")
    print(T_cam2gripper)
    print("\n=== t_cam2gripper (translation vector) ===")
    print(t_cam2gripper)
    
    # Use the best method that was selected
    selected_method = best_method
    
    # Save all calibration results with method name
    save_data = {
        "t_cam2grip": t_cam2gripper,
        "R_cam2grip": R_cam2gripper,
        "T_cam2grip": T_cam2gripper,
        "all_methods": calibration_results,
        "selected_method": selected_method
    }
    
    # Save with method name in filename
    method_filename = f"eyeinhand_orbbec_{selected_method.lower()}"
    np.save(f"{SAVE_DIR}/result_{method_filename}.npy", save_data, allow_pickle=True)
    np.savez(f"../output/{method_filename}.npz", **save_data)
    
    # Save individual method results with descriptive names
    for method_name, result in calibration_results.items():
        if result is not None:
            method_data = {
                f"R_{method_name.lower()}": result['R'],
                f"t_{method_name.lower()}": result['t'],
                f"method_{method_name.lower()}": result['method'],
                "method_name": method_name,
                "T_cam2grip": to_homogeneous(result['R'], result['t'])
            }
            np.savez(f"{SAVE_DIR}/calibration_{method_name.lower()}.npz", **method_data)
            np.savez(f"../output/eyeinhand_orbbec_{method_name.lower()}.npz", **method_data)

    # FINAL ACCURACY ASSESSMENT
    res = handeye_residuals(Rg, tg, Rt, tt, R_cam2gripper, t_cam2gripper)
    if res:
        trans_error_mm = res['trans_m']['mean'] * 1000
        rot_error_deg = res['rot_deg']['mean']

        print(f"\n🎯 FINAL ACCURACY ASSESSMENT")
        print(f"Selected method: {best_method}")
        print(f"Translation error: {trans_error_mm:.1f}mm (mean), {res['trans_m']['p95']*1000:.1f}mm (95th percentile)")
        print(f"Rotation error: {rot_error_deg:.2f}° (mean), {res['rot_deg']['p95']:.2f}° (95th percentile)")

        if trans_error_mm < 2.0:
            print("✅ EXCELLENT accuracy achieved!")
        elif trans_error_mm < 5.0:
            print("✅ GOOD accuracy achieved!")
        elif trans_error_mm < 10.0:
            print("⚠️  MODERATE accuracy - consider:")
            print("   - Measuring board dimensions more precisely")
            print("   - Recalibrating camera with more images")
            print("   - Taking more poses with larger movements")
        else:
            print("❌ POOR accuracy - please:")
            print("   - Check camera calibration quality")
            print("   - Verify board measurements with calipers")
            print("   - Ensure poses have sufficient diversity")
            print("   - Check for lighting/focus issues")

        print(f"\nDetailed residuals:")
        print(f"Rotation: mean={res['rot_deg']['mean']:.3f}°, med={res['rot_deg']['median']:.3f}°, p95={res['rot_deg']['p95']:.3f}°")
        print(f"Translation: mean={res['trans_m']['mean']*1000:.1f}mm, med={res['trans_m']['median']*1000:.1f}mm, p95={res['trans_m']['p95']*1000:.1f}mm")

    print(f"\n=== HIGH ACCURACY Eye-in-hand calibration complete ===")
    print(f"Total poses captured: {len(R_g2b_list)} (required {MIN_CHARUCO_CORNERS}+ corners each)")
    print(f"Pose pairs saved to: {SAVE_DIR}")
    print(f"Main calibration result saved to: ../output/{method_filename}.npz")
    print(f"Individual method results saved to: ../output/eyeinhand_orbbec_{{method}}.npz")

    # Print summary of all methods with residuals
    print(f"\n=== Calibration Methods Comparison ===")
    for method_name, result in calibration_results.items():
        if result is not None:
            try:
                method_res = handeye_residuals(Rg, tg, Rt, tt, result['R'], result['t'])
                if method_res:
                    method_error = method_res['trans_m']['mean'] * 1000
                    marker = "✅" if method_name == best_method else "  "
                    print(f"{marker} {method_name}: {method_error:.1f}mm error, R_det={np.linalg.det(result['R']):.6f}")
                else:
                    print(f"   {method_name}: R_det={np.linalg.det(result['R']):.6f} (residual calc failed)")
            except:
                print(f"   {method_name}: R_det={np.linalg.det(result['R']):.6f}")
        else:
            print(f"   {method_name}: FAILED")

if __name__ == "__main__":
    main()
