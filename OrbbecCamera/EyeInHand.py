#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import os
import argparse
import sys
import time
import shutil
import select
import termios
import tty
from scipy.spatial.transform import Rotation as R

# Import from Realsense directory
sys.path.insert(0, '../Realsense')
from utils import rpy_to_matrix, rot_angle_deg, to_homogeneous, invert_rt, to_cv_lists, rel_motion
from camera import create_camera
from terminal_display import Display, draw_axes_ascii_friendly
sys.path.pop(0)  # Remove from path after import to avoid conflicts

# =========================
# CONFIG
# =========================
XARM_IP = "192.168.10.202"
USE_DEG = True

# Performance settings
FAST_MODE = False  # Set to False for high accuracy (slower)
ULTRA_FAST_MODE = False  # Even more aggressive optimizations
SKIP_UNDISTORTION = False  # Skip image undistortion for speed

# Orbbec camera settings (1920x1080 is typical for Orbbec)
ORBBEC_WIDTH = 1920
ORBBEC_HEIGHT = 1080
ORBBEC_FPS = 30

# calib.io ChArUco board (you said: rows=4, columns=6)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M      = 0.0345  # measure your printed square side (meters)
MARKER_LEN_RATIO  = 0.81     # calib.io default unless you changed it
MARKER_LEN_M      = MARKER_LEN_RATIO * SQUARE_LEN_M  # measure your printed marker side (meters)
MARKER_LEN_M      = 0.0276
# If you KNOW these, set them; otherwise leave None to auto-lock from the image
ARUCO_DICT_ID    = None     # e.g. cv2.aruco.DICT_4X4_250
FIRST_MARKER_ID  = None     # e.g. 17

# Capture gating (encourage diverse robot poses)
MIN_ANGLE_DEG    = 8.0
MIN_TRANS_M      = 0.03
MIN_SAMPLES      = 3
TARGET_SAMPLES   = 20

AXIS_LEN_M       = 0.08
SAVE_DIR         = "../output/poses_orbbec"  # Directory to save pose pairs

# Load Orbbec intrinsics/distortion from calibration file
# You'll need to create this file using OrbbecIntrinsics.py first
try:
    calib = np.load("../output/orbbec_calibration.npz")
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    print("✓ Loaded Orbbec calibration:")
    print(K)
    print(dist)
except FileNotFoundError:
    print("❌ Orbbec calibration file not found!")
    print("Please run OrbbecIntrinsics.py first to generate ../output/orbbec_calibration.npz")
    print("Or create a calibration file with 'camera_matrix' and 'dist_coeffs' keys")
    sys.exit(1)

# =========================
# Kalman Filter for 6-DoF Pose
# =========================
class PoseKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        Kalman filter for 6-DoF pose tracking
        State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (13D)
        """
        self.dt = 1.0/30.0  # Assume 30 FPS
        self.state_dim = 13
        self.measurement_dim = 7  # [x, y, z, qx, qy, qz, qw]
        
        # State vector: [position, quaternion, velocity, angular_velocity]
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0  # Covariance matrix
        
        # Process noise
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Measurement matrix
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:7, :7] = np.eye(7)  # Direct measurement of pose
        
        self.initialized = False
    
    def update(self, position, rotation_matrix):
        """Update filter with new pose measurement"""
        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Measurement vector
        z = np.array([position[0], position[1], position[2], 
                     quat[0], quat[1], quat[2], quat[3]])
        
        if not self.initialized:
            # Initialize state
            self.x[:3] = position
            self.x[3:7] = quat
            self.initialized = True
            return position, rotation_matrix
        
        # Prediction step
        F = self._get_transition_matrix()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        # Update step
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        # Extract filtered pose
        filtered_position = self.x[:3]
        filtered_quat = self.x[3:7]
        
        # Convert quaternion back to rotation matrix
        r_filtered = R.from_quat(filtered_quat)
        filtered_rotation = r_filtered.as_matrix()
        
        return filtered_position, filtered_rotation
    
    def _get_transition_matrix(self):
        """Get state transition matrix for constant velocity model"""
        F = np.eye(self.state_dim)
        F[:3, 7:10] = np.eye(3) * self.dt  # position += velocity * dt
        return F

# =========================
# Utilities
# =========================
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    # Delegate to shared utility. utils.rpy_to_matrix expects degrees.
    if not degrees:
        roll = np.degrees(roll); pitch = np.degrees(pitch); yaw = np.degrees(yaw)
    return rpy_to_matrix(roll, pitch, yaw)

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
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    
    if ULTRA_FAST_MODE:
        # Ultra-fast settings - minimal processing
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 2
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # No refinement
        params.detectInvertedMarker = False  # Skip inverted detection
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.adaptiveThreshConstant = 7
        params.minCornerDistanceRate = 0.005
        params.markerBorderBits = 1
        params.useAruco3Detection = False  # Disable Aruco3 for speed
    else:
        # Robust-ish defaults; good for prints/screens
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 75
        params.adaptiveThreshWinSizeStep = 1
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 20       # Larger refinement window
        params.cornerRefinementMaxIterations = 100 # More iterations
        params.cornerRefinementMinAccuracy = 0.01 
        params.detectInvertedMarker = True
        params.minMarkerPerimeterRate = 0.005      # Allow smaller markers
        params.maxMarkerPerimeterRate = 6.0 
        params.adaptiveThreshConstant = 3     
        params.minCornerDistanceRate = 0.003  
        params.markerBorderBits = 1  
        params.useAruco3Detection = True

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
        'kalman_filter': PoseKalmanFilter(process_noise=0.01, measurement_noise=0.1),
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

def estimate_charuco_pose(undistorted_img, K, dist, debug_img=None, state=None):
    """
    Auto-lock dictionary and firstMarkerId from observed IDs.
    Returns (R, t) or None. Updates state['last_markers'], state['last_charuco'].
    Note: undistorted_img should be the undistorted image for accurate detection.
    """
    assert state is not None

def draw_counts(img, markers, charuco):
    cv2.putText(img, f"markers:{markers}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(img, f"charuco:{charuco}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

def draw_bounding_boxes(img, charuco_corners, charuco_ids, marker_corners=None, marker_ids=None):
    """Draw bounding boxes around detected ChArUco corners and markers"""
    try:
        # # Draw bounding boxes around ChArUco corners
        # if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
        #     for i, corner in enumerate(charuco_corners):
        #         # ChArUco corners are individual points, not 4-point polygons
        #         if len(corner.shape) == 2 and corner.shape[1] == 2:  # Shape (1, 2) for single corner
        #             x, y = int(corner[0, 0]), int(corner[0, 1])
        #             # Draw a medium-sized box around the corner point
        #             box_size = 15  # Medium-sized box for visibility
        #             x_min, x_max = x - box_size, x + box_size
        #             y_min, y_max = y - box_size, y + box_size
                    
        #             # Draw bounding box in cyan with thin lines
        #             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)  # Thin line
                    
        #             # Add corner ID label with smaller font
        #             if i < len(charuco_ids):
        #                 cv2.putText(img, f"Ch{charuco_ids[i]}", (x_min, y_min-5), 
        #                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Medium font
        
        # Draw bounding boxes around individual markers
        if marker_corners is not None and marker_ids is not None and len(marker_corners) > 0:
            for i, corners in enumerate(marker_corners):
                # ArUco marker corners have shape (1, 4, 2)
                if len(corners.shape) == 3 and corners.shape[1] == 4:
                    # Get bounding rectangle for this marker
                    x_coords = corners[0][:, 0]
                    y_coords = corners[0][:, 1]
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    # Expand the bounding box for better visibility
                    margin = 10
                    x_min, x_max = x_min - margin, x_max + margin
                    y_min, y_max = y_min - margin, y_max + margin
                    
                    # Draw bounding box in green with thin lines
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # Thin line
                    
                    # Add marker ID label with smaller font
                    if i < len(marker_ids):
                        cv2.putText(img, f"M{marker_ids[i][0]}", (x_min, y_min-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Medium font
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        import traceback
        traceback.print_exc()

def estimate_charuco_pose(undistorted_img, K, dist, debug_img=None, state=None):
    """
    Auto-lock dictionary and firstMarkerId from observed IDs.
    Returns (R, t) or None. Updates state['last_markers'], state['last_charuco'].
    Note: undistorted_img should be the undistorted image for accurate detection.
    """
    assert state is not None

    if not state['locked']:
        # try each common dict
        for did in _COMMON_DICTS if ARUCO_DICT_ID is None else [ARUCO_DICT_ID]:
            if did not in state['candidates']:
                state['candidates'][did] = _make_board_and_detector(did)
            board, det = state['candidates'][did]

            # Use the new CharucoDetector API
            charuco_corners, charuco_ids, marker_corners, marker_ids = det.detectBoard(undistorted_img)
            state['last_charuco'] = 0 if charuco_ids is None else int(len(charuco_ids))
            state['last_markers'] = 0 if marker_ids is None else int(len(marker_ids))
            
            if debug_img is not None and charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
                # Draw bounding boxes around detected elements
                draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)

            if charuco_ids is None or len(charuco_ids) < 10:
                if debug_img is not None:
                    draw_counts(debug_img, 0, state['last_charuco'])
                continue

            # Apply subpixel refinement based on mode
            if charuco_corners is not None and len(charuco_corners) > 0 and not FAST_MODE:
                # Convert to float32 for subpixel refinement
                ch_corners_float = np.array(charuco_corners, dtype=np.float32)
                # Convert BGR to grayscale for subpixel refinement
                gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
                # Apply subpixel corner refinement
                cv2.cornerSubPix(gray_img, ch_corners_float, (5, 5), (-1, -1), 
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
                charuco_corners = ch_corners_float

            # Use solvePnP directly with board object points
            obj_points = board.getChessboardCorners()
            ok, rvec, tvec = cv2.solvePnP(obj_points[charuco_ids.flatten()], charuco_corners, K, dist)
            if not ok:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            # Apply Kalman filter based on mode
            R_measured, _ = cv2.Rodrigues(rvec)
            t_measured = tvec.reshape(3)
            if FAST_MODE:
                # Skip Kalman filter for faster processing
                t_filtered, R_filtered = t_measured, R_measured
            else:
                # Apply Kalman filter to reduce jitter
                t_filtered, R_filtered = state['kalman_filter'].update(t_measured, R_measured)

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

            return R_filtered, t_filtered

        return None

    # locked path: reuse board/detector and detect board directly
    board, det = state['board'], state['detector']
    
    # Use the new CharucoDetector API
    charuco_corners, charuco_ids, marker_corners, marker_ids = det.detectBoard(undistorted_img)
    state['last_charuco'] = 0 if charuco_ids is None else int(len(charuco_ids))
    state['last_markers'] = 0 if marker_ids is None else int(len(marker_ids))
    
    if debug_img is not None and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
        # Draw bounding boxes around detected elements
        draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)

    if charuco_ids is None or len(charuco_ids) < 10:
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
        return None

    # Apply subpixel refinement based on mode
    if charuco_corners is not None and len(charuco_corners) > 0 and not FAST_MODE:
        # Convert to float32 for subpixel refinement
        ch_corners_float = np.array(charuco_corners, dtype=np.float32)
        # Convert BGR to grayscale for subpixel refinement
        gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        # Apply subpixel corner refinement
        cv2.cornerSubPix(gray_img, ch_corners_float, (5, 5), (-1, -1), 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        charuco_corners = ch_corners_float

    # Use solvePnP directly with board object points
    obj_points = board.getChessboardCorners()
    ok, rvec, tvec = cv2.solvePnP(obj_points[charuco_ids.flatten()], charuco_corners, K, dist)
    if not ok:
        if debug_img is not None:
            draw_counts(debug_img, 0, state['last_charuco'])
        return None

    # Apply Kalman filter based on mode
    R_measured, _ = cv2.Rodrigues(rvec)
    t_measured = tvec.reshape(3)
    if FAST_MODE:
        # Skip Kalman filter for faster processing
        t_filtered, R_filtered = t_measured, R_measured
    else:
        # Apply Kalman filter to reduce jitter
        t_filtered, R_filtered = state['kalman_filter'].update(t_measured, R_measured)

    if debug_img is not None:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
        # Draw bounding boxes around detected elements
        draw_bounding_boxes(debug_img, charuco_corners, charuco_ids, marker_corners, marker_ids)
        draw_counts(debug_img, state['last_markers'], state['last_charuco'])

    return R_filtered, t_filtered

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
        t_bg = np.array([x, y, z], dtype=np.float64) / 1000.0  # mm -> m
        R_bg = euler_rpy_to_R(roll, pitch, yaw, degrees=USE_DEG)
        return R_bg, t_bg

    def close(self):
        try: self.arm.disconnect()
        except Exception: pass

# =========================
# Residual diagnostics
# =========================
def handeye_residuals(Rg, tg, Rt, tt, R_cam2base, t_cam2base):
    """
    Calculate residuals for eye-in-hand hand-eye calibration.
    Rg: gripper poses in base frame (camera poses in base frame)
    tg: gripper translations in base frame
    Rt: target poses in camera frame
    tt: target translations in camera frame
    R_cam2base, t_cam2base: camera to base transformation
    """
    X = to_homogeneous(R_cam2base, t_cam2base)
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
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-Hand hand-eye calibration for Orbbec Camera (terminal-friendly)")
    parser.add_argument("--mode", choices=["gui", "ascii", "ascii_hi", "headless"],
                        default=("gui" if os.environ.get("DISPLAY") else "ascii"),
                        help="Display mode: OpenCV GUI, ASCII in terminal, or headless")
    parser.add_argument("--camera", choices=["auto", "orbbec", "realsense", "opencv"], default="orbbec",
                        help="Camera backend to use: Orbbec (default), RealSense (if available) or OpenCV UVC")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Enable fast mode (skip subpixel refinement and Kalman filtering)")
    parser.add_argument("--accurate", action="store_true", default=False,
                        help="Enable accurate mode (slower but more precise)")
    parser.add_argument("--ultra-fast", action="store_true", default=False,
                        help="Enable ultra-fast mode (maximum speed, lower resolution)")
    args = parser.parse_args()
    
    # Set performance mode based on arguments
    global FAST_MODE, ULTRA_FAST_MODE, SKIP_UNDISTORTION, ORBBEC_WIDTH, ORBBEC_HEIGHT
    
    if args.ultra_fast:
        ULTRA_FAST_MODE = True
        FAST_MODE = True
        SKIP_UNDISTORTION = True
        ORBBEC_WIDTH = 1280  # Lower resolution for speed
        ORBBEC_HEIGHT = 720
        print("🚀 Running in ULTRA-FAST mode (maximum speed, lower resolution)")
    elif args.accurate:
        FAST_MODE = False
        ULTRA_FAST_MODE = False
        SKIP_UNDISTORTION = False
        print("🔬 Running in ACCURATE mode (slower but more precise)")
    else:
        FAST_MODE = True
        ULTRA_FAST_MODE = False
        SKIP_UNDISTORTION = False
        print("⚡ Running in FAST mode (faster but less precise)")

    print("\n=== Instructions ===")
    print("• Mount Orbbec camera rigidly on the gripper (eye-in-hand).")
    print("• Fix ChArUco board rigidly in the environment.")
    print("• Move to varied poses (large rotations + translations).")
    if args.mode == "gui":
        print("• Press [SPACE] to capture, [q] to finish in the GUI window.\n")
    else:
        print("• In terminal, press SPACE to capture, q to finish.\n")

    print("Opening Orbbec camera...")
    orbbec_cam = OrbbecSource(ORBBEC_WIDTH, ORBBEC_HEIGHT, ORBBEC_FPS, camera_kind=args.camera)

    print("Connecting xArm...")
    xarm = XArmClient(XARM_IP)

    ch_state = make_charuco_state()
    R_g2b_list, t_g2b_list = [], []  # gripper to base (camera pose)
    R_t2c_list, t_t2c_list = [], []  # target to camera
    captured_images = []  # Store captured images for saving

    last_Rg, last_tg = None, None
    frame_skip_counter = 0  # For frame skipping in ultra-fast mode

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

            # Skip frames in ultra-fast mode for even more speed
            if ULTRA_FAST_MODE:
                frame_skip_counter += 1
                if frame_skip_counter % 2 != 0:  # Process every other frame
                    continue

            # Undistort the image for accurate ArUco detection (skip in ultra-fast mode)
            if SKIP_UNDISTORTION:
                undistorted = color  # Use raw image for maximum speed
                vis = color.copy()
                det = estimate_charuco_pose(undistorted, K, dist, debug_img=vis, state=ch_state)
            else:
                undistorted = cv2.undistort(color, K, dist)
                vis = undistorted.copy()
                det = estimate_charuco_pose(undistorted, K, dist, debug_img=vis, state=ch_state)

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

                # In eye-in-hand: camera is on gripper, so gripper pose = camera pose
                R_g2b = R_bg  # gripper to base (same as camera to base)
                t_g2b = t_bg

                accept = True
                if last_Rg is not None:
                    dR, dt = rel_motion(last_Rg, last_tg, R_g2b, t_g2b)
                    ang = rot_angle_deg(dR); d = np.linalg.norm(dt)
                    if ang < MIN_ANGLE_DEG and d < MIN_TRANS_M:
                        print(f"Pose too similar (Δang={ang:.1f}°, Δt={d*1000:.1f} mm); move more.")
                        accept = False

                if accept:
                    R_g2b_list.append(R_g2b); t_g2b_list.append(t_g2b)
                    R, t = det
                    R_t2c_list.append(R); t_t2c_list.append(t)
                    captured_images.append(color.copy())  # Store the captured image
                    last_Rg, last_tg = R_g2b, t_g2b
                    
                    # Save pose pair immediately
                    pose_num = len(R_g2b_list)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    # Convert rotation matrices to Euler angles for saving
                    R_g2b_euler = cv2.Rodrigues(R_g2b)[0]
                    R_t2c_euler = cv2.Rodrigues(R)[0]  # Use R from det
                    
                    # Calculate base to gripper transformation (inverse of gripper to base)
                    R_base2gripper, t_base2gripper = invert_rt(R_g2b, t_g2b)
                    
                    # Save as JPG
                    img_name = f"{SAVE_DIR}/pose{pose_num:03d}.jpg"
                    cv2.imwrite(img_name, captured_images[-1])
                    
                    # Save as NPY with base to gripper transformation
                    np.save(f"{SAVE_DIR}/pose{pose_num:03d}.npy", {
                        "R_base2gripper": R_base2gripper,      # Base to gripper rotation
                        "t_base2gripper": t_base2gripper,      # Base to gripper translation
                        "R_gripper2base": R_g2b,               # Original gripper to base (for reference)
                        "t_gripper2base": t_g2b,               # Original gripper to base (for reference)
                        "R_target2cam": R,                      # Target to camera (from det)
                        "t_target2cam": t,                      # Target to camera (from det)
                        "R_base2gripper_euler": cv2.Rodrigues(R_base2gripper)[0],  # Base to gripper in Euler angles
                        "R_target2cam_euler": R_t2c_euler,                         # Target to camera in Euler angles
                        "pose_number": pose_num,
                        "timestamp": pose_num  # You could add actual timestamp here if needed
                    }, allow_pickle=True)
                    
                    print(f"Captured #{pose_num}  (markers:{ch_state['last_markers']}, charuco:{ch_state['last_charuco']})")
                    print(f"Saved → {img_name}")
                    print(f"Saved → {SAVE_DIR}/pose{pose_num:03d}.npy")
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
                R_gripper2base=Rg, t_gripper2base=tg,  # Camera poses in base frame
                R_target2cam=Rt,  t_target2cam=tt,     # Target poses in camera frame
                method=method
            )
            t = t.reshape(3)
            calibration_results[name] = {
                'R': R,
                't': t,
                'method': method
            }
            print(f"Method {name} → t={t}, R={R}")
        except Exception as e:
            print(f"Method {name} failed: {e}")
            calibration_results[name] = None
    
    # Use PARK method as default (most commonly used)
    if calibration_results.get('PARK') is not None:
        R_cam2base = calibration_results['PARK']['R']
        t_cam2base = calibration_results['PARK']['t']
        print(f"\nUsing PARK method as default")
    else:
        # Fallback to first available method
        for name, result in calibration_results.items():
            if result is not None:
                R_cam2base = result['R']
                t_cam2base = result['t']
                print(f"\nUsing {name} method as fallback")
                break
        else:
            raise RuntimeError("All calibration methods failed")

    # For eye-in-hand: camera is mounted on gripper
    # We want the transformation from camera frame to gripper frame
    # This is the fixed offset between camera and gripper TCP
    # We can calculate this from the calibration results
    
    # Since we have gripper poses in base frame and camera poses in base frame,
    # we can find the camera-to-gripper transformation
    # T_gripper2base = T_cam2base * T_gripper2cam
    # Therefore: T_gripper2cam = inv(T_cam2base) * T_gripper2base
    # And: T_cam2gripper = inv(T_gripper2cam)
    
    # For now, let's use the first pose to calculate this relationship
    if len(R_g2b_list) > 0:
        R_gripper2base = R_g2b_list[0]  # First gripper pose
        t_gripper2base = t_g2b_list[0]
        
        # T_gripper2base = T_cam2base * T_gripper2cam
        # T_gripper2cam = inv(T_cam2base) * T_gripper2base
        T_cam2base_4x4 = np.eye(4)
        T_cam2base_4x4[:3,:3] = R_cam2base
        T_cam2base_4x4[:3,3] = t_cam2base
        
        T_gripper2base_4x4 = np.eye(4)
        T_gripper2base_4x4[:3,:3] = R_gripper2base
        T_gripper2base_4x4[:3,3] = t_gripper2base
        
        # T_gripper2cam = inv(T_cam2base) * T_gripper2base
        T_gripper2cam = np.linalg.inv(T_cam2base_4x4) @ T_gripper2base_4x4
        
        # T_cam2gripper = inv(T_gripper2cam)
        T_cam2gripper = np.linalg.inv(T_gripper2cam)
        
        R_cam2gripper = T_cam2gripper[:3,:3]
        t_cam2gripper = T_cam2gripper[:3,3]
    else:
        # Fallback if no poses captured
        R_cam2gripper = np.eye(3)
        t_cam2gripper = np.zeros(3)
        T_cam2gripper = np.eye(4)

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== T_cam2gripper (camera to gripper transformation) ===")
    print(T_cam2gripper)
    print("\n=== t_cam2gripper (translation vector) ===")
    print(t_cam2gripper)
    
    # Determine the selected method name
    selected_method = "PARK" if calibration_results.get('PARK') is not None else "FALLBACK"
    if selected_method == "FALLBACK":
        for method_name, result in calibration_results.items():
            if result is not None:
                selected_method = method_name
                break
    
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

    res = handeye_residuals(Rg, tg, Rt, tt, R_cam2base, t_cam2base)
    if res:
        print("\nResiduals: "
              f"rot mean={res['rot_deg']['mean']:.3f}°, med={res['rot_deg']['median']:.3f}°, p95={res['rot_deg']['p95']:.3f}°; "
              f"trans mean={res['trans_m']['mean']:.4f} m, med={res['trans_m']['median']:.4f} m, p95={res['trans_m']['p95']:.4f} m")

    print(f"\n=== Eye-in-hand calibration complete ===")
    print(f"Total poses captured: {len(R_g2b_list)}")
    print(f"Pose pairs saved to: {SAVE_DIR}")
    print(f"Main calibration result saved to: ../output/{method_filename}.npz")
    print(f"Individual method results saved to: ../output/eyeinhand_orbbec_{{method}}.npz")
    
    # Print summary of all methods
    print(f"\n=== Calibration Methods Summary ===")
    for method_name, result in calibration_results.items():
        if result is not None:
            print(f"{method_name}: t={result['t']}, R_det={np.linalg.det(result['R']):.6f}")
        else:
            print(f"{method_name}: FAILED")

if __name__ == "__main__":
    main()
