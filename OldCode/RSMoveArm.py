#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# =========================
# CONFIG
# =========================

# RealSense settings (848x480)
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

# ChArUco board parameters (same as EyeToHandConvert.py)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M = 0.025   # square side length (meters)
MARKER_LEN_RATIO = 0.78     # marker to square ratio
MARKER_LEN_M = SQUARE_LEN_M * MARKER_LEN_RATIO

# Display settings
AXIS_LEN_M = 0.08

# xArm configuration
XARM_IP = '192.168.10.201'

# =========================
# Camera-to-Base Matrix (provided by user)
# =========================
print("Loading camera-to-base transformation matrix...")

# T_CB: Camera to Base transformation matrix
T_CB = np.array([[-0.958302,  0.283689, -0.034326,  0.428970],
    [-0.032867, -0.228749, -0.972930,  1.048211],
    [-0.283861, -0.931233,  0.228535,  0.142686],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])



print("✓ Using provided camera-to-base matrix:")
print("T_CB (Camera to Base):\n", T_CB)

# Validate that it's a proper transformation matrix
R = T_CB[:3, :3]
det_R = np.linalg.det(R)
if abs(det_R - 1.0) > 0.1:
    print(f"Warning: Determinant of rotation matrix is {det_R:.3f} (should be close to 1)")
else:
    print(f"✓ Rotation matrix determinant: {det_R:.6f} (valid)")

# Calculate T_BC (Base to Camera) for reference
T_BC = np.linalg.inv(T_CB)
print("T_BC (Base to Camera, inverse):\n", T_BC)

# =========================
# Camera Intrinsics (same as EyeToHandConvert.py)
# =========================
print("\nSetting up camera intrinsics...")

# Load RealSense intrinsics/distortion from calibration file
try:
    # First try to load RealSense-specific calibration
    calib = np.load("output/realsense_calibration.npz")
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    print("✓ Loaded RealSense intrinsics from realsense_calibration.npz")
    print("Camera matrix:\n", K)
    print("Distortion coefficients:", dist)
except Exception as e:
    print(f"✗ Error loading RealSense calibration: {e}")
    try:
        # Fallback to general calibration
        calib = np.load("output/calibration_data.npz")
        K = calib["camera_matrix"]
        dist = calib["dist_coeffs"]
        print("✓ Loaded camera calibration from calibration_data.npz")
        print("Camera matrix:\n", K)
        print("Distortion coefficients:", dist)
    except Exception as e2:
        print(f"✗ Error loading camera calibration: {e2}")
        print("Using default camera matrix (NOT RECOMMENDED for accurate results)...")
        # Default camera matrix for 848x480
        K = np.array([
            [385.75439453,   0.        , 322.67864990],
    [  0.        , 385.29150391, 237.11767578],
    [  0.        ,   0.        ,   1.        ]
        ])
        dist = np.array([-0.05688492,  0.06663066,  0.00022735,  0.00087410, -0.02116243])

# If an OpenCV call complains about shape, use:
# dist_cv = dist.reshape(1, -1)  # shape (1,5)
# or
# dist_cv = dist.reshape(-1, 1)  # shape (5,1)


# =========================
# RealSense source (same as EyeToHandConvert.py)
# =========================
class RealSenseSource:
    def __init__(self, width=848, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def close(self):
        self.pipeline.stop()

# =========================
# ChArUco detection (same as EyeToHandConvert.py)
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
    # Robust-ish defaults; good for prints/screens
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.detectInvertedMarker = True
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M, MARKER_LEN_M, aruco_dict
    )
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
    return state

def _adjust_ids(ids, offset, dict_sz):
    # ids is shape (N,1); keep shape
    ids_i = ids.astype(np.int32)
    ids_adj = ((ids_i - int(offset)) % dict_sz).astype(np.int32)
    return ids_adj

def estimate_charuco_pose(img_bgr, K, dist, debug_img=None, state=None):
    """
    Auto-lock dictionary and firstMarkerId from observed IDs.
    Returns (R, t) or None. Updates state['last_markers'], state['last_charuco'].
    """
    assert state is not None

    def draw_counts(img, markers, charuco):
        cv2.putText(img, f"markers:{markers}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(img, f"charuco:{charuco}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if not state['locked']:
        # try each common dict
        for did in _COMMON_DICTS:
            if did not in state['candidates']:
                state['candidates'][did] = _make_board_and_detector(did)
            board, det = state['candidates'][did]

            corners, ids, _ = det.detectMarkers(img_bgr)
            state['last_markers'] = 0 if ids is None else int(len(ids))
            if debug_img is not None and ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

            if ids is None or len(ids) < 4:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], 0)
                continue

            dict_sz = _dict_size(did)
            guess_off = int(np.min(ids))
            ids_adj = _adjust_ids(ids, guess_off, dict_sz)

            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids_adj, img_bgr, board)
            state['last_charuco'] = 0 if (retval is None) else int(retval)
            if retval is None or retval < 4:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, dist, None, None)
            if not ok:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            # Lock in
            state.update({'locked': True, 'dict_id': did, 'first_off': guess_off,
                          'board': board, 'detector': det})
            print(f"[ChArUco] Locked dict={did}, firstMarkerId={guess_off}, "
                  f"markers={len(ids)}, charuco={int(retval)}")

            if debug_img is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, ch_corners, ch_ids)
                draw_counts(debug_img, state['last_markers'], state['last_charuco'])

            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)

        return None

    # locked path: reuse board/detector and adjust ids each time
    board, det = state['board'], state['detector']
    corners, ids, _ = det.detectMarkers(img_bgr)
    state['last_markers'] = 0 if ids is None else int(len(ids))
    if debug_img is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

    if ids is None or len(ids) < 4:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], 0)
        return None

    dict_sz = _dict_size(state['dict_id'])
    ids_adj = _adjust_ids(ids, state['first_off'], dict_sz)

    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids_adj, img_bgr, board)
    state['last_charuco'] = 0 if (retval is None) else int(retval)
    if retval is None or retval < 4:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], state['last_charuco'])
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, dist, None, None)
    if not ok:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], state['last_charuco'])
        return None

    if debug_img is not None:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, ch_corners, ch_ids)
        draw_counts(debug_img, state['last_markers'], state['last_charuco'])

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)

# =========================
# xArm helper functions
# =========================
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    """Convert roll, pitch, yaw to rotation matrix"""
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx  # yaw-pitch-roll (ZYX)

def to_homogeneous(R, t):
    """Convert rotation matrix and translation vector to homogeneous transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

# =========================
# Main loop (same display as EyeToHandConvert.py)
# =========================
print("\nStarting main loop...")
print("Press 'q' to quit")

print("Opening RealSense...")
rs_cam = RealSenseSource(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)

print("Connecting to xArm...")
try:
    arm = XArmAPI(XARM_IP)
    arm.motion_enable(True)
    print("✓ Connected to xArm")
except Exception as e:
    print(f"✗ Error connecting to xArm: {e}")
    print("Continuing without robot connection...")
    arm = None

ch_state = make_charuco_state()

try:
    while True:
        ok, color = rs_cam.read()
        if not ok or color is None:
            print("Camera read failed")
            break

        vis = color.copy()
        det = estimate_charuco_pose(color, K, dist, debug_img=vis, state=ch_state)

        if det is not None:
            R, t = det
            cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(R)[0], t.reshape(3,1), AXIS_LEN_M)
            cv2.putText(vis, "Board detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            # Create transformation matrix from camera to board origin
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R
            T_cam_board[:3, 3] = t
            
            # Print the camera-to-board transformation matrix
            print(f"\n=== Frame - Camera to Board Origin Matrix ===")
            print("T_cam_board (4x4 transformation matrix):")
            print(T_cam_board)
            print(f"Board origin position in camera frame (mm): ({t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f})")
            
            # Also print in a more readable format
            print("\nCamera to Board Origin Matrix (formatted):")
            print("Rotation Matrix (3x3):")
            print(R)
            print("Translation Vector (3x1, mm):")
            print(f"[{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}]")
            
            # Calculate estimated end effector to base transformation
            T_EEB_Estimated = T_CB @ T_cam_board
            
            print(f"\n=== Estimated End Effector to Base (T_EEB_Estimated) ===")
            print("T_EEB_Estimated (4x4 transformation matrix):")
            print(T_EEB_Estimated)
            
            # Extract position and orientation from estimated EE
            t_ee_estimated = T_EEB_Estimated[:3, 3]
            R_ee_estimated = T_EEB_Estimated[:3, :3]
            
            # Convert rotation matrix to roll, pitch, yaw
            def matrix_to_rpy(R):
                """Convert rotation matrix to roll, pitch, yaw angles in degrees"""
                sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                singular = sy < 1e-6
                if not singular:
                    roll = np.arctan2(R[2, 1], R[2, 2])
                    pitch = np.arctan2(-R[2, 0], sy)
                    yaw = np.arctan2(R[1, 0], R[0, 0])
                else:
                    roll = np.arctan2(-R[1, 2], R[1, 1])
                    pitch = np.arctan2(-R[2, 0], sy)
                    yaw = 0
                return np.degrees([roll, pitch, yaw])
            
            rpy_ee_estimated = matrix_to_rpy(R_ee_estimated)
            
            print(f"Estimated EE position in base frame (mm): ({t_ee_estimated[0]*1000:.1f}, {t_ee_estimated[1]*1000:.1f}, {t_ee_estimated[2]*1000:.1f})")
            print(f"Estimated EE orientation in base frame (deg): roll={rpy_ee_estimated[0]:.1f}, pitch={rpy_ee_estimated[1]:.1f}, yaw={rpy_ee_estimated[2]:.1f}")
            
            # Get actual end effector to base transformation from xArm
            if arm is not None:
                try:
                    code, pose = arm.get_position()
                    if code == 0:  # Success
                        # Robot pose is in mm, convert to meters
                        x, y, z, roll, pitch, yaw = pose
                        t_ee_base = np.array([x, y, z]) / 1000.0  # mm to meters
                        R_ee_base = euler_rpy_to_R(roll, pitch, yaw, degrees=True)
                        
                        # Create T_EEB (End Effector to Base) transformation matrix
                        T_EEB = to_homogeneous(R_ee_base, t_ee_base)
                        
                        print(f"\n=== Actual End Effector to Base (T_EEB) ===")
                        print("T_EEB (4x4 transformation matrix):")
                        print(T_EEB)
                        print(f"Actual EE position in base frame (mm): ({x:.1f}, {y:.1f}, {z:.1f})")
                        print(f"Actual EE orientation in base frame (deg): roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
                        
                        # Calculate error between estimated and actual
                        def calculate_pose_error(T_estimated, T_actual):
                            """Calculate translation and rotation error between two poses"""
                            # Translation error
                            t_error = T_estimated[:3, 3] - T_actual[:3, 3]
                            translation_error_mm = np.linalg.norm(t_error) * 1000  # Convert to mm
                            
                            # Rotation error
                            R_error = T_estimated[:3, :3] @ T_actual[:3, :3].T
                            trace_R = np.trace(R_error)
                            # Handle numerical issues that can cause NaN
                            if abs(trace_R - 3.0) < 1e-6:
                                angle_error_deg = 0.0  # Very small rotation error
                            else:
                                cos_angle = np.clip((trace_R - 1) / 2, -1, 1)
                                angle_error_deg = np.degrees(np.arccos(cos_angle))
                            
                            return translation_error_mm, angle_error_deg, t_error
                        
                        trans_error_mm, rot_error_deg, trans_error_vec = calculate_pose_error(T_EEB_Estimated, T_EEB)
                        
                        print(f"\n=== Error Analysis ===")
                        print(f"Translation Error: {trans_error_mm:.2f} mm")
                        print(f"Translation Error Vector (mm): x={trans_error_vec[0]*1000:.1f}, y={trans_error_vec[1]*1000:.1f}, z={trans_error_vec[2]*1000:.1f}")
                        print(f"Rotation Error: {rot_error_deg:.2f} degrees")
                        
                    else:
                        print(f"Failed to get robot pose, error code: {code}")
                except Exception as e:
                    print(f"Error getting robot pose: {e}")
            else:
                print("No robot connection - T_EEB not available")
            
        else:
            cv2.putText(vis, "No board", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("RealSense", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    rs_cam.close()
    if arm is not None:
        arm.disconnect()
    print("Program finished.")
