#!/usr/bin/env python3
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'

# ---- ArUco marker parameters ----
ARUCO_DICT_ID     = cv2.aruco.DICT_4X4_250  # change if your tag uses a different family
TARGET_MARKER_ID  = 0                    # set an int (e.g., 23) if you want a specific ID
MARKER_LENGTH_M   = 0.108                   # 50 mm marker side length
AXIS_LEN_M        = 0.08

# ----------------- Eye-to-hand (Base ← Camera) -----------------
data = np.load('../output/markercalibration.npz')
T_cam_base = data['last_mark'].astype(np.float64)  # <-- verify this is actually Base ← Camera!
T_base_from_cam = T_cam_base  # ensure later code uses the same name

# If you want to override with a fixed matrix, uncomment and edit:
# T_cam_base = np.array([
#     [-0.982374, -0.084552, -0.166708,  0.573300],
#     [ 0.186508, -0.383933, -0.904328,  0.506070],
#     [ 0.012458, -0.919481,  0.392936,  0.139193],
#     [ 0.000000,  0.000000,  0.000000,  1.000000]
# ])


# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ----------------- Camera intrinsics -----------------
with np.load("../output/realsense_calibration.npz") as data_cal:
    K    = data_cal["camera_matrix"].astype(np.float64)
    dist = data_cal["dist_coeffs"].astype(np.float64)

# ----------------- RealSense color stream -----------------
REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS = 640, 480, 30
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT, rs.format.bgr8, REALSENSE_FPS)
pipeline.start(cfg)
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0,     0,      1 ]], dtype=np.float64)
dist = np.array(intr.coeffs, dtype=np.float64)

# ----------------- Helpers -----------------
# --- add this helper near your other helpers ---
def draw_axes_custom(img, K, dist, R_cm, tvec, axis_len=0.08):
    """
    Draw custom axes at the marker origin using existing pose (Camera ← Marker):
      red (x):   right  = +X_marker
      green (y): up     = -Y_marker
      blue (z):  out    = +Z_marker
    """
    # Directions in the marker frame
    dir_x_m = np.array([ 1.0,  0.0,  0.0], dtype=np.float64)  # right
    dir_y_m = np.array([ 0.0, 1.0,  0.0], dtype=np.float64)  # up (note: Y)
    dir_z_m = np.array([ 0.0,  0.0,  1.0], dtype=np.float64)  # out of marker plane

    # Convert to camera frame
    o_cam = tvec.reshape(3)
    x_cam = o_cam + R_cm @ (dir_x_m * axis_len)
    y_cam = o_cam + R_cm @ (dir_y_m * axis_len)
    z_cam = o_cam + R_cm @ (dir_z_m * axis_len)

    # Project (use identity extrinsics since points are already in camera frame)
    pts_cam = np.stack([o_cam, x_cam, y_cam, z_cam], axis=0).reshape(-1, 1, 3)
    rvec0 = np.zeros(3, dtype=np.float64)
    tvec0 = np.zeros(3, dtype=np.float64)
    pts_px, _ = cv2.projectPoints(pts_cam, rvec0, tvec0, K, dist)
    O, Xp, Yp, Zp = pts_px.reshape(-1, 2).astype(int)

    # Draw: red (x), green (y), blue (z) in BGR
    cv2.line(img, tuple(O), tuple(Xp), (0,   0, 255), 2)  # red
    cv2.line(img, tuple(O), tuple(Yp), (0, 255,   0), 2)  # green
    cv2.line(img, tuple(O), tuple(Zp), (255, 0,   0), 2)  # blue

    # optional end dots
    cv2.circle(img, tuple(Xp), 3, (0, 0, 255), -1)
    cv2.circle(img, tuple(Yp), 3, (0, 255, 0), -1)
    cv2.circle(img, tuple(Zp), 3, (255, 0, 0), -1)


def to_homogeneous(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.reshape(3)
    return T

def invert_se3(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti

def matrix_to_rpy(R):
    # ZYX (yaw-pitch-roll) -> returns (roll, pitch, yaw) in degrees
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])

def rpy_to_matrix(roll, pitch, yaw):
    roll  = np.radians(roll)
    pitch = np.radians(pitch)
    yaw   = np.radians(yaw)
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# ----------------- Movement check config -----------------
# TRANS_THRESH_MM = 20.0
# ROT_THRESH_DEG  = 20.0
thresh = 3
E_from_M_baseline = None  # baseline (EE ← Marker)

def T_from_pose_mm_rpy(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
    R = rpy_to_matrix(roll_deg, pitch_deg, yaw_deg)
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return to_homogeneous(R, t)

def transform_residual(T_ref, T_now):
    if T_ref is None or T_now is None:
        return None, None, None
    dT = invert_se3(T_ref) @ T_now
    dtrans_m = dT[:3, 3]                     # meters
    dpos_vec_mm = (dtrans_m * 1000.0).astype(float)  # mm per-axis [dx, dy, dz]
    dpos_mm = float(np.linalg.norm(dtrans_m) * 1000.0)
    dang_deg = float(rot_angle_deg(dT[:3, :3]))
    return dpos_vec_mm, dpos_mm, dang_deg


def rot_angle_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))

def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

def choose_marker_index(corners, ids):
    """Pick a specific marker ID if provided; otherwise choose the largest-area marker."""
    if ids is None or len(ids) == 0:
        return None
    if TARGET_MARKER_ID is not None:
        matches = np.where(ids.flatten() == TARGET_MARKER_ID)[0]
        if len(matches) > 0:
            return int(matches[0])
        # fall through to largest if target not found
    # choose by area
    areas = []
    for i, c in enumerate(corners):
        # c shape: (1, 4, 2)
        poly = c.reshape(4, 2).astype(np.float32)
        areas.append(cv2.contourArea(poly))
    return int(np.argmax(areas))

# ----------------- ArUco detector -----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
# aruco_params.adaptiveThreshWinSizeMin = 5
# aruco_params.adaptiveThreshWinSizeMax = 45
# aruco_params.adaptiveThreshWinSizeStep = 5
# aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# aruco_params.detectInvertedMarker = True
# aruco_params.minMarkerPerimeterRate = 0.03
# aruco_params.maxMarkerPerimeterRate = 4.0
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ----------------- Main loop -----------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Detect ArUco
        corners, ids, _ = detector.detectMarkers(frame)

        status_text = "NO TAG"
        moved = False
        dpos_mm = None
        dang_deg = None

        if ids is not None and len(ids) > 0:
            idx = choose_marker_index(corners, ids)
            if idx is not None:
                # Pose (Camera ← Marker)
                rvecs, tvecs, _objPts = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_M, K, dist
                )
                rvec = rvecs[idx].reshape(3)
                tvec = tvecs[idx].reshape(3)

                R_cm, _ = cv2.Rodrigues(rvec)
                T_cam_from_marker = to_homogeneous(R_cm, tvec)   # C ← M

                # Base ← Marker = (Base ← Camera) @ (Camera ← Marker)
                T_base_from_marker = T_base_from_cam @ T_cam_from_marker

                # Draw on image
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], ids[idx:idx+1])
                draw_axes_custom(frame, K, dist, R_cm, tvec, AXIS_LEN_M)

                # Robot pose (Base ← EE) from xArm
                code, pose = arm.get_position()  # [x(mm), y(mm), z(mm), roll, pitch, yaw]
                if code == 0 and pose is not None and len(pose) >= 6:
                    T_base_from_ee = T_from_pose_mm_rpy(
                        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
                    )

                    # EE ← Marker (what we care about: should be constant if marker==TCP)
                    T_ee_from_marker = invert_se3(T_base_from_ee) @ T_base_from_marker

                    # Initialize / load baseline if needed
                    if E_from_M_baseline is None:
                        E_from_M_baseline = T_ee_from_marker.copy()

                    # Residual vs baseline
                    # Residual vs baseline
                    dpos_vec_mm, dpos_mm, dang_deg = transform_residual(E_from_M_baseline, T_ee_from_marker)

                    if (dpos_mm is not None) and (dang_deg is not None):
                        print(dpos_vec_mm)
                        dx, dy, dz = dpos_vec_mm  # unpack x, y, z changes (mm)
                        moved = (abs(dx) > thresh) or (abs(dy) > thresh) or (abs(dz) > thresh)
                        status_text = (
                            f"pos={dpos_mm:.1f} mm  "
                            f"ang={dang_deg:.1f}deg  "
                            f"(x={dx:.1f}, y={dy:.1f}, z={dz:.1f} mm)"
                        )


                else:
                    status_text = "ROBOT POSE N/A"

        # --- HUD ---
        banner = "MOVED" if moved else ("STABLE" if dpos_mm is not None else "SEARCHING")
        color = (0, 0, 255) if moved else ((0, 200, 0) if dpos_mm is not None else (200, 200, 0))

        cv2.putText(frame, "ArUco Eye-to-Hand (Marker==TCP)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, banner, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        cv2.putText(frame, status_text, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)
        cv2.putText(frame, "[r] reset baseline   [s] save   [l] load   [q] quit", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        cv2.imshow("ArUco (RealSense)", frame)
        key = (cv2.waitKey(1) & 0xFF)
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset baseline to current reading if available
            try:
                E_from_M_baseline = T_ee_from_marker.copy()  # will raise if not defined this frame
            except:
                pass
        elif key == ord('s'):
            try:
                np.savez("../output/marker_tcp_baseline.npz", E_from_M_baseline=E_from_M_baseline)
            except Exception as e:
                print("Save baseline failed:", e)
        elif key == ord('l'):
            try:
                d = np.load("../output/marker_tcp_baseline.npz")
                E_from_M_baseline = d["E_from_M_baseline"].astype(np.float64)
            except Exception as e:
                print("Load baseline failed:", e)

finally:
    cv2.destroyAllWindows()
    try: pipeline.stop()
    except: pass
    try: arm.disconnect()
    except: pass
