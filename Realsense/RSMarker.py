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
MARKER_LENGTH_M   = 0.072                   # 50 mm marker side length
AXIS_LEN_M        = 0.08

# ----------------- Eye-to-hand (Base ← Camera) -----------------
data = np.load('../output/poses/result.npy', allow_pickle=True)
#T_cam_base = data['T_cam2grip'].astype(np.float64)  # Base ← Camera
# If you want to override with a fixed matrix, uncomment and edit:
T_cam_base = np.array([
    [0.51097,  -0.559244,  0.652806, -1.189456],
    [0.742469, -0.095573, -0.663027, -1.593206],
    [0.433185,  0.823476,  0.366386,  0.600184],
    [0.0,       0.0,       0.0,       1.0]
])


# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
# arm.motion_enable(True)

# ----------------- Camera intrinsics -----------------
with np.load("../output/realsense_calibration.npz") as data_cal:
    K    = data_cal["camera_matrix"].astype(np.float64)
    dist = data_cal["dist_coeffs"].astype(np.float64)

# ----------------- RealSense color stream -----------------
REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS = 1280, 800, 30
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT, rs.format.bgr8, REALSENSE_FPS)
pipeline.start(cfg)

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
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    if degrees:
        roll = np.deg2rad(roll); pitch = np.deg2rad(pitch); yaw = np.deg2rad(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx  # yaw-pitch-roll (ZYX)
# ----------------- Main loop -----------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Detect single ArUco
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            idx = choose_marker_index(corners, ids)
            if idx is not None:
                # Pose of each detected marker: Camera ← Marker
                rvecs, tvecs, _objPts = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_M, K, dist
                )
                rvec = rvecs[idx].reshape(3)
                tvec = tvecs[idx].reshape(3)
                code, pose = arm.get_position(is_radian = False)  # degrees\
                x, y, z, roll, pitch, yaw = pose
                t_gb = np.array([x, y, z], dtype=np.float64) / 1000.0  # mm -> m
                R_gb = euler_rpy_to_R(roll, pitch, yaw, degrees=True)
                GripperTBase = to_homogeneous(R_gb, t_gb)
                R_cm, _ = cv2.Rodrigues(rvec)                   # Camera ← Marker rotation
                T_C_from_M = to_homogeneous(R_cm, tvec)         # Camera ← Marker
                T_B_from_M = GripperTBase@  T_cam_base @ T_C_from_M            # Base ← Marker

                # Draw axes on the chosen tag
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], ids[idx:idx+1])
                # cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN_M)  # old
                draw_axes_custom(frame, K, dist, R_cm, tvec, AXIS_LEN_M)     # new


                # Robot pose (Base ← EE)
                code, pose = arm.get_position()  # [x(mm), y(mm), z(mm), roll, pitch, yaw] (deg)
                if code == 0:
                    t_gripper_base = np.array(pose[:3], dtype=np.float64) / 1000.0  # mm → m
                    R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
                    T_base_gripper = to_homogeneous(R_gripper_base, t_gripper_base) 
                    # x, y, z, roll,DetectorParameterspitch, yaw = pose
                    # t_B_from_G = np.array([x, y, z], dtype=np.float64) / 1000.0
                    # R_B_from_G = rpy_to_matrix(roll, pitch, yaw)
                    # T_B_from_G = to_homogeneous(R_B_from_G, t_B_from_G)

                    # # Assume Marker == TCP → compare Base ← Marker vs Base ← EE
                    # delta      = T_B_from_M @ invert_se3(T_B_from_G)  # ~Identity if perfect
                    # d_t_mm     = delta[:3, 3]*1000
                    # d_R        = delta[:3, :3]
                    # d_ang_deg  = rot_angle_deg(d_R)

                    # # Pretty pose prints (xyz & rpy)
                    # est_xyz_mm = (T_B_from_M[:3, 3] * 1000.0)
                    # est_rpy    = matrix_to_rpy(T_B_from_M[:3, :3])
                    # act_xyz_mm = (T_B_from_G[:3, 3] * 1000.0)
                    # act_rpy    = matrix_to_rpy(T_B_from_G[:3, :3])

                    # d_rpy = np.array([wrap180(est_rpy[0]-act_rpy[0]),
                    #                   wrap180(est_rpy[1]-act_rpy[1]),
                    #                   wrap180(est_rpy[2]-act_rpy[2])])

                    print("\n=== ArUco (single tag) Pose Compare ===")
                    if TARGET_MARKER_ID is None:
                        print(f"Detected IDs: {ids.flatten().tolist()}  | using index {idx} (by largest area)")
                    else:
                        print(f"Detected IDs: {ids.flatten().tolist()}  | target={TARGET_MARKER_ID}, using idx={idx}")
                    print("--- Base ← Marker (from ArUco) ---")
                    actual_rpy = matrix_to_rpy(T_base_gripper)
                    actual_xyz = T_base_gripper[:3, 3]*1000
                    estimated_rpy = matrix_to_rpy(T_B_from_M)
                    estimated_xyz = T_B_from_M[:3, 3]*1000
                    # # Extra: numeric agreement, same-frame check
                    # print("delta (should be ~Identity):")
                    # print(delta)
                    # print(f"Δt (mm): [{d_t[0]*1000:.1f}, {d_t[1]*1000:.1f}, {d_t[2]*1000:.1f}]  |  Δangle: {d_ang:.2f} deg  |  ΔRPY: {d_rpy}")
                    print(actual_xyz)
                    print(estimated_xyz)
                    print(actual_xyz - estimated_xyz)
                    print(actual_rpy - estimated_rpy)
                    # print(f"Estimated (Base←Marker) xyz (mm): [{est_xyz_mm[0]:.2f}, {est_xyz_mm[1]:.2f}, {est_xyz_mm[2]:.2f}]")
                    # print(f"Estimated (Base←Marker) rpy (deg): [{est_rpy[0]:.2f}, {est_rpy[1]:.2f}, {est_rpy[2]:.2f}]")

                    # print(f"Actual    (Base←EE)     xyz (mm): [{act_xyz_mm[0]:.2f}, {act_xyz_mm[1]:.2f}, {act_xyz_mm[2]:.2f}]")
                    # print(f"Actual    (Base←EE)     rpy (deg): [{act_rpy[0]:.2f}, {act_rpy[1]:.2f}, {act_rpy[2]:.2f}]")

                    # print(f"Δ xyz (mm): [{d_t_mm[0]:.2f}, {d_t_mm[1]:.2f}, {d_t_mm[2]:.2f}]  |  ||Δt|| = {np.linalg.norm(d_t_mm):.2f} mm")
                    # print(f"Δ rpy (deg): [{d_rpy[0]:.2f}, {d_rpy[1]:.2f}, {d_rpy[2]:.2f}]  |  geodesic ΔR = {d_ang_deg:.2f}°")

        # HUD
        cv2.putText(frame, "ArUco Eye-to-Hand (Marker==TCP)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("ArUco (RealSense)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    try:
        pipeline.stop()
    except:
        pass
    try:
        arm.disconnect()
    except:
        pass
