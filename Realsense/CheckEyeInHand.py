#!/usr/bin/env python3
import numpy as np
import cv2
import pyrealsense2 as rs
from utils import (
    draw_axes_custom,
    to_homogeneous,
    invert_se3,
    matrix_to_rpy,
    rpy_to_matrix,
    rot_angle_deg,
    choose_marker_index,
)
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
    [-0.011733, -0.999899, -0.008041,  0.07153 ],
    [ 0.999915, -0.011778,  0.005573,  0.011725],
    [-0.005668, -0.007975,  0.999952, -0.131624],
    [ 0.0,       0.0,       0.0,       1.0     ]
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
# Use shared utilities from utils.py

# ----------------- ArUco detector -----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
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
            idx = choose_marker_index(corners, ids, TARGET_MARKER_ID)
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
                    print(actual_xyz)
                    print(estimated_xyz)
                    print(actual_xyz - estimated_xyz)
                    print(actual_rpy - estimated_rpy)
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
