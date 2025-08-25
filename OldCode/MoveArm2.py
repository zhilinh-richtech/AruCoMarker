#!/usr/bin/env python3
import cv2
import numpy as np
import time
import math
from xarm.wrapper import XArmAPI

XARM_IP = '192.168.10.201'

# ---------- Camera calibration ----------
with np.load("./output/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

data = np.load('./output/eye_to_hand_calibration.npz')
T_cam_base = data['T_cam_base']
print("Camera Matrix:\n", camera_matrix)
print("T_cam_base:\n", T_cam_base)

# ---------- Connect xArm ----------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ---------- ArUco setup ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_len = 0.05
TRACK_ID = 0

# ---------- GStreamer pipeline ----------
def gstreamer_pipeline(sensor_id=1, capture_width=1280, capture_height=720,
                       display_width=1280, display_height=720,
                       framerate=30, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open CSI camera via GStreamer")

def rpy_to_matrix(roll, pitch, yaw):
    roll, pitch, yaw = map(np.radians, [roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(R):
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

def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

# ---------- Calibration phase ----------
init_duration = 5
padding = 0
samples = []
bounds = None
start_time = time.time()

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and TRACK_ID in ids:
        idx = int(np.where(ids == TRACK_ID)[0][0])
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, camera_matrix, dist_coeffs)
        rvec = rvec[0]
        tvec = tvec[0][0]

        cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        R_marker_cam, _ = cv2.Rodrigues(rvec)
        T_marker_cam = to_homogeneous(R_marker_cam, tvec)
        T_marker_base_est = T_cam_base @ T_marker_cam

        code, pose = arm.get_position()
        t_gripper_base = np.array(pose[:3]) / 1000.0
        R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])

        if bounds is None:
            samples.append(tvec)
            cv2.putText(frame, "Calibrating...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            if time.time() - start_time >= init_duration:
                samples_np = np.array(samples)
                bounds = {
                    "min": samples_np.min(axis=0) - padding,
                    "max": samples_np.max(axis=0) + padding
                }
                avg = (bounds["min"] + bounds["max"]) / 2
                print("✅ Calibration finished; avg:", avg)

        print("=== Pose Comparison ===")
        print("Given    XYZ (m):", t_gripper_base)
        print("Estimated XYZ (m):", T_marker_base_est[:3, 3])
        print("Translation Error (mm):", (t_gripper_base - T_marker_base_est[:3, 3]) * 1000)

        rpy_given = matrix_to_rpy(R_gripper_base)
        rpy_est = matrix_to_rpy(T_marker_base_est[:3, :3])
        print("Given    RPY (deg):", rpy_given)
        print("Estimated RPY (deg):", rpy_est)
        print("Rotation Error (deg):", np.array(rpy_given) - np.array(rpy_est))

    else:
        cv2.putText(frame, "Marker 0 not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    h, w, _ = frame.shape
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

    cv2.imshow("ArUco Pose – ID 0 monitor", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
