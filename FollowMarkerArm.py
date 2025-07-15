#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI

# ---------- Load camera intrinsics ----------
with np.load("./output/calibration_data.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# ---------- Load eye-to-hand calibration ----------
R_cam2gripper = np.load("R_cam2gripper.npy")
t_cam2gripper = np.load("t_cam2gripper.npy")

# ---------- Connect to xArm ----------
arm = XArmAPI('192.168.10.201')  # Change to your IP
arm.motion_enable(True)

# ---------- ArUco setup ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_len = 50  # mm
TRACK_ID = 0

# ---------- RPY to matrix ----------
def rpy_to_matrix(rpy_angles_deg):
    rx, ry, rz = np.deg2rad(rpy_angles_deg)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

# ---------- GStreamer pipeline ----------
def gstreamer_pipeline(sensor_id=1,
                       capture_width=1280, capture_height=720,
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
    raise RuntimeError("❌ Failed to open camera")

print("🚨 Move robot to pose with marker visible. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        continue

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and TRACK_ID in ids:
        idx = int(np.where(ids == TRACK_ID)[0][0])
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx:idx+1], marker_len, camera_matrix, dist_coeffs
        )
        cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                          rvec[0], tvec[0][0], 30)  # axes length in mm

        # Marker position in camera frame (mm)
        t_marker_cam = tvec[0][0].reshape((3, 1))

        # Transform to gripper frame (mm)
        t_marker_gripper = R_cam2gripper @ t_marker_cam + t_cam2gripper

        # Get robot TCP pose (mm)
        code, pose = arm.get_position(is_radian=False)
        if code != 0 or not pose or len(pose) < 6:
            print("Error: failed to get robot pose.")
            continue

        xyz_base = np.array(pose[:3], dtype=np.float64).reshape(3, 1)  # mm
        rpy = pose[3:6]
        R_G2B = rpy_to_matrix(rpy)

        # Transform marker to base frame (mm)
        t_marker_base = R_G2B @ t_marker_gripper + xyz_base

        # Compare
        est_base = t_marker_base.flatten()
        real_base = xyz_base.flatten()
        diff_vec = est_base - real_base
        error_norm = np.linalg.norm(diff_vec)

        print(f"🎯 Estimated marker position (base frame): {est_base}")
        print(f"🤖 Robot TCP position (base frame):      {real_base}")
        print(f"⚖️  Difference vector (mm):             {diff_vec}")
        print(f"✅ Euclidean error norm (mm):           {error_norm:.2f}\n")

        cv2.putText(frame, f"Error: {error_norm:.1f} mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Marker not detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Gripper & Base Frame View", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
