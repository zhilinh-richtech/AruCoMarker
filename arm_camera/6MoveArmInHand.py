#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI

# ----------------- Config -----------------
XARM_IP = '192.168.2.112'  # 根据实际情况修改
ARUCO_DICT = cv2.aruco.DICT_4X4_250
MARKER_LENGTH = 0.05  # 单位：米
save_dir = 'output_left'

# ----------------- Utility Functions -----------------
def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rpy_to_matrix(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
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

# ----------------- Connect to xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ----------------- Load camera calibration -----------------
with np.load(f"./{save_dir}/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# ----------------- Load hand-eye calibration -----------------
data = np.load(f'./{save_dir}/eye_to_hand_calibration.npz')
T_cam_gripper = data['T_cam_gripper']  # 4x4 齐次矩阵

# ----------------- Setup camera -----------------
cap = cv2.VideoCapture(10)
# 设置分辨率和帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("[INFO] 按ESC退出。检测到marker会显示其在基座下的坐标和姿态。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        break

    corners, ids, _ = detector.detectMarkers(frame)
    marker_pos_base = None
    marker_rpy_base = None
    marker_detected = False

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        # 只处理第一个检测到的marker
        R_marker_cam, _ = cv2.Rodrigues(rvec[0])
        t_marker_cam = tvec[0].flatten()
        T_marker_cam = to_homogeneous(R_marker_cam, t_marker_cam)

        # 获取机械臂末端在基座下的位姿
        code, pose = arm.get_position()
        t_gripper_base = np.array(pose[:3]) / 1000.0  # mm -> m
        R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
        T_gripper_base = to_homogeneous(R_gripper_base, t_gripper_base)

        # 计算marker在基座下的位姿
        T_marker_base = T_gripper_base @ T_cam_gripper @ T_marker_cam
        marker_pos_base = T_marker_base[:3, 3]
        marker_rpy_base = matrix_to_rpy(T_marker_base[:3, :3])
        marker_detected = True

        # 显示在画面上
        text = f"Marker@Base: x={marker_pos_base[0]:.3f} y={marker_pos_base[1]:.3f} z={marker_pos_base[2]:.3f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text2 = f"rpy(deg): {marker_rpy_base[0]:.1f}, {marker_rpy_base[1]:.1f}, {marker_rpy_base[2]:.1f}"
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 画坐标轴
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.03)

    else:
        cv2.putText(frame, "Marker not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('MoveArmInHand (Eye-in-Hand)', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break
    if key == ord('p'):
        if marker_detected:
            print(f"[P] Marker@Base: x={marker_pos_base[0]:.4f}, y={marker_pos_base[1]:.4f}, z={marker_pos_base[2]:.4f}")
            print(f"    rpy(deg): {marker_rpy_base[0]:.2f}, {marker_rpy_base[1]:.2f}, {marker_rpy_base[2]:.2f}")
        else:
            print("[P] Marker not detected!")

cap.release()
cv2.destroyAllWindows()
arm.disconnect() 