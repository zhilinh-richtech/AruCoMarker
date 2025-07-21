#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import os

# ----------------- Config -----------------
XARM_IP = '192.168.2.112'
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

# ----------------- Connect to xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)


# ----------------- Load camera calibration -----------------
with np.load(f"./{save_dir}/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

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

# ----------------- Lists for pose pairs -----------------
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# ========== 新增：自动加载历史样本 ==========
sample_path = os.path.join(save_dir, 'handeye_samples.npz')
os.makedirs(save_dir, exist_ok=True)
if os.path.exists(sample_path):
    data = np.load(sample_path)
    R_gripper2base = list(data['R_gripper2base'])
    t_gripper2base = list(data['t_gripper2base'])
    R_target2cam = list(data['R_target2cam'])
    t_target2cam = list(data['t_target2cam'])
    print(f"[INFO] 已加载历史样本 {len(R_gripper2base)} 个")
else:
    print("[INFO] 未检测到历史样本，将从零开始采集")
# ========== 新增结束 ==========

pose_count = len(R_gripper2base)
print("[INFO] Move robot to various poses. Press 's' to save a sample, 'q' to calibrate.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.03)

        cv2.imshow('Aruco Detection', frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            # 检查是否只检测到一个id且为0
            if len(ids) == 1 and ids[0][0] == 0:
                # 获取 Marker 在相机坐标系下的位置
                R_marker_cam, _ = cv2.Rodrigues(rvec[0])
                t_marker_cam = tvec[0].flatten()

                # 获取机械臂末端位姿（单位：米）
                code, pose = arm.get_position()
                t_gripper_base = np.array(pose[:3]) / 1000.0
                R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])

                # 保存两组变换
                R_gripper2base.append(R_gripper_base)
                t_gripper2base.append(t_gripper_base)

                R_target2cam.append(R_marker_cam)
                t_target2cam.append(t_marker_cam)

                pose_count += 1
                print(f"[INFO] Pose #{pose_count} saved.")
                # ========== 新增：采集时自动保存 ==========
                np.savez(sample_path,
                         R_gripper2base=np.array(R_gripper2base),
                         t_gripper2base=np.array(t_gripper2base),
                         R_target2cam=np.array(R_target2cam),
                         t_target2cam=np.array(t_target2cam))
                print(f"[INFO] 样本已保存到 {sample_path}")
                # ========== 新增结束 ==========
            else:
                print(f"[ERROR] 保存失败：检测到的marker数量为{len(ids)}，id内容为{ids.flatten().tolist()}，必须且只能有一个id为0的marker！")

        elif key == ord('q'):
            break

    else:
        cv2.imshow('Aruco Detection', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

# ----------------- Solve Eye-in-Hand Calibration -----------------
if pose_count >= 3:
    print("\n[INFO] Running eye-in-hand calibration...")

    # 使用 Eye-in-Hand 解法
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam
    )

    T_cam_gripper = to_homogeneous(R_cam2gripper, t_cam2gripper)
    T_gripper_cam = np.linalg.inv(T_cam_gripper)

    print("\n=== Calibration Result (Eye-in-Hand) ===")
    print("T_cam_gripper (Camera → Gripper):\n", T_cam_gripper)
    print("\nT_gripper_cam (Gripper → Camera):\n", T_gripper_cam)

    # # 验证标定质量
    # print("\n=== Calibration Quality Check ===")
    # total_error = 0
    #
    # # 计算所有样本中标定板的平均位姿作为参考
    # all_target_positions = []
    # all_target_rotations = []
    # for i in range(len(R_target2cam)):
    #     T_target_cam_i = to_homogeneous(R_target2cam[i], t_target2cam[i])
    #     all_target_positions.append(T_target_cam_i[:3, 3])
    #     all_target_rotations.append(T_target_cam_i[:3, :3])
    #
    # # 计算平均位置和平均旋转
    # mean_target_position = np.mean(all_target_positions, axis=0)
    # mean_target_rotation = np.mean(all_target_rotations, axis=0)
    # # 重新正交化旋转矩阵
    # U, _, Vt = np.linalg.svd(mean_target_rotation)
    # mean_target_rotation = U @ Vt
    #
    # T_actual = to_homogeneous(mean_target_rotation, mean_target_position)
    # print(f"使用 {len(R_gripper2base)} 个样本的平均位姿作为参考")
    #
    # for i in range(len(R_gripper2base)):
    #     # 计算预测的标定板位姿
    #     T_gripper_base_i = to_homogeneous(R_gripper2base[i], t_gripper2base[i])
    #     T_target_cam_i = to_homogeneous(R_target2cam[i], t_target2cam[i])
    #     T_pred = T_gripper_base_i @ T_cam_gripper @ np.linalg.inv(T_target_cam_i)
    #     # 计算误差
    #     error = np.linalg.norm(T_pred[:3, 3] - T_actual[:3, 3])
    #     total_error += error
    #     print(f"Sample {i+1}: position error = {error:.4f} m")
    #
    # mean_error = total_error / len(R_gripper2base)
    # print(f"Mean position error: {mean_error:0.4f} m")
    # print(f"Recommendation: {'Good' if mean_error < 0.1 else 'Need more samples or better calibration'}")

    np.savez(f'./{save_dir}/eye_to_hand_calibration.npz',
             T_cam_gripper=T_cam_gripper,
             T_gripper_cam=T_gripper_cam)
    print(f"\n✅ Saved: ./{save_dir}/eye_to_hand_calibration.npz")

    # 标定完成后可选择清空样本文件（可选）
    # os.remove(sample_path)
else:
    print("❌ Not enough pose samples! You need at least 3.")

# ----------------- Cleanup -----------------
cap.release()
cv2.destroyAllWindows()
arm.disconnect()
