#!/usr/bin/env python3
import os

import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs

# 这个脚本用来获取深度相机的的坐标转换，但是没来得及测试是否准确。
# ----------------- Config -----------------
XARM_IP = '192.168.2.112'  #
ARUCO_DICT = cv2.aruco.DICT_4X4_250
MARKER_LENGTH = 0.05  # Marker side length in meters
save_dir = 'output_depth'

# ----------------- Functions -----------------
def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rpy_to_matrix(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
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

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
    if eye_to_hand:
        # Convert gripper→base to base→gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # Calibrate
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam
    )
    return R_cam2gripper, t_cam2gripper

# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)


# ----------------- Setup camera -----------------
# 配置 RealSense 流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取相机内参
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

print("[INFO] RealSense Camera Intrinsics:")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs)
np.savez(os.path.join(save_dir, "charuco_calibration.npz"),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)



aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ----------------- Lists for poses -----------------
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

pose_count = 0
print("[INFO] Move robot to different poses. Press 's' to save pose, 'q' to quit and calibrate.")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and MARKER_LENGTH > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.03)

        cv2.imshow('Aruco Detection', frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s'):
            # -------- Get marker pose in camera frame --------
            R_marker_cam, _ = cv2.Rodrigues(rvec[0])
            t_marker_cam = tvec[0].flatten()

            # -------- Get robot TCP pose in base frame --------
            code, pose = arm.get_position()
            t_gripper_base = np.array(pose[:3]) / 1000.0  # mm to meters
            R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])

            # -------- Append to lists --------
            R_gripper2base.append(R_gripper_base)
            t_gripper2base.append(t_gripper_base)

            R_target2cam.append(R_marker_cam)
            t_target2cam.append(t_marker_cam)

            pose_count += 1
            print(f"[INFO] Pose #{pose_count} saved.")
            print("--- Gripper to Base ---")
            print("Translation (m):", t_gripper_base)
            print("Rotation matrix:\n", R_gripper_base)

            print("--- Marker to Camera ---")
            print("Translation (m):", t_marker_cam)
            print("Rotation matrix:\n", R_marker_cam)

        elif key == ord('q'):
            break

    else:
        cv2.imshow('Aruco Detection', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

# ----------------- Solve hand-eye calibration -----------------
if pose_count >= 3:
    R_cam2gripper, t_cam2gripper = calibrate_eye_hand(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        eye_to_hand=True
    )

    T_cam_gripper = to_homogeneous(R_cam2gripper, t_cam2gripper)
    T_gripper_cam = np.linalg.inv(T_cam_gripper)

    print("\n=== Calibration result ===")
    print("T_cam_gripper (Camera to Gripper):\n", T_cam_gripper)
    print("\nT_gripper_cam (Gripper to Camera):\n", T_gripper_cam)

    np.savez(f'./{save_dir}/eye_to_hand_calibration.npz',
             T_cam_gripper=T_cam_gripper,
             T_gripper_cam=T_gripper_cam)
else:
    print("Not enough poses collected! Need at least 3.")

# ----------------- Cleanup -----------------
pipeline.stop()
cv2.destroyAllWindows()
arm.disconnect()
