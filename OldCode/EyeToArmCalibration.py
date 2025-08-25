#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import os
import time

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'  # << Change to your xArm IP
ARUCO_DICT = cv2.aruco.DICT_4X4_250
MARKER_LENGTH = 0.07  # Marker side length in meters

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

# ----------------- Load camera calibration -----------------
with np.load("./output/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# ----------------- Setup camera -----------------
def gstreamer_pipeline(sensor_id=0,
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

cap = cv2.VideoCapture(gstreamer_pipeline(
    sensor_id=1,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2
), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

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
    ret, frame = cap.read()
    if not ret:
        continue

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

    T_cam_base = to_homogeneous(R_cam2gripper, t_cam2gripper)
    T_gripper_cam = np.linalg.inv(T_cam_base)

    print("\n=== Calibration result ===")
    print("T_cam_base (Camera to Gripper):\n", T_cam_base)
    print("\nT_gripper_cam (Gripper to Camera):\n", T_gripper_cam)

    # np.savez('./output/eye_to_hand_calibration.npz',
    #          T_cam_base=T_cam_base,
    #          T_gripper_cam=T_gripper_cam)
    # -----------------------------------------------------------
    # ➊  Compute constant marker → TCP transform (T_marker_tcp)
    #     We use the first saved pose; for higher accuracy you can
    #     loop over all poses and average quaternions / translations.
    # -----------------------------------------------------------
    # First pose components
    R_marker_cam0 = R_target2cam[0]
    t_marker_cam0 = t_target2cam[0]
    R_gripper_base0 = R_gripper2base[0]
    t_gripper_base0 = t_gripper2base[0]

    # (keep everything above unchanged …)

    # --------  constant offset  --------
    T_marker_cam0   = to_homogeneous(R_marker_cam0,  t_marker_cam0)      #  ^C T_M
    T_gripper_base0 = to_homogeneous(R_gripper_base0, t_gripper_base0)   #  ^B T_G
    T_marker_base0  = T_cam_base @ T_marker_cam0                         #  ^B T_M

    # ✅  correct order:  ^M T_G = (^B T_M)⁻¹ · ^B T_G
    T_marker_tcp    = np.linalg.inv(T_marker_base0) @ T_gripper_base0

    np.savez('./output/eye_to_hand_calibration.npz',
            T_cam_base   = T_cam_base,
            T_marker_tcp = T_marker_tcp)        #  (drop the old wrong file)


    save_path = './output/handeye_samples.npz'
    if os.path.exists(save_path):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = save_path.replace('.npz', f'_{timestamp}.npz')
        os.rename(save_path, backup_path)
        print(f"[WARNING] Existing file backed up to: {backup_path}")
        np.savez(save_path,
         R_gripper2base=np.array(R_gripper2base),
         t_gripper2base=np.array(t_gripper2base),
         R_target2cam=np.array(R_target2cam),
         t_target2cam=np.array(t_target2cam))

else:
    print("Not enough poses collected! Need at least 3.")

# ----------------- Cleanup -----------------
cap.release()
cv2.destroyAllWindows()
arm.disconnect()
