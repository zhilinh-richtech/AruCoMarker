#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import os
import sys
from typing import Tuple

# ----------------- Config -----------------
XARM_IP = '192.168.2.112'
CAM_INDEX = 10
FRAME_SIZE = (1280, 720)
FPS = 30

# Your Charuco board (update if different!)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LEN_M = 0.02474
CHARUCO_MARKER_LEN_M = 0.78 * CHARUCO_SQUARE_LEN_M  # default from calib.io unless you changed it

AXIS_LEN_M = 0.05  # draw axis length for visualization
save_dir = 'output_left'

# ----------------- Helpers -----------------
def to_homogeneous(R, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rpy_to_matrix(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0            ],
                   [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def make_charuco_board(aruco_dict) -> "cv2.aruco_CharucoBoard":
    # OpenCV API compatibility: new vs old constructors
    try:
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LEN_M,
            CHARUCO_MARKER_LEN_M,
            aruco_dict
        )
    except Exception:
        board = cv2.aruco.CharucoBoard_create(
            CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
            CHARUCO_SQUARE_LEN_M, CHARUCO_MARKER_LEN_M,
            aruco_dict
        )
    return board

def make_detector_params():
    # OpenCV API compatibility
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return params, detector
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        return params, None

def interpolate_charuco(corners, ids, image, board, K, dist):
    # OpenCV returns (charucoCorners, charucoIds) or + extra; normalize here
    out = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board, K, dist)
    if isinstance(out, tuple) and len(out) >= 2:
        charuco_corners, charuco_ids = out[0], out[1]
    else:
        charuco_corners, charuco_ids = out, None
    return charuco_corners, charuco_ids

def estimate_charuco_pose(charuco_corners, charuco_ids, board, K, dist) -> Tuple[bool, np.ndarray, np.ndarray]:
    # OpenCV newer returns (retval, rvec, tvec); older may always return rvec,tvec with validity check
    try:
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist, None, None
        )
        valid = bool(retval)
    except Exception:
        rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist, None, None
        )
        valid = rvec is not None and tvec is not None
    return valid, rvec, tvec

# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ----------------- Load camera intrinsics -----------------
calib_path = os.path.join(save_dir, "charuco_calibration.npz")
if not os.path.exists(calib_path):
    print(f"❌ Missing intrinsics: {calib_path}")
    sys.exit(1)

with np.load(calib_path) as data:
    camera_matrix = data["camera_matrix"].astype(np.float64)
    dist_coeffs = data["dist_coeffs"].astype(np.float64)

# ----------------- Camera -----------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
cap.set(cv2.CAP_PROP_FPS, FPS)
if not cap.isOpened():
    print("❌ Failed to open camera")
    sys.exit(1)

# ----------------- ArUco/Charuco setup -----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = make_charuco_board(aruco_dict)
params, detector = make_detector_params()
if detector is None:
    # Old API path: detection will be via detectMarkers(...)
    pass

# ----------------- Lists for pose pairs -----------------
R_gripper2base, t_gripper2base = [], []
R_target2cam,  t_target2cam  = [], []

# -------- Load previous samples (optional persistence) --------
os.makedirs(save_dir, exist_ok=True)
sample_path = os.path.join(save_dir, 'handeye_samples.npz')
if os.path.exists(sample_path):
    data_prev = np.load(sample_path)
    R_gripper2base = list(data_prev['R_gripper2base'])
    t_gripper2base = list(data_prev['t_gripper2base'])
    R_target2cam   = list(data_prev['R_target2cam'])
    t_target2cam   = list(data_prev['t_target2cam'])
    print(f"[INFO] Loaded {len(R_gripper2base)} previous samples.")
else:
    print("[INFO] No previous samples. Starting fresh.")

pose_count = len(R_gripper2base)
print("[INFO] Move robot to varied poses. Press 's' to save a sample, 'q' to calibrate/quit.")

last_valid = {"rvec": None, "tvec": None, "have_pose": False}

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Detect markers (API split: new detector vs legacy)
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    # Draw detected markers
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Charuco interpolation
        charuco_corners, charuco_ids = interpolate_charuco(
            corners, ids, frame, board, camera_matrix, dist_coeffs
        )

        if charuco_ids is not None and len(charuco_ids) >= 4:
            valid, rvec, tvec = estimate_charuco_pose(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs
            )
            if valid:
                # Draw pose axis (Board → Camera)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, AXIS_LEN_M)
                last_valid.update({"rvec": rvec, "tvec": tvec, "have_pose": True})
            else:
                last_valid["have_pose"] = False
        else:
            last_valid["have_pose"] = False
    else:
        last_valid["have_pose"] = False

    # HUD
    cv2.putText(frame, f"samples: {pose_count}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 's' to save pose, 'q' to run calibration",
                (15, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Charuco Eye-in-Hand', frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('s'):
        if last_valid["have_pose"]:
            # Board → Camera
            R_board_cam, _ = cv2.Rodrigues(last_valid["rvec"])
            t_board_cam = last_valid["tvec"].reshape(3).astype(np.float64)

            # Gripper → Base from robot (xArm returns mm + RPY deg)
            code, pose = arm.get_position()
            if code != 0 or pose is None:
                print("[ERROR] Failed to read robot pose; sample not saved.")
                continue

            t_grip_base_m = np.array(pose[:3], dtype=np.float64) / 1000.0
            R_grip_base   = rpy_to_matrix(pose[3], pose[4], pose[5])

            # Save both transforms for calibrateHandEye
            R_gripper2base.append(R_grip_base)
            t_gripper2base.append(t_grip_base_m)
            R_target2cam.append(R_board_cam)
            t_target2cam.append(t_board_cam)

            pose_count += 1
            print(f"[INFO] Pose #{pose_count} saved.")

            # Persist samples
            np.savez(sample_path,
                     R_gripper2base=np.array(R_gripper2base, dtype=object),
                     t_gripper2base=np.array(t_gripper2base, dtype=object),
                     R_target2cam=np.array(R_target2cam, dtype=object),
                     t_target2cam=np.array(t_target2cam, dtype=object))
            print(f"[INFO] Samples written to {sample_path}")
        else:
            print("[WARN] No valid Charuco pose this frame; hold the board steady and try again.")

    elif key == ord('q'):
        break

# ----------------- Solve Eye-in-Hand Calibration -----------------
if pose_count >= 3:
    print("\n[INFO] Running eye-in-hand calibration...")
    # cv2.calibrateHandEye expects lists of R (3x3) and t (3x1)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        [t.reshape(3, 1) for t in t_gripper2base],
        R_target2cam,
        [t.reshape(3, 1) for t in t_target2cam]
    )

    T_cam_gripper = to_homogeneous(R_cam2gripper, t_cam2gripper.reshape(3))
    T_gripper_cam = np.linalg.inv(T_cam_gripper)

    print("\n=== Calibration Result (Eye-in-Hand) ===")
    print("T_cam_gripper (Camera → Gripper):\n", T_cam_gripper)
    print("\nT_gripper_cam (Gripper → Camera):\n", T_gripper_cam)

    np.savez(os.path.join(save_dir, 'eye_to_hand_calibration.npz'),
             T_cam_gripper=T_cam_gripper,
             T_gripper_cam=T_gripper_cam)
    print(f"\n✅ Saved: {os.path.join(save_dir, 'eye_to_hand_calibration.npz')}")
else:
    print("❌ Not enough pose samples! Need at least 3 (8–15 recommended for accuracy).")

# ----------------- Cleanup -----------------
cap.release()
cv2.destroyAllWindows()
arm.disconnect()
