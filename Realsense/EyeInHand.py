#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
import os
import sys
import signal
from typing import Tuple
import pyrealsense2 as rs

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'
REALSENSE_WIDTH  = 1280
REALSENSE_HEIGHT = 720   # use a supported RGB mode
REALSENSE_FPS    = 30

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LEN_M = 0.034
CHARUCO_MARKER_LEN_M = 0.78 * CHARUCO_SQUARE_LEN_M
AXIS_LEN_M = 0.05
save_dir = 'output_left'

# ----------------- Helpers -----------------
def to_homogeneous(R, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t.reshape(3)
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

def make_charuco_board(aruco_dict):
    try:
        return cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LEN_M,
            CHARUCO_MARKER_LEN_M,
            aruco_dict
        )
    except Exception:
        return cv2.aruco.CharucoBoard_create(
            CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
            CHARUCO_SQUARE_LEN_M, CHARUCO_MARKER_LEN_M,
            aruco_dict
        )

def make_detector_params(aruco_dict):
    try:
        params = cv2.aruco.DetectorParameters()
        # enable subpixel refinement for stability
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return params, detector
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        try:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        return params, None

def interpolate_charuco(corners, ids, gray, board, K, dist):
    # Optional: refine to board geometry (helps a lot)
    try:
        cv2.aruco.refineDetectedMarkers(
            image=gray, board=board, detectedCorners=corners, detectedIds=ids,
            rejectedCorners=None, cameraMatrix=K, distCoeffs=dist
        )
    except Exception:
        pass

    out = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, K, dist)
    if isinstance(out, tuple) and len(out) >= 2:
        charuco_corners, charuco_ids = out[0], out[1]
    else:
        charuco_corners, charuco_ids = out, None

    # Normalize outputs
    if charuco_corners is None or charuco_ids is None:
        return None, None
    if len(charuco_corners) == 0 or len(charuco_ids) == 0:
        return None, None

    # Ensure ids are Nx1 int32 (OpenCV expects this)
    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    charuco_corners = np.asarray(charuco_corners, dtype=np.float32)

    # Fix rare length mismatches (OpenCV 4.11 bug on some builds)
    n = min(len(charuco_corners), len(charuco_ids))
    if n < 4:
        return None, None
    if len(charuco_corners) != len(charuco_ids):
        charuco_corners = charuco_corners[:n]
        charuco_ids = charuco_ids[:n]

    return charuco_corners, charuco_ids


def estimate_charuco_pose(charuco_corners, charuco_ids, board, K, dist):
    # Guard: make sure lengths match *again*
    if (charuco_corners is None or charuco_ids is None or
        len(charuco_corners) != len(charuco_ids) or len(charuco_ids) < 4):
        return False, None, None

    # Newer API
    try:
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist, None, None
        )
        valid = bool(retval) and rvec is not None and tvec is not None
        return valid, rvec, tvec
    except Exception:
        # Legacy return signature
        rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist, None, None
        )
        valid = (rvec is not None and tvec is not None)
        return valid, rvec, tvec


# ----------------- RealSense wrapper -----------------
class RealSenseSource:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)

        # Tame buffering & keep exposure stable (prevents freezes on some systems)
        dev = self.profile.get_device()
        for s in dev.query_sensors():
            if s.supports(rs.option.frames_queue_size):
                s.set_option(rs.option.frames_queue_size, 1)
            if s.supports(rs.option.auto_exposure_priority):
                s.set_option(rs.option.auto_exposure_priority, 0)

    def read(self):
        # Non-blocking read to keep UI responsive
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return False, None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    def get_color_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array(intr.coeffs, dtype=np.float64)  # typically 5 values
        return K, dist, (intr.width, intr.height)

    def close(self):
        self.pipeline.stop()

# ----------------- Main -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# prefer live intrinsics (avoids mismatch when resolution changes)
saved_K = saved_dist = None
calib_path = "../output/realsense_calibration.npz"
if os.path.exists(calib_path):
    with np.load(calib_path) as data:
        saved_K   = data["camera_matrix"].astype(np.float64)
        saved_dist= data["dist_coeffs"].astype(np.float64)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = make_charuco_board(aruco_dict)
params, detector = make_detector_params(aruco_dict)

R_gripper2base, t_gripper2base = [], []
R_target2cam,  t_target2cam  = [], []

os.makedirs(save_dir, exist_ok=True)
sample_path = os.path.join(save_dir, 'handeye_samples.npz')
if os.path.exists(sample_path):
    prev = np.load(sample_path, allow_pickle=True)
    R_gripper2base = list(prev['R_gripper2base'])
    t_gripper2base = list(prev['t_gripper2base'])
    R_target2cam   = list(prev['R_target2cam'])
    t_target2cam   = list(prev['t_target2cam'])
    print(f"[INFO] Loaded {len(R_gripper2base)} previous samples.")
else:
    print("[INFO] No previous samples. Starting fresh.")

pose_count = len(R_gripper2base)
print("[INFO] Move robot to varied poses. Press 's' to save, 'q' to calibrate/quit.")

# Graceful Ctrl+C
stop = {"flag": False}
def _sigint(_, __): stop["flag"] = True
signal.signal(signal.SIGINT, _sigint)

rs_cam = None
try:
    rs_cam = RealSenseSource(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)
    K_live, dist_live, (w_live, h_live) = rs_cam.get_color_intrinsics()
    if saved_K is not None:
        # warn on mismatch
        if np.linalg.norm(saved_K - K_live) > 1e-2:
            print("⚠️  Saved intrinsics differ from live stream. Using LIVE intrinsics.")
    K, dist = K_live, dist_live

    last_valid = {"rvec": None, "tvec": None, "have_pose": False}

    while not stop["flag"]:
        ok, frame = rs_cam.read()
        if not ok:
            # no new frame; keep UI alive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---- Grayscale for ArUco/ChArUco (prevents crashes) ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Interpolate and pose (Board → Camera)
            charuco_corners, charuco_ids = interpolate_charuco(corners, ids, gray, board, K, dist)
            if charuco_corners is not None:
                valid, rvec, tvec = estimate_charuco_pose(charuco_corners, charuco_ids, board, K, dist)
                if valid:
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN_M)
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

        cv2.imshow('Charuco Eye-in-Hand (RealSense Color)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if last_valid["have_pose"]:
                R_board_cam, _ = cv2.Rodrigues(last_valid["rvec"])
                t_board_cam = last_valid["tvec"].reshape(3).astype(np.float64)

                code, pose = arm.get_position()
                if code != 0 or pose is None:
                    print("[ERROR] Failed to read robot pose; sample not saved.")
                else:
                    t_grip_base_m = np.array(pose[:3], dtype=np.float64) / 1000.0
                    R_grip_base   = rpy_to_matrix(pose[3], pose[4], pose[5])

                    R_gripper2base.append(R_grip_base)
                    t_gripper2base.append(t_grip_base_m)
                    R_target2cam.append(R_board_cam)
                    t_target2cam.append(t_board_cam)

                    pose_count += 1
                    print(f"[INFO] Pose #{pose_count} saved.")

                    np.savez(sample_path,
                             R_gripper2base=np.array(R_gripper2base, dtype=object),
                             t_gripper2base=np.array(t_gripper2base, dtype=object),
                             R_target2cam=np.array(R_target2cam, dtype=object),
                             t_target2cam=np.array(t_target2cam, dtype=object))
                    print(f"[INFO] Samples written to {sample_path}")
            else:
                print("[WARN] No valid Charuco pose this frame; hold steady and try again.")
        elif key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    try:
        if rs_cam is not None:
            rs_cam.close()
    except Exception:
        pass
    try:
        arm.disconnect()
    except Exception:
        pass

# ----------------- Solve Eye-in-Hand Calibration -----------------
if pose_count >= 3:
    print("\n[INFO] Running eye-in-hand calibration...")
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
    out_path = os.path.join(save_dir, 'eye_to_hand_calibration.npz')
    np.savez(out_path, T_cam_gripper=T_cam_gripper, T_gripper_cam=T_gripper_cam)
    print(f"\n✅ Saved: {out_path}")
else:
    print("❌ Not enough pose samples! Need at least 3 (8–15 recommended).")
