#!/usr/bin/env python3
import numpy as np
import cv2
import pyrealsense2 as rs
from utils import (
    to_homogeneous,
    invert_se3,
    matrix_to_rpy,
    rpy_to_matrix,
    rot_angle_deg,
)
from xarm.wrapper import XArmAPI

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'

# ---- ChArUco board parameters (must match your printed board) ----
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M      = 0.02474
MARKER_LEN_RATIO  = 0.78
MARKER_LEN_M      = SQUARE_LEN_M * MARKER_LEN_RATIO
AXIS_LEN_M        = 0.08

# ----------------- Load calibration -----------------
# Your file stores Base <- Camera, and you print it as "base to cam"
data = np.load('../output/eye_to_hand_calibration.npz')
mark = np.load("../output/markercalibration.npz")
T_cam_base = mark['last_mark'].astype(np.float64)  # Base ← Camera
print(T_cam_base)
#T_cam_base = data['T_cam_base']  # Base ← Camera
# T_cam_base = np.array([[-0.99504,  -0.005857 ,-0.099307,  0.446634],
#  [ 0.099255, -0.125581 ,-0.987106 , 1.065138],
#  [-0.00669 , -0.992066,  0.125539 , 0.146064],
#     [ 0.000000,  0.000000,  0.000000,  1.000000]])

print("T_cam_base:\n", T_cam_base)

# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ----------------- Load camera intrinsics -----------------
with np.load("../output/realsense_calibration.npz") as data:
    K    = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)

# ----------------- RealSense color stream -----------------
REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS = 1280, 800, 30
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT, rs.format.bgr8, REALSENSE_FPS)
pipeline.start(cfg)

# ----------------- Charuco detector -----------------
def make_board_and_detector(dict_id=cv2.aruco.DICT_4X4_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.detectInvertedMarker = True
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0
    det = cv2.aruco.ArucoDetector(aruco_dict, params)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M, MARKER_LEN_M, aruco_dict
    )
    return board, det

board, detector = make_board_and_detector(cv2.aruco.DICT_4X4_250)

# ----------------- Helpers -----------------
# Using shared utilities from utils.py

# ----------------- Main loop -----------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(corners) >= 4:
            # Interpolate Charuco corners
            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, frame, board
            )
            if retval is not None and retval >= 4:
                ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    ch_corners, ch_ids, board, K, dist, None, None
                )
                if ok:
                    # Camera ← Board (OpenCV rvec/tvec are object->camera)
                    R_cb, _ = cv2.Rodrigues(rvec)
                    t_cb = tvec.reshape(3)
                    charucoToCam = to_homogeneous(R_cb, t_cb)  # Camera ← Board

                    # Draw axes
                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, AXIS_LEN_M)

                    # Robot pose
                    code, pose = arm.get_position()  # degrees
                    print(pose)
                    if code == 0:
                        t_gripper_base = np.array(pose[:3], dtype=np.float64) / 1000.0  # mm → m
                        R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
                        T_base_gripper = to_homogeneous(R_gripper_base, t_gripper_base)  # Base ← Gripper
                        # T_gripper_base = invert_se3(T_base_gripper)                      # Gripper ← Base
                        # ---------- Error analysis (Base frame; assumes Board == TCP) ----------
                        # Estimated Base ← EE from Charuco:
                        T_B_from_Board = T_cam_base @ charucoToCam  # Base ← Board ≡ Base ← EE if Board==TCP

                        # Actual Base ← EE from robot:
                        T_B_from_G = T_base_gripper

                        # Residual (should be ~Identity):
                        delta = T_B_from_Board @ invert_se3(T_B_from_G)
                        t_err_m   = delta[:3, 3]
                        R_err     = delta[:3, :3]
                        t_err_mm  = t_err_m * 1000.0
                        trans_mm  = float(np.linalg.norm(t_err_mm))
                        rot_deg   = float(rot_angle_deg(R_err))
                        rpy_err   = matrix_to_rpy(R_err)
                        

                        print("\n=== Error Analysis (Base frame; Board==TCP) ===")
                        print(f"Estimated Base ← EE (from Charuco):\n{T_B_from_Board}")
                        print(f"Δt (mm): x={t_err_mm[0]:.2f}, y={t_err_mm[1]:.2f}, z={t_err_mm[2]:.2f}  |  ||Δt|| = {trans_mm:.2f} mm")
                        print(f"ΔR angle: {rot_deg:.2f}°   ΔRPY (deg): [{rpy_err[0]:.2f}, {rpy_err[1]:.2f}, {rpy_err[2]:.2f}]")

                        

                        # Your flow:
                        # "calculated gripper to cam" = Gripper ← Camera
                        # gripperToCam = charucoToCam @ T_cam_base
                        gripperToCam = T_cam_base @ charucoToCam
                        # Compare in the SAME frame:
                        # Camera ← Gripper (robot) vs Camera ← Board (ChArUco, Board==TCP)

                        # delta = gripperToCam @ T_base_gripper  # ~I if Board==TCP
                        # d_t = delta[:3,3]
                        # d_R = delta[:3,:3]
                        # d_rpy = matrix_to_rpy(d_R)
                        # d_ang = rot_angle_deg(d_R)

                        # ----- Prints (matching your labels) -----
                        # print("base to cam")
                        # print(T_cam_base)
                        print("gripper to base")
                        print(T_base_gripper)
                        # # print("gripper to base")
                        # # print(T_gripper_base)
                        # print("direct board to cam:")
                        # print(charucoToCam)
                        print("calculated gripper to cam")
                        print(gripperToCam)
                        # print(pose)
                        actual_rpy = matrix_to_rpy(T_base_gripper)
                        actual_xyz = T_base_gripper[:3, 3]*1000
                        estimated_rpy = matrix_to_rpy(gripperToCam)
                        estimated_xyz = gripperToCam[:3, 3]*1000
                        # # Extra: numeric agreement, same-frame check
                        # print("delta (should be ~Identity):")
                        # print(delta)
                        # print(f"Δt (mm): [{d_t[0]*1000:.1f}, {d_t[1]*1000:.1f}, {d_t[2]*1000:.1f}]  |  Δangle: {d_ang:.2f} deg  |  ΔRPY: {d_rpy}")
                        print(actual_xyz)
                        print(estimated_xyz)
                        print(actual_xyz - estimated_xyz)
                        print(actual_rpy - estimated_rpy)

        # HUD
        cv2.putText(frame, "ChArUco Eye-to-Hand (Board==TCP, RealSense)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("ChArUco (RealSense)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    try: pipeline.stop()
    except: pass
    try: arm.disconnect()
    except: pass
