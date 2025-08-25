#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import pyrealsense2 as rs

from xarm.wrapper import XArmAPI

XARM_IP = '192.168.10.201'

# ---------- Camera calibration ----------
with np.load("./output/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

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

data = np.load('./arm_camera/eye_to_hand_calibration.npz')
T_cam_base   = data['T_cam_gripper']    #  ^B T_C
# T_marker_tcp = data['T_marker_tcp']  #  ^M T_G   (marker → TCP)

print(camera_matrix)
print("T_cam_base:\n", T_cam_base)

# ----------------- Connect xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ---------- ArUco setup ----------
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters  = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_len  = 0.07  # marker side length in metres
TRACK_ID    = 0     # we care only about ID 0

# squares_x = 7           # Number of chessboard squares in X direction
# squares_y = 9              # Number of chessboard squares in Y direction
# square_length = 0.035     # Square size (meters)
# marker_length = 0.027      # Marker size inside each square (meters)

# charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
# board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, charuco_dict)

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
    raise RuntimeError("❌ Failed to open CSI camera via GStreamer")

def rpy_to_matrix(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
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

# ---------- Calibration parameters ----------
init_duration = 5          # seconds to watch the stationary marker
padding       = 0      # x mm extra tolerance per axis
samples       = []          # accumulated pose samples (tvecs) for ID 0
bounds        = None        # dict with 'min' and 'max' arrays after calibration
start_time    = time.time()
calibrated    = False

# ---------- Main loop ----------
while True:
    # ret, frame = cap.read()
    # if not ret:
    #     print("❌ Frame grab failed")

    #     break

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    corners, ids, _ = detector.detectMarkers(frame)

    # if ids is not None and TRACK_ID in ids:
    if len(corners) > 0:
        # find the row where the marker of interest appears
        idx = int(np.where(ids == TRACK_ID)[0][0])

        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


        marker_points = np.array([[-marker_len / 2, marker_len / 2, 0],
                              [marker_len / 2, marker_len / 2, 0],
                              [marker_len / 2, -marker_len / 2, 0],
                              [-marker_len / 2, -marker_len / 2, 0]], dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, camera_matrix, dist_coeffs)
        rvec = rvec[0]
        tvec = tvec[0][0]


        # --------- Calibration phase ---------
        if not calibrated:
            # still collecting
            samples.append(tvec)
            calibrated = True
            # Visual feedback
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec, tvec, 0.03)
            cv2.putText(frame, "Calibrating...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            

            if time.time() - start_time >= init_duration:
                samples_np = np.array(samples)
                bounds = {
                    "min": samples_np.min(axis=0) - padding,
                    "max": samples_np.max(axis=0) + padding
                }
                avg = (bounds["min"] + bounds["max"]) / 2
                print("✅ Calibration finished; avg: ", avg)
                r, t = rvec, tvec

                r, jacobian = cv2.Rodrigues(r)
                temp = t[:, None]
                t = temp
                print(r, t)


        # --------- Post‑calibration monitoring ---------
                origin = [[0], [0], [0], [1]]
                
                extra_row_homgen = np.array([[0, 0, 0, 1]])

                objectToCamera = np.concatenate((r, t), axis = 1)
                objectToCamera = np.concatenate((objectToCamera, extra_row_homgen), axis = 0)
                # objectToCamera = np.array([rvec, tvec,
                #                   [0, 0, 0, 1]])

                cameraToBase = np.array([[0, 0, -1, 0.56],
                                [1, 0, 0, -0.725],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])
                
                posy = np.array([[0, 0, 1],
                                 [0, 1, 0],
                                 [-1, 0, 0]])
                posz = np.array([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]])
                posx = np.array([[1, 0, 0],
                                 [0, math.cos(math.pi/2), -1 * math.sin(math.pi/2)],
                                 [0, math.sin(math.pi/2), math.cos(math.pi/2)]])
                rot1 = posz
                rot2 = posx
                rot3 = np.dot(-1*posx, posy)
                rot4 = -1 * posy
                rot5 = np.dot(-1 * posz, posy)

                eeRot = rot1 * rot2 * rot3 * rot4 * rot5
  
                eeRot = [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]

                eeTrans = [[0.15], [0], [-0.2]]

                code, pose = arm.get_position()
                t_gripper_base = np.array(pose[:3]) / 1000.0  # mm → m
                R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
                baseToEE = to_homogeneous(R_gripper_base, t_gripper_base)

                R_g2b = R_gripper_base.T
                t_g2b = -R_g2b @ t_gripper_base
                eeToBase = to_homogeneous(R_g2b, t_g2b)



                # ----------------  NEW POSE ESTIMATION  ----------------
                T_marker_cam   = objectToCamera                      #  ^C T_M
                T_marker_base  = T_cam_base @ T_marker_cam           #  ^B T_M
                # T_gripper_base_est = T_marker_base @ T_marker_tcp    #  ^B T_G   (no inverse!)


                # -----  Build actual pose from robot feedback  -----
                code, pose = arm.get_position()
                t_gripper_base = np.array(pose[:3]) / 1000.0       # mm → m
                R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
                T_gripper_base_act = to_homogeneous(R_gripper_base, t_gripper_base)

                #  --- Diagnostics ---
                trans_err = T_gripper_base_act[:3, 3] - T_marker_base[:3, 3]
                print("Estimated t:", T_marker_base[:3, 3])
                print("Actual    t:", T_gripper_base_act[:3, 3])
                print("Δt (m):", trans_err)

                rot_err_rpy = matrix_to_rpy(R_gripper_base) - matrix_to_rpy(T_marker_base[:3, :3])
                print("Estimated RPY:", matrix_to_rpy(T_marker_base[:3, :3]))
                print("Actual    RPY:", matrix_to_rpy(R_gripper_base))
                print("ΔRPY (deg):", rot_err_rpy)
        else:
            objectToCamera = np.eye(4)
            r_mat, _ = cv2.Rodrigues(rvec)
            objectToCamera[:3, :3] = r_mat
            objectToCamera[:3, 3]  = tvec.reshape(3)

            # Estimate gripper pose
            T_marker_base      = T_cam_base @ objectToCamera    # camera→base · marker→camera
            # T_gripper_base_est = T_marker_base @ T_marker_tcp   # marker→base · marker→TCP

           # Actual gripper pose
            code, pose = arm.get_position()
            t_gripper_base = np.array(pose[:3]) / 1000.0
            R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
            T_gripper_base_act = to_homogeneous(R_gripper_base, t_gripper_base)

            # Diagnostics (updates every loop!)
            trans_err = T_gripper_base_act[:3, 3] - T_marker_base[:3, 3]
            rot_err_rpy = matrix_to_rpy(R_gripper_base) - matrix_to_rpy(T_marker_base[:3, :3])

            print(f"[Δt] Est {T_marker_base[:3, 3]} | Act {T_gripper_base_act[:3, 3]} | Err {trans_err}")
            print(f"[ΔRPY] Est {matrix_to_rpy(T_marker_base[:3, :3])} | Act {matrix_to_rpy(R_gripper_base)} | Err {rot_err_rpy}")
            print("─" * 60)


    else:
        # Marker 0 not visible
        cv2.putText(frame, "Marker 0 not detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        # Get image dimensions
    height, width, _ = frame.shape

    # Draw horizontal center line
    cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

    # Draw vertical center line
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

    cv2.imshow("ArUco Pose – ID 0 monitor", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()