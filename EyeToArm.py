#!/usr/bin/env python3
import numpy as np
import cv2
from xarm.wrapper import XArmAPI

# # ----------------- Config -----------------
# XARM_IP = '192.168.10.201'
# MARKER_LENGTH = 0.05  # in meters

# # ----------------- Load calibration -----------------
# data = np.load('./output/eye_to_hand_calibration.npz')
# T_gripper_cam = data['T_gripper_cam']
# print("T_gripper_cam:\n", T_gripper_cam)

# # ----------------- Connect xArm -----------------
# arm = XArmAPI(XARM_IP)
# arm.motion_enable(True)

# # ----------------- Load camera calibration -----------------
# with np.load("./output/charuco_calibration.npz") as data:
#     camera_matrix = data["camera_matrix"]
#     dist_coeffs = data["dist_coeffs"]

# # ----------------- Setup camera -----------------
# def gstreamer_pipeline(sensor_id=0,
#                        capture_width=1280, capture_height=720,
#                        display_width=1280, display_height=720,
#                        framerate=30, flip_method=2):
#     return (
#         f"nvarguscamerasrc sensor-id={sensor_id} ! "
#         f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
#         f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
#         f"nvvidconv flip-method={flip_method} ! "
#         f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
#         f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
#     )

# cap = cv2.VideoCapture(gstreamer_pipeline(
#     sensor_id=1,
#     capture_width=1280,
#     capture_height=720,
#     display_width=1280,
#     display_height=720,
#     framerate=30,
#     flip_method=2
# ), cv2.CAP_GSTREAMER)

# if not cap.isOpened():
#     print("Failed to open camera")
#     exit(1)

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# # ----------------- Helper functions -----------------
# def to_homogeneous(R, t):
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = t.reshape(3)
#     return T

# def matrix_to_rpy(R):
#     sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
#     singular = sy < 1e-6
#     if not singular:
#         roll = np.arctan2(R[2, 1], R[2, 2])
#         pitch = np.arctan2(-R[2, 0], sy)
#         yaw = np.arctan2(R[1, 0], R[0, 0])
#     else:
#         roll = np.arctan2(-R[1, 2], R[1, 1])
#         pitch = np.arctan2(-R[2, 0], sy)
#         yaw = 0
#     return np.degrees([roll, pitch, yaw])

# def rpy_to_matrix(roll, pitch, yaw):
#     roll = np.radians(roll)
#     pitch = np.radians(pitch)
#     yaw = np.radians(yaw)
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(roll), -np.sin(roll)],
#         [0, np.sin(roll), np.cos(roll)]
#     ])
#     Ry = np.array([
#         [np.cos(pitch), 0, np.sin(pitch)],
#         [0, 1, 0],
#         [-np.sin(pitch), 0, np.cos(pitch)]
#     ])
#     Rz = np.array([
#         [np.cos(yaw), -np.sin(yaw), 0],
#         [np.sin(yaw), np.cos(yaw), 0],
#         [0, 0, 1]
#     ])
#     return Rz @ Ry @ Rx

# # ----------------- Main loop -----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     corners, ids, _ = detector.detectMarkers(frame)
#     if ids is not None:
#         rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)

#                 # Marker pose in camera frame
#         R_marker_cam, _ = cv2.Rodrigues(rvec[0])
#         t_marker_cam = tvec[0].flatten()
#         T_marker_cam = to_homogeneous(R_marker_cam, t_marker_cam)

#         # Check when marker on TCP
#         T_check = T_marker_cam @ T_gripper_cam

#         delta_t = T_check[:3, 3]
#         delta_R = T_check[:3, :3]
#         rpy_error = matrix_to_rpy(delta_R)

#         print("=== Marker on TCP Check (should be near identity) ===")
#         print(f"Translation error (mm): x={delta_t[0]*1000:.1f}, y={delta_t[1]*1000:.1f}, z={delta_t[2]*1000:.1f}")
#         print(f"Rotation error (deg): roll={rpy_error[0]:.1f}, pitch={rpy_error[1]:.1f}, yaw={rpy_error[2]:.1f}\n")


#         # Draw marker axes
#         cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.03)

#     cv2.imshow("Marker Detection", frame)
#     key = cv2.waitKey(10) & 0xFF
#     if key == ord('q'):
#         break

# # ----------------- Cleanup -----------------
# cap.release()
# cv2.destroyAllWindows()
# arm.disconnect()
# #!/usr/bin/env python3
# import numpy as np
# import cv2
# from xarm.wrapper import XArmAPI

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'
MARKER_LENGTH = 0.05  # in meters

# ----------------- Load calibration -----------------
data = np.load('./output/eye_to_hand_calibration.npz')
T_cam_base = data['T_cam_gripper']
print("T_cam_base:\n", T_cam_base)

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

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ----------------- Helper functions -----------------
def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

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

# ----------------- Main loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        extra_row_homgen = np.array([[0, 0, 0, 1]])

        # Marker pose in camera frame
        R_marker_cam, _ = cv2.Rodrigues(rvec[0])
        t_marker_cam = tvec[0].flatten()
        t_marker_cam = t_marker_cam[:, None]
        # print(R_marker_cam, t_marker_cam)
        # T_marker_cam = to_homogeneous(R_marker_cam, t_marker_cam)
        markerToCam = np.concatenate((R_marker_cam, t_marker_cam), axis = 1)
        markerToCam = np.concatenate((markerToCam, extra_row_homgen), axis = 0)

        # Get gripper pose from robot
        code, pose = arm.get_position()
        t_gripper_base = np.array(pose[:3]) / 1000.0  # mm → m
        R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
        T_base_gripper = to_homogeneous(R_gripper_base, t_gripper_base)

        R_g2b = R_gripper_base.T
        t_g2b = -R_g2b @ t_gripper_base
        T_gripper_base = to_homogeneous(R_g2b, t_g2b)
        
        # Marker pose in base frame estimated via camera
        gripperToCam = T_gripper_base @ T_cam_base
        # np.matmul(H_cam2gripper, T_marker_cam)

        # # Extract
        # t_marker_est = T_marker_base_est[:3, 3]
        # R_marker_est = T_marker_base_est[:3, :3]
        # rpy_marker_est = matrix_to_rpy(R_marker_est)

        # # Extract robot TCP pose
        # rpy_gripper = np.array(pose[3:])

        # print("=== Marker (Camera-estimated) in Base Frame ===")
        # print(f"x={t_marker_est[0]*1000:.1f}, y={t_marker_est[1]*1000:.1f}, z={t_marker_est[2]*1000:.1f}, "
        #       f"roll={rpy_marker_est[0]:.1f}, pitch={rpy_marker_est[1]:.1f}, yaw={rpy_marker_est[2]:.1f}")

        # print("=== Robot TCP in Base Frame ===")
        # print(f"x={pose[0]:.1f}, y={pose[1]:.1f}, z={pose[2]:.1f}, "
        #       f"roll={rpy_gripper[0]:.1f}, pitch={rpy_gripper[1]:.1f}, yaw={rpy_gripper[2]:.1f}\n")

        print("base to cam")
        print(T_cam_base)
        print("base to gripper")
        print(T_base_gripper)
        print("gripper to base")
        print(T_gripper_base)
        print("direct marker to cam: ")
        print(markerToCam)
        print("calculated gripper to cam")
        print(gripperToCam)
        
        # Draw marker axes
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.03)

    cv2.imshow("Marker Detection", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

# ----------------- Cleanup -----------------
cap.release()
cv2.destroyAllWindows()
arm.disconnect()
