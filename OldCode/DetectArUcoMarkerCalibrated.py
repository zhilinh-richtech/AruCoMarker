import cv2
import numpy as np

# Load camera calibration data
with np.load('output/calibration_data.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Known size of your ArUco marker in meters (change accordingly)
marker_length = 0.05  # 5 cm for example

# GStreamer pipeline for Jetson camera, adjust if using another camera
def gstreamer_pipeline(
    sensor_id=1,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2,
):
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
    print("❌ Failed to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Detect markers
    markerCorners, markerIds, rejected = detector.detectMarkers(frame)

    if markerIds is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, marker_length, camera_matrix, dist_coeffs)

        for i in range(len(markerIds)):
            # Draw axis for each mark            er
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            # Print the marker ID and its position
            pos = tvecs[i][0]
            print(f"Marker ID: {markerIds[i][0]}, Position (x, y, z) in meters: {pos}")

    cv2.imshow("ArUco Pose Estimation", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()