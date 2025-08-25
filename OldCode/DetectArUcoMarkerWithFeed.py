import cv2 as cv

def gstreamer_pipeline(
    sensor_id=1,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# Create ArUco detector
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
aruco_params = cv.aruco.DetectorParameters()
aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

# Open camera using GStreamer
cap = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open CSI camera via GStreamer.")
    exit()

print("✅ Camera stream opened. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)

    # Draw markers
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

    cv.imshow("📷 ArUco Marker Detection", frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv.destroyAllWindows()