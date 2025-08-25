import cv2
import os

# Create directory if it doesn't exist
os.makedirs("ArUcoBoardcalib_images", exist_ok=True)

# GStreamer pipeline for Jetson CSI camera
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

count = 0
print("📷 Press SPACE to capture an image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Calibration Capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("🛑 Exit.")
        break
    elif key % 256 == 32:  # SPACE
        filename = f"ArUcoBoardcalib_images/image_{count:02d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()
