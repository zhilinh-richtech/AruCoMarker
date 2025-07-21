import cv2
import os

# --------------- CONFIG -----------------
image_dir = "ArUcoBoardcalib_images_right"
os.makedirs(image_dir, exist_ok=True)
# -------------- END CONFIG --------------

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

# cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(10)

# 设置分辨率和帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

count = 0
print("Press SPACE to capture an image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Calibration Capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("Exit.")
        break
    elif key % 256 == 32:  # SPACE
        filename = f"{image_dir}/image_{count:02d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()
