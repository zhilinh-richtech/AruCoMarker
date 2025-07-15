#!/usr/bin/env python3
import cv2
import numpy as np
import time

# ---------- Camera calibration ----------
with np.load("output/charuco_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]

# ---------- ArUco setup ----------
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters  = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_len  = 0.05  # marker side length in metres
TRACK_ID    = 0     # we care only about ID 0

print(camera_matrix)

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

# ---------- Calibration parameters ----------
init_duration = 15          # seconds to watch the stationary marker
padding       = 0      # x mm extra tolerance per axis
samples       = []          # accumulated pose samples (tvecs) for ID 0
bounds        = None        # dict with 'min' and 'max' arrays after calibration
start_time    = time.time()

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")

        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and TRACK_ID in ids:
        # find the row where the marker of interest appears
        idx = int(np.where(ids == TRACK_ID)[0][0])
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx:idx+1], marker_len, camera_matrix, dist_coeffs
        )
        tvec = tvec[0][0]   # shape (3,)

        # --------- Calibration phase ---------
        if bounds is None:                 # still collecting
            samples.append(tvec)

            # Visual feedback
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec[0], tvec, 0.03)
            cv2.putText(frame, "Calibrating...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            if time.time() - start_time >= init_duration:
                samples_np = np.array(samples)
                bounds = {
                    "min": samples_np.min(axis=0) - padding,
                    "max": samples_np.max(axis=0) + padding
                }
                print("✅ Calibration finished")
                print("  Min xyz:", bounds["min"])
                print("  Max xyz:", bounds["max"])

        # --------- Post‑calibration monitoring ---------
        else:
            moved = np.any((tvec < bounds["min"]) | (tvec > bounds["max"]))
            status = "MOVED!" if moved else "stationary"
            print(f"Marker 0 → {status}  pos={tvec} {rvec}")

            # Draw outline/axes
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec[0], tvec, 0.03)
            cv2.putText(frame, status,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255) if moved else (0, 255, 0), 2)
                    
            height, width, _ = frame.shape
            # Draw horizontal center line
            cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

            # Draw vertical center line
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

    else:
        # Marker 0 not visible
        cv2.putText(frame, "Marker 0 not detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("ArUco Pose – ID 0 monitor", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
