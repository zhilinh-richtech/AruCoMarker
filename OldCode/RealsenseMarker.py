#!/usr/bin/env python3
import cv2
import numpy as np
import time
import pyrealsense2 as rs

# =========================
# CONFIG (must match your intrinsics file resolution)
# =========================
REALSENSE_WIDTH  = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS    = 30

# ---------- Camera calibration ----------
# Make sure this K/dist were calibrated at REALSENSE_WIDTH x REALSENSE_HEIGHT
with np.load("../output/realsense_calibration.npz") as data:
    camera_matrix = data["camera_matrix"].astype(np.float64)
    dist_coeffs   = data["dist_coeffs"].astype(np.float64)

# ---------- ArUco setup ----------
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters  = cv2.aruco.DetectorParameters()
# (Optional) slightly more robust defaults
# parameters.adaptiveThreshWinSizeMin = 5
# parameters.adaptiveThreshWinSizeMax = 45
# parameters.adaptiveThreshWinSizeStep = 5
# parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# parameters.cornerRefinementWinSize = 7
# parameters.detectInvertedMarker = True
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_len  = 0.125  # marker side length in METERS (black square only)
TRACK_ID    = 0     # we care only about ID 0

print("Camera matrix:\n", camera_matrix)
print("Distortion:\n", dist_coeffs.ravel())

# ---------- RealSense color stream ----------
class RealSenseColor:
    def __init__(self, width=640, height=480, fps=30, lock_ae=False):
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.cfg)

        if lock_ae:
            # Optional: lock exposure to reduce flicker/jitter
            try:
                dev = self.profile.get_device()
                for s in dev.query_sensors():
                    name = s.get_info(rs.camera_info.name).lower()
                    if "color" in name:
                        s.set_option(rs.option.enable_auto_exposure, 0)
                        s.set_option(rs.option.exposure, 8000)  # microseconds; tune for lighting
                        s.set_option(rs.option.gain, 32)       # tune as needed
                        break
            except Exception as e:
                print("Warning: could not lock exposure:", e)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return False, None
        img = np.asanyarray(color.get_data())
        return True, img

    def release(self):
        self.pipeline.stop()

# ---------- Calibration parameters ----------
init_duration = 15        # seconds to watch the stationary marker
padding       = 0.0       # meters of extra tolerance per axis (use 0.005 for 5 mm, etc.)
samples       = []        # accumulated tvec samples (in camera frame, meters)
bounds        = None      # dict with 'min' and 'max' arrays after calibration
start_time    = time.time()

# ---------- Main ----------
def main():
    cam = RealSenseColor(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS, lock_ae=False)
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("❌ Frame grab failed")
                break

            # Detect markers
            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and TRACK_ID in ids:
                # find the row where the marker of interest appears
                idx = int(np.where(ids == TRACK_ID)[0][0])

                # Pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[idx:idx+1], marker_len, camera_matrix, dist_coeffs
                )
                rvec = rvecs[0]       # shape (1,3)
                tvec = tvecs[0][0]    # shape (3,)

                # --------- Calibration phase ---------
                if bounds is None:
                    samples.append(tvec.copy())

                    # Visual feedback
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                    cv2.putText(frame, "Calibrating...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

                    if time.time() - start_time >= init_duration:
                        samples_np = np.array(samples)  # (N,3)
                        bounds_local = {
                            "min": samples_np.min(axis=0) - padding,
                            "max": samples_np.max(axis=0) + padding
                        }
                        print("✅ Calibration finished")
                        print("  Min xyz (m):", bounds_local["min"])
                        print("  Max xyz (m):", bounds_local["max"])
                        # Assign to outer scope
                        globals()["bounds"] = bounds_local

                # --------- Post-calibration monitoring ---------
                else:
                    moved = np.any((tvec < bounds["min"]) | (tvec > bounds["max"]))
                    status = "MOVED!" if moved else "stationary"
                    print(f"Marker 0 → {status}  tvec(m)={tvec}  rvec={rvec.flatten()}")

                    # Draw outline/axes
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                    cv2.putText(frame, status,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255) if moved else (0, 255, 0), 2)

                    # Center lines
                    h, w = frame.shape[:2]
                    cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 2)   # horizontal
                    cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 2)   # vertical

            else:
                # Marker 0 not visible
                cv2.putText(frame, "Marker 0 not detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            cv2.imshow("ArUco Pose – ID 0 monitor (RealSense)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
