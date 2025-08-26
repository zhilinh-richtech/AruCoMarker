#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs

# =========================
# CONFIG (must match your intrinsics file resolution)
# =========================
REALSENSE_WIDTH  = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS    = 30

TARGET_CAPTURES  = 20           # number of poses to capture with SPACE
TRACK_ID         = 0            # capture only this ArUco ID
MARKER_LEN_M     = 0.125        # marker side length in METERS (black square only)

# ---------- Camera calibration ----------
with np.load("../output/realsense_calibration.npz") as data:
    K    = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64)

# ---------- ArUco setup ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ---------- RealSense color stream ----------
class RealSenseColor:
    def __init__(self, width=640, height=480, fps=30, lock_ae=False):
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.cfg)
        if lock_ae:
            try:
                dev = self.profile.get_device()
                for s in dev.query_sensors():
                    if "color" in s.get_info(rs.camera_info.name).lower():
                        s.set_option(rs.option.enable_auto_exposure, 0)
                        s.set_option(rs.option.exposure, 8000)  # µs; tune as needed
                        s.set_option(rs.option.gain, 32)        # tune as needed
                        break
            except Exception as e:
                print("Warning: could not lock exposure:", e)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def release(self):
        self.pipeline.stop()

# ---------- Helpers ----------
def rt_to_T(rvec, tvec):
    """Rodrigues rvec (3,) + tvec (3,) -> 4x4 homogeneous (Camera <- Marker)."""
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = tvec.reshape(3)
    return T

def invert_T(T):
    """Invert 4x4 rigid transform."""
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti

# ---------- Main ----------
def main():
    np.set_printoptions(precision=6, suppress=True)
    cam = RealSenseColor(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS, lock_ae=False)

    captured = 0
    last_T_marker_from_cam = None  # will hold the final transform (Marker ← Camera)

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("❌ Frame grab failed")
                break

            corners, ids, _ = detector.detectMarkers(frame)

            has_target = ids is not None and (TRACK_ID in ids)
            if has_target:
                idx = int(np.where(ids == TRACK_ID)[0][0])

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[idx:idx+1], MARKER_LEN_M, K, dist
                )
                rvec = rvecs[0].reshape(3)
                tvec = tvecs[0][0].reshape(3)

                # Draw
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[TRACK_ID]]))
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05)

                cv2.putText(frame, f"ID {TRACK_ID}  Captures: {captured}/{TARGET_CAPTURES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Marker {TRACK_ID} not detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # crosshairs
            h, w = frame.shape[:2]
            cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
            cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)

            cv2.imshow("ArUco Pose – capture cam→marker transform", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key == ord(' ') and has_target:
                # Build transforms
                T_cam_from_marker = rt_to_T(rvec, tvec)      # Camera ← Marker
                T_marker_from_cam = invert_T(T_cam_from_marker)  # Marker ← Camera (cam→marker)

                captured += 1
                last_T_marker_from_cam = T_marker_from_cam.copy()

                print(f"\nCapture #{captured} complete.")

                if captured >= TARGET_CAPTURES:
                    print("\n=== Final transformation (Marker ← Camera) ===")
                    print(last_T_marker_from_cam)
                    # Save intrinsics to .npz
                    np.savez("../output/markercalibration.npz",
                    last_mark = last_T_marker_from_cam,)
                    break  # done after printing

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
