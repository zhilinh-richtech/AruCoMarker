#!/usr/bin/env python3
import numpy as np
import cv2
import pyrealsense2 as rs
import time
import psutil

# ----------------- Config -----------------
# ---- ArUco marker parameters ----
ARUCO_DICT_ID     = cv2.aruco.DICT_4X4_250  # change if your tag uses a different family
MARKER_LENGTH_M   = 0.108                   # marker side length in meters

# ----------------- Camera intrinsics -----------------
try:
    with np.load("../output/realsense_calibration.npz") as data_cal:
        K    = data_cal["camera_matrix"].astype(np.float64)
        dist = data_cal["dist_coeffs"].astype(np.float64)
    print("Camera calibration loaded successfully")
except:
    print("Warning: Could not load camera calibration, using default values")
    # Default values if calibration not available
    K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

# ----------------- RealSense color stream -----------------
REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS = 1280, 800, 30

# List available cameras
ctx = rs.context()
devices = ctx.query_devices()
print(f"Found {len(devices)} RealSense device(s):")
for i, device in enumerate(devices):
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"  Camera {i}: {name} (Serial: {serial})")

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT, rs.format.bgr8, REALSENSE_FPS)
pipeline.start(cfg)

# ----------------- ArUco detector -----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

print("Starting ArUco marker detection")
print("Press Ctrl+C to stop")
print("-" * 60)

# ----------------- Main loop -----------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            # Process each detected marker
            for i, (corner, marker_id) in enumerate(zip(corners, ids)):
                # Get marker pose in camera frame
                rvecs, tvecs, _objPts = cv2.aruco.estimatePoseSingleMarkers(
                    [corner], MARKER_LENGTH_M, K, dist
                )
                
                rvec = rvecs[0].reshape(3)
                tvec = tvecs[0].reshape(3)
                
                # Convert rotation vector to rotation matrix
                R_cm, _ = cv2.Rodrigues(rvec)
                
                # Get position in camera frame (in meters)
                pos_camera = tvec.reshape(3)  # [x, y, z] in meters
                
                # Get rotation in camera frame (convert to degrees)
                roll, pitch, yaw = cv2.RQDecomp3x3(R_cm)[0]
                
                # Print marker information
                print(f"Marker {marker_id[0]}:")
                print(f"  Position in camera frame: X={pos_camera[0]:.3f}m, Y={pos_camera[1]:.3f}m, Z={pos_camera[2]:.3f}m")
                print(f"  Rotation in camera frame: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
                print(f"  Distance from camera: {np.linalg.norm(pos_camera):.3f}m")
                print("-" * 40)
        else:
            print("No markers detected")

        # Get and print RAM usage every tick
        process = psutil.Process()
        memory_info = process.memory_info()
        ram_usage = memory_info.rss / 1024 / 1024  # Convert bytes to MB
        print(f"RAM Usage: {ram_usage:.1f} MB")
        
        # Slow down to 3 times per second (0.33 second intervals)
        time.sleep(0.33)

except KeyboardInterrupt:
    print("\n\nStopping marker detection...")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    try:
        pipeline.stop()
    except:
        pass
    print("Cleanup completed.")
