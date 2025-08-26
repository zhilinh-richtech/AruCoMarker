#!/usr/bin/env python3
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time
import json
from datetime import datetime
import os

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'
ERROR_THRESHOLD_MM = 1.0  # Calibration error threshold in mm

# ---- ArUco marker parameters ----
ARUCO_DICT_ID     = cv2.aruco.DICT_4X4_250
MARKER_LENGTH_M   = 0.108  # marker side length in meters

# Data storage for calibration sets
calibration_sets = []
current_set_errors = []

# ----------------- Eye-to-hand (Base ← Camera) -----------------
try:
    data = np.load('../output/markercalibration.npz')
    T_cam_base = data['last_mark'].astype(np.float64)  # Base ← Camera
    print("Loaded existing camera calibration")
except:
    print("Warning: Could not load camera calibration, using identity matrix")
    T_cam_base = np.eye(4, dtype=np.float64)

# ----------------- Connect xArm -----------------
try:
    arm = XArmAPI(XARM_IP)
    arm.motion_enable(True)
    print(f"Successfully connected to xArm at {XARM_IP}")
    XARM_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not connect to xArm at {XARM_IP}: {e}")
    print("Continuing without robot connection - only camera data will be collected")
    XARM_AVAILABLE = False
    arm = None

# ----------------- Camera intrinsics -----------------
try:
    with np.load("../output/realsense_calibration.npz") as data_cal:
        K    = data_cal["camera_matrix"].astype(np.float64)
        dist = data_cal["dist_coeffs"].astype(np.float64)
    print("Camera calibration loaded successfully")
except:
    print("Warning: Could not load camera calibration, using default values")
    K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

# ----------------- RealSense color stream -----------------
REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS = 1280, 800, 30
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, REALSENSE_WIDTH, REALSENSE_HEIGHT, rs.format.bgr8, REALSENSE_FPS)
pipeline.start(cfg)

# ----------------- Helpers -----------------
def to_homogeneous(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.reshape(3)
    return T

def matrix_to_rpy(R):
    # ZYX (yaw-pitch-roll) -> returns (roll, pitch, yaw) in degrees
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])

def rpy_to_matrix(roll, pitch, yaw):
    roll  = np.radians(roll)
    pitch = np.radians(pitch)
    yaw   = np.radians(yaw)
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def calculate_xyz_error(estimated_pose, actual_pose):
    """Calculate xyz error between estimated and actual poses"""
    # Extract translation components
    est_xyz = estimated_pose[:3, 3] * 1000  # Convert to mm
    act_xyz = actual_pose[:3, 3] * 1000     # Convert to mm
    
    # Calculate error vector
    error_xyz = est_xyz - act_xyz
    
    # Calculate absolute sum as specified
    abs_sum = np.abs(np.sum(error_xyz))
    
    return error_xyz, abs_sum

def take_calibration_shot():
    """Take a single calibration shot and return pose data"""
    print("Taking calibration shot...")
    
    # Wait for frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    
    frame = np.asanyarray(color_frame.get_data())
    
    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is None or len(ids) == 0:
        print("No markers detected, retrying...")
        return None
    
    # Use first detected marker
    corner = corners[0]
    marker_id = ids[0]
    
    # Estimate pose
    rvecs, tvecs, _objPts = cv2.aruco.estimatePoseSingleMarkers(
        [corner], MARKER_LENGTH_M, K, dist
    )
    
    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3)
    
    # Convert to homogeneous transformation
    R_cm, _ = cv2.Rodrigues(rvec)
    T_C_from_M = to_homogeneous(R_cm, tvec)  # Camera ← Marker
    T_B_from_M = T_cam_base @ T_C_from_M     # Base ← Marker
    
    # Get robot pose if available
    if XARM_AVAILABLE:
        code, pose = arm.get_position()
        if code == 0:
            t_gripper_base = np.array(pose[:3], dtype=np.float64) / 1000.0  # mm → m
            R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])
            T_base_gripper = to_homogeneous(R_gripper_base, t_gripper_base)
            
            # Calculate error
            error_xyz, abs_sum = calculate_xyz_error(T_B_from_M, T_base_gripper)
            
            return {
                'marker_id': int(marker_id[0]),
                'estimated_pose': T_B_from_M,
                'actual_pose': T_base_gripper,
                'error_xyz': error_xyz,
                'abs_sum_mm': abs_sum
            }
    
    # If no robot connection, return estimated pose only
    return {
        'marker_id': int(marker_id[0]),
        'estimated_pose': T_B_from_M,
        'actual_pose': None,
        'error_xyz': None,
        'abs_sum_mm': None
    }

def take_calibration_set(set_number):
    """Take 3 calibration shots and return the set data"""
    print(f"\n=== Taking Calibration Set {set_number} ===")
    print("Taking 3 calibration shots...")
    
    set_data = []
    for i in range(3):
        print(f"Shot {i+1}/3...")
        shot_data = take_calibration_shot()
        
        if shot_data is None:
            print(f"Failed to take shot {i+1}, retrying...")
            i -= 1  # Retry this shot
            time.sleep(1)
            continue
        
        set_data.append(shot_data)
        time.sleep(0.5)  # Small delay between shots
    
    # Calculate average error for this set
    if all(shot['abs_sum_mm'] is not None for shot in set_data):
        avg_abs_sum = np.mean([shot['abs_sum_mm'] for shot in set_data])
        print(f"Set {set_number} average absolute sum: {avg_abs_sum:.3f} mm")
    else:
        avg_abs_sum = None
        print(f"Set {set_number} completed (no robot connection for error calculation)")
    
    return {
        'set_number': set_number,
        'shots': set_data,
        'avg_abs_sum_mm': avg_abs_sum,
        'timestamp': datetime.now().isoformat()
    }

def compare_calibration_sets(set1, set2):
    """Compare two calibration sets and show improvement/worsening"""
    if set1['avg_abs_sum_mm'] is None or set2['avg_abs_sum_mm'] is None:
        print("Cannot compare sets - no error data available")
        return
    
    error_change = set2['avg_abs_sum_mm'] - set1['avg_abs_sum_mm']
    
    print(f"\n=== Calibration Comparison ===")
    print(f"Set {set1['set_number']} average: {set1['avg_abs_sum_mm']:.3f} mm")
    print(f"Set {set2['set_number']} average: {set2['avg_abs_sum_mm']:.3f} mm")
    print(f"Error change: {error_change:+.3f} mm")
    
    if error_change < 0:
        print("Status: IMPROVED ✓")
    elif error_change > 0:
        print("Status: WORSENED ✗")
    else:
        print("Status: NO CHANGE")

def save_calibration_results():
    """Save all calibration results to file"""
    filename = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'error_threshold_mm': ERROR_THRESHOLD_MM,
                'total_sets': len(calibration_sets),
                'successful': any(set['avg_abs_sum_mm'] is not None and 
                                set['avg_abs_sum_mm'] < ERROR_THRESHOLD_MM 
                                for set in calibration_sets)
            },
            'calibration_sets': calibration_sets
        }, f, indent=2)
    
    print(f"\nCalibration results saved to: {filename}")

# ----------------- Main calibration loop -----------------
print("Eye-to-Hand Final Calibration")
print(f"Error threshold: {ERROR_THRESHOLD_MM} mm")
print("Press Ctrl+C to stop at any time")
print("=" * 60)

try:
    set_number = 1
    
    while True:
        # Take calibration set
        current_set = take_calibration_set(set_number)
        calibration_sets.append(current_set)
        
        # Check if we have error data and if it's within threshold
        if current_set['avg_abs_sum_mm'] is not None:
            if current_set['avg_abs_sum_mm'] < ERROR_THRESHOLD_MM:
                print(f"\n🎉 SUCCESS! Error {current_set['avg_abs_sum_mm']:.3f} mm is within threshold {ERROR_THRESHOLD_MM} mm")
                print(f"Calibration completed in {set_number} sets!")
                break
            else:
                print(f"Error {current_set['avg_abs_sum_mm']:.3f} mm exceeds threshold {ERROR_THRESHOLD_MM} mm")
                print("Continuing with next calibration set...")
        
        # Compare with previous set if available
        if set_number > 1:
            compare_calibration_sets(calibration_sets[set_number-2], current_set)
        
        # Ask user if they want to continue
        if set_number >= 1:
            response = input(f"\nTake calibration set {set_number + 1}? (y/n): ").lower().strip()
            if response != 'y' and response != 'yes':
                print("Calibration stopped by user")
                break
        
        set_number += 1

except KeyboardInterrupt:
    print("\n\nCalibration stopped by user")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    # Save results
    if calibration_sets:
        save_calibration_results()
    
    # Cleanup
    try:
        pipeline.stop()
    except:
        pass
    try:
        if arm:
            arm.disconnect()
    except:
        pass
    print("Cleanup completed.")
