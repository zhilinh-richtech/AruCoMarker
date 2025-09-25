#!/usr/bin/env python3
"""
Performance comparison between fast and accurate modes
"""
import time
import sys
import os

# Add the Realsense directory to the path
sys.path.append('../Realsense')
from camera import create_camera
import cv2
import numpy as np

def test_performance():
    print("🔬 Performance Comparison: Fast vs Accurate Mode")
    print("=" * 50)
    
    # Test parameters
    num_frames = 30
    modes = [
        ("Fast Mode", True),
        ("Accurate Mode", False)
    ]
    
    results = {}
    
    for mode_name, fast_mode in modes:
        print(f"\n📊 Testing {mode_name}...")
        
        # Create camera
        cam = create_camera(kind='orbbec', width=1920, height=1080, fps=30)
        
        # Wait for initialization
        time.sleep(2)
        
        frame_times = []
        detection_times = []
        
        for i in range(num_frames):
            start_time = time.time()
            
            # Read frame
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
                
            frame_time = time.time() - start_time
            
            # Simulate detection (simplified version)
            start_detect = time.time()
            
            # Simulate subpixel refinement (only in accurate mode)
            if not fast_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Simulate corner refinement
                dummy_corners = np.array([[[100, 100]]], dtype=np.float32)
                cv2.cornerSubPix(gray, dummy_corners, (5, 5), (-1, -1), 
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            
            # Simulate Kalman filter (only in accurate mode)
            if not fast_mode:
                # Simulate some computation
                dummy_matrix = np.random.rand(3, 3)
                dummy_vector = np.random.rand(3)
                # Simulate matrix operations
                result = np.linalg.inv(dummy_matrix) @ dummy_vector
            
            detect_time = time.time() - start_detect
            
            frame_times.append(frame_time * 1000)  # Convert to ms
            detection_times.append(detect_time * 1000)  # Convert to ms
        
        cam.close()
        
        # Calculate statistics
        avg_frame_time = np.mean(frame_times)
        avg_detect_time = np.mean(detection_times)
        total_avg = avg_frame_time + avg_detect_time
        fps_estimate = 1000.0 / total_avg if total_avg > 0 else 0
        
        results[mode_name] = {
            'frame_time': avg_frame_time,
            'detect_time': avg_detect_time,
            'total_time': total_avg,
            'fps': fps_estimate
        }
        
        print(f"  📈 Average frame read time: {avg_frame_time:.2f} ms")
        print(f"  🔍 Average detection time: {avg_detect_time:.2f} ms")
        print(f"  ⏱️  Total processing time: {total_avg:.2f} ms")
        print(f"  🎯 Estimated FPS: {fps_estimate:.1f}")
    
    # Print comparison
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    fast_fps = results["Fast Mode"]["fps"]
    accurate_fps = results["Accurate Mode"]["fps"]
    speedup = fast_fps / accurate_fps if accurate_fps > 0 else 0
    
    print(f"Fast Mode FPS:     {fast_fps:.1f}")
    print(f"Accurate Mode FPS: {accurate_fps:.1f}")
    print(f"Speedup:           {speedup:.1f}x faster")
    
    if speedup > 1.5:
        print("✅ Significant speedup achieved in fast mode!")
    elif speedup > 1.1:
        print("⚡ Modest speedup achieved in fast mode")
    else:
        print("📊 Minimal speedup difference")

if __name__ == "__main__":
    test_performance()

