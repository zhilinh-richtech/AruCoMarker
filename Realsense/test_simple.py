#!/usr/bin/env python3
import os

# Set environment variables to avoid XCB issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':0'

print("Environment variables set:")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
print(f"DISPLAY: {os.environ.get('DISPLAY')}")

try:
    import cv2
    print("OpenCV imported successfully")
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"Error importing OpenCV: {e}")

try:
    import pyrealsense2 as rs
    print("RealSense imported successfully")
except Exception as e:
    print(f"Error importing RealSense: {e}")

print("Test completed.")
