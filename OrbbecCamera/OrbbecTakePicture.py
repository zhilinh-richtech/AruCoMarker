#!/usr/bin/env python3
"""
Orbbec Camera Image Saver Script

This script captures RGB images from an Orbbec camera at 1920x1080 resolution and 30fps.
Press 's' to save the current frame as an image.
Press 'q' or ESC to quit.

Requirements:
- pyorbbecsdk
- opencv-python
- numpy

Usage:
    python orbbec_image_saver.py
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional
from pyorbbecsdk import Pipeline, Context, Config, OBStreamType, OBFormat, OBSensorType

class OrbbecImageSaver:
    """Class to handle Orbbec camera image capture and saving."""
    
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        """
        Initialize the Orbbec image saver.
        
        Args:
            width: Image width (default: 1920)
            height: Image height (default: 1080) 
            fps: Frames per second (default: 30)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.save_counter = 0
        self.output_dir = "saved_images"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(self.output_dir)}")
    
    def initialize_camera(self) -> bool:
        """
        Initialize the Orbbec camera pipeline.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("Initializing Orbbec camera...")
            
            # Initialize pipeline
            self.pipeline = Pipeline()
            
            # Check for connected devices
            device_list = Context().query_devices()
            if len(device_list) == 0:
                print("❌ No Orbbec devices found!")
                print("Please check:")
                print("  1. Camera is connected via USB")
                print("  2. Camera is not being used by another application")
                print("  3. You have proper permissions (try running with sudo)")
                return False
            
            device = device_list[0]
            print(f"✓ Found device: {device.get_device_info().get_name()}")
            
            # Get available color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile_count = profile_list.get_count()
            print(f"Found {profile_count} color stream profiles")
            
            # Find the desired profile (1920x1080@30fps RGB)
            selected_profile = None
            fallback_profile = None
            
            for i in range(profile_count):
                profile = profile_list.get_stream_profile_by_index(i)
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    width = vp.get_width()
                    height = vp.get_height()
                    format_type = vp.get_format()
                    fps = vp.get_fps()
                    
                    print(f"  Available: {width}x{height} @ {fps}fps, format: {format_type}")
                    
                    # Look for exact match first (prefer MJPG for high resolution)
                    if (width == self.width and height == self.height and 
                        format_type == OBFormat.MJPG):
                        selected_profile = profile
                        print(f"✓ Selected exact match: {width}x{height} @ {fps}fps, format: MJPG")
                        break
                    
                    # Store fallback option (MJPG with correct resolution)
                    if (width == self.width and height == self.height and 
                        format_type == OBFormat.MJPG and fallback_profile is None):
                        fallback_profile = profile
                    
                    # Store best MJPG option (highest resolution MJPG available)
                    if (format_type == OBFormat.MJPG and 
                        (fallback_profile is None or width * height > 
                         fallback_profile.as_video_stream_profile().get_width() * 
                         fallback_profile.as_video_stream_profile().get_height())):
                        fallback_profile = profile
            
            # Use fallback if exact match not found
            if selected_profile is None and fallback_profile is not None:
                selected_profile = fallback_profile
                vp = selected_profile.as_video_stream_profile()
                print(f"✓ Using fallback profile: {vp.get_width()}x{vp.get_height()} @ {vp.get_fps()}fps")
            
            if selected_profile is None:
                print(f"❌ No suitable profile found for {self.width}x{self.height} RGB format")
                print("Available profiles listed above. Consider using a different resolution.")
                return False
            
            # Configure pipeline
            self.config = Config()
            self.config.enable_stream(selected_profile)
            
            # Start pipeline
            print("Starting camera pipeline...")
            self.pipeline.start(self.config)
            
            # Wait for pipeline to stabilize
            import time
            time.sleep(2.0)
            
            print("✓ Camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            np.ndarray: BGR image array, or None if failed
        """
        try:
            # Try multiple times as first attempt often times out
            frames = None
            for attempt in range(3):
                frames = self.pipeline.wait_for_frames(3000)  # 3 second timeout
                if frames is not None:
                    break
                if attempt < 2:
                    print(f"Frame capture attempt {attempt+1} timed out, retrying...")
            
            if frames is None:
                return None
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None
            
            # Convert to numpy array
            color_data = np.asanyarray(color_frame.get_data())
            
            if color_data is None or color_data.size == 0:
                return None
            
            # Handle different data formats
            width = color_frame.get_width()
            height = color_frame.get_height()
            
            # For MJPG format, data is compressed and should be decoded directly
            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if color_image is None:
                    print("Failed to decode MJPG frame")
                    return None
            else:
                # For other formats (RGB, BGR, etc.), handle as raw data
                # If data is 1D, reshape it
                if len(color_data.shape) == 1:
                    expected_size = width * height * 3
                    if color_data.size == expected_size:
                        color_image = color_data.reshape((height, width, 3))
                    else:
                        print(f"Size mismatch: expected {expected_size}, got {color_data.size}")
                        return None
                else:
                    color_image = color_data
            
            # Handle different color formats
            if color_frame.get_format() == OBFormat.RGB:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            return color_image
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def save_image(self, image: np.ndarray) -> bool:
        """
        Save an image to disk.
        
        Args:
            image: BGR image array to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_counter += 1
            filename = f"orbbec_image_{timestamp}_{self.save_counter:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            success = cv2.imwrite(filepath, image)
            
            if success:
                print(f"✓ Saved image: {filename}")
                return True
            else:
                print(f"❌ Failed to save image: {filename}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False
    
    def run(self):
        """Main loop to capture and display frames."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("Orbbec Image Saver - Controls:")
        print("  's' - Save current frame")
        print("  'q' or ESC - Quit")
        print("="*50)
        
        try:
            while True:
                # Get frame
                frame = self.get_frame()
                
                if frame is None:
                    continue
                
                # Add text overlay with instructions
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press 's' to save, 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Saved: {self.save_counter} images", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Orbbec Camera - Press 's' to save", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s') or key == ord('S'):
                    # Save image
                    self.save_image(frame)
                
                elif key == ord('q') or key == ord('Q') or key == 27:  # ESC key
                    print("Quitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.pipeline:
                self.pipeline.stop()
                print("✓ Camera pipeline stopped")
        except:
            pass
        
        cv2.destroyAllWindows()
        print(f"✓ Saved {self.save_counter} images total")


def main():
    """Main function."""
    print("Orbbec Camera Image Saver")
    print("========================")
    
    # Create and run the image saver
    saver = OrbbecImageSaver(width=1920, height=1080, fps=30)
    saver.run()


if __name__ == "__main__":
    main()
