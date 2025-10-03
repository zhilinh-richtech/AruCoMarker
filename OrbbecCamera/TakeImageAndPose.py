#!/usr/bin/env python3
"""
Take Image and Pose Script

This script captures RGB images from an Orbbec camera and simultaneously
records the current robot pose from the xArm. The image and pose are saved
with matching filenames for easy correlation.

Requirements:
- pyorbbecsdk
- opencv-python
- numpy
- xarm-python-sdk

Usage:
    python TakeImageAndPose.py --output-dir output/poses_orbbec

Press 's' to save current frame and robot pose
Press 'q' or ESC to quit
"""

import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from typing import Optional, Tuple
from pyorbbecsdk import Pipeline, Context, Config, OBStreamType, OBFormat, OBSensorType
from xarm.wrapper import XArmAPI

class ImageAndPoseSaver:
    """Class to handle synchronized image and robot pose capture."""

    def __init__(self, xarm_ip: str = "192.168.10.202",
                 width: int = 1920, height: int = 1080, fps: int = 30,
                 output_dir: str = "output/poses_orbbec",
                 use_degrees: bool = True):
        """
        Initialize the image and pose saver.

        Args:
            xarm_ip: IP address of the xArm robot
            width: Image width (default: 1920)
            height: Image height (default: 1080)
            fps: Frames per second (default: 30)
            output_dir: Directory to save images and poses
            use_degrees: Use degrees for robot pose angles (default: True)
        """
        self.xarm_ip = xarm_ip
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.use_degrees = use_degrees

        self.pipeline = None
        self.config = None
        self.xarm = None
        self.save_counter = self._get_next_pose_number()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Images and poses will be saved to: {os.path.abspath(self.output_dir)}")
        print(f"Starting from pose number: {self.save_counter + 1}")

    def _get_next_pose_number(self) -> int:
        """
        Find the highest existing pose number in the output directory
        and return the next available number.

        Returns:
            int: Next available pose number (0 if no existing poses)
        """
        if not os.path.exists(self.output_dir):
            return 0

        max_num = 0
        for filename in os.listdir(self.output_dir):
            # Look for files matching pattern: pose###.jpg or pose###.npy
            if filename.startswith("pose") and (filename.endswith(".jpg") or filename.endswith(".npy")):
                try:
                    # Extract number from filename like "pose001.jpg" -> "001" -> 1
                    num_str = filename[4:7]  # Get characters at positions 4, 5, 6
                    num = int(num_str)
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    continue

        return max_num

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
                print("L No Orbbec devices found!")
                print("Please check:")
                print("  1. Camera is connected via USB")
                print("  2. Camera is not being used by another application")
                print("  3. You have proper permissions (try running with sudo)")
                return False

            device = device_list[0]
            print(f" Found device: {device.get_device_info().get_name()}")

            # Get available color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile_count = profile_list.get_count()
            print(f"Found {profile_count} color stream profiles")

            # Find the desired profile (1920x1080@30fps MJPG)
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

                    # Look for exact match (prefer MJPG for high resolution)
                    if (width == self.width and height == self.height and
                        format_type == OBFormat.MJPG):
                        selected_profile = profile
                        print(f" Selected: {width}x{height} @ {fps}fps, format: MJPG")
                        break

                    # Store fallback option (MJPG with correct resolution)
                    if (width == self.width and height == self.height and
                        format_type == OBFormat.MJPG and fallback_profile is None):
                        fallback_profile = profile

            # Use fallback if exact match not found
            if selected_profile is None and fallback_profile is not None:
                selected_profile = fallback_profile
                vp = selected_profile.as_video_stream_profile()
                print(f" Using fallback: {vp.get_width()}x{vp.get_height()} @ {vp.get_fps()}fps")

            if selected_profile is None:
                print(f"L No suitable profile found for {self.width}x{self.height} MJPG format")
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

            print(" Camera initialized successfully!")
            return True

        except Exception as e:
            print(f"L Failed to initialize camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialize_robot(self) -> bool:
        """
        Initialize connection to the xArm robot.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print(f"Connecting to xArm at {self.xarm_ip}...")
            self.xarm = XArmAPI(self.xarm_ip)
            self.xarm.connect()
            print(" xArm connected successfully!")
            return True
        except Exception as e:
            print(f"L Failed to connect to xArm: {e}")
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

            # Handle MJPG format (decode compressed data)
            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                if color_image is None:
                    print("Failed to decode MJPG frame")
                    return None
            else:
                # For other formats, handle as raw data
                width = color_frame.get_width()
                height = color_frame.get_height()

                if len(color_data.shape) == 1:
                    expected_size = width * height * 3
                    if color_data.size == expected_size:
                        color_image = color_data.reshape((height, width, 3))
                    else:
                        return None
                else:
                    color_image = color_data

                # Handle RGB format (convert to BGR for OpenCV)
                if color_frame.get_format() == OBFormat.RGB:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            return color_image

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def get_robot_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get current robot pose (position and orientation).

        Returns:
            Tuple of (rotation_matrix, translation_vector) or None if failed
        """
        try:
            code, pos = self.xarm.get_position(is_radian=False)
            if code != 0:
                print(f"� xArm get_position failed, code={code}")
                return None

            x, y, z, roll, pitch, yaw = pos

            # Convert position from mm to meters
            t = np.array([x, y, z], dtype=np.float64) / 1000.0

            # Convert Euler angles to rotation matrix
            if self.use_degrees:
                roll_rad = np.radians(roll)
                pitch_rad = np.radians(pitch)
                yaw_rad = np.radians(yaw)
            else:
                roll_rad, pitch_rad, yaw_rad = roll, pitch, yaw

            # Create rotation matrix from Euler angles (ZYX order)
            Rz = np.array([
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1]
            ])
            Ry = np.array([
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)]
            ])
            R = Rz @ Ry @ Rx

            return R, t

        except Exception as e:
            print(f"Error getting robot pose: {e}")
            return None

    def save_image_and_pose(self, image: np.ndarray, R: np.ndarray, t: np.ndarray) -> bool:
        """
        Save image and robot pose with matching filenames.

        Args:
            image: BGR image array to save
            R: Rotation matrix (3x3)
            t: Translation vector (3,)

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            self.save_counter += 1

            # Generate filenames with matching numbers
            base_filename = f"pose{self.save_counter:03d}"
            img_filepath = os.path.join(self.output_dir, f"{base_filename}.jpg")
            pose_filepath = os.path.join(self.output_dir, f"{base_filename}.npy")

            # Save image as JPEG
            success = cv2.imwrite(img_filepath, image)
            if not success:
                print(f"L Failed to save image: {img_filepath}")
                return False

            # Save pose data as numpy file
            pose_data = {
                "R": R,
                "t": t,
                "pose_number": self.save_counter,
                "timestamp": datetime.now().isoformat()
            }
            np.save(pose_filepath, pose_data, allow_pickle=True)

            print(f" Saved #{self.save_counter}:")
            print(f"  Image: {img_filepath}")
            print(f"  Pose:  {pose_filepath}")
            print(f"  Translation (m): [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
            print(f"  Rotation Matrix:")
            for row in R:
                print(f"    [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}]")
            print()

            return True

        except Exception as e:
            print(f"L Error saving image and pose: {e}")
            return False

    def run(self):
        """Main loop to capture and display frames."""
        if not self.initialize_camera():
            return

        if not self.initialize_robot():
            print("� Robot initialization failed. Will only save images without pose data.")

        print("\n" + "="*70)
        print("Image and Pose Saver - Controls:")
        print("  's' - Save current frame and robot pose")
        print("  'q' or ESC - Quit")
        print("="*70)

        try:
            while True:
                # Get frame
                frame = self.get_frame()

                if frame is None:
                    continue

                # Get robot pose
                pose = self.get_robot_pose() if self.xarm else None
                pose_status = " Robot connected" if pose is not None else " Robot disconnected"

                # Add text overlay with instructions and status
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press 's' to save, 'q' to quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Saved: {self.save_counter} pairs",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, pose_status,
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if pose else (0, 0, 255), 2)

                # Display frame
                cv2.imshow("Orbbec Camera + Robot Pose - Press 's' to save", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') or key == ord('S'):
                    # Save image and pose
                    if pose is None:
                        print("� Cannot get robot pose. Skipping save.")
                    else:
                        R, t = pose
                        self.save_image_and_pose(frame, R, t)

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
                print(" Camera pipeline stopped")
        except:
            pass

        try:
            if self.xarm:
                self.xarm.disconnect()
                print(" Robot disconnected")
        except:
            pass

        cv2.destroyAllWindows()
        print(f" Saved {self.save_counter} image-pose pairs total")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Capture images and robot poses simultaneously")
    parser.add_argument("--xarm-ip", type=str, default="192.168.10.202",
                       help="IP address of xArm robot (default: 192.168.10.202)")
    parser.add_argument("--width", type=int, default=1920,
                       help="Camera width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Camera height (default: 1080)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Camera FPS (default: 30)")
    parser.add_argument("--output-dir", type=str, default="./new_poses_folder",
                       help="Output directory for images and poses (default: ../output/poses_images)")
    parser.add_argument("--radians", action="store_true",
                       help="Use radians for robot angles (default: degrees)")

    args = parser.parse_args()

    print("Image and Pose Saver")
    print("=" * 70)

    # Create and run the saver
    saver = ImageAndPoseSaver(
        xarm_ip=args.xarm_ip,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=args.output_dir,
        use_degrees=not args.radians
    )
    saver.run()


if __name__ == "__main__":
    main()
