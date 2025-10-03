#!/usr/bin/env python3
"""
ChArUco Marker Position Prediction
Detects ChArUco markers in camera frame and predicts their 3D position using:
- IPPE (Infinitesimal Plane-based Pose Estimation)
- Iterative refinement
- Subpixel corner detection
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple
from pyorbbecsdk import Pipeline, Context, Config, OBStreamType, OBFormat, OBSensorType


def load_intrinsics(intrinsics_path):
    """Load camera intrinsics from JSON file."""
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)

    # Handle two different JSON formats
    if 'camera_matrix' in data:
        # Format 1: Direct camera_matrix and dist_coeffs
        camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64).flatten()
    else:
        # Format 2: Nested format with fx, fy, cx, cy
        # Find the first key that contains camera parameters
        camera_key = None
        for key in data.keys():
            if isinstance(data[key], dict) and 'fx' in data[key]:
                camera_key = key
                break

        if camera_key is None:
            raise ValueError("Could not find camera intrinsics in JSON file")

        params = data[camera_key]
        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.array(params['distortion'], dtype=np.float64).flatten()

    return camera_matrix, dist_coeffs, data


def create_charuco_board(squares_x: int, squares_y: int, square_len_m: float, marker_len_m: float):
    """Create ChArUco board from configuration."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_len_m,
        marker_len_m,
        dictionary
    )

    return board, dictionary


def detect_aruco_markers(image, dictionary, camera_matrix, dist_coeffs):
    """
    Detect ArUco markers with subpixel refinement.

    Returns:
        marker_corners: ArUco marker corners (refined)
        marker_ids: ArUco marker IDs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers with subpixel refinement
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 30
    detector_params.cornerRefinementMinAccuracy = 0.01

    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    return marker_corners, marker_ids


def detect_charuco_with_refinement(image, board, dictionary, camera_matrix, dist_coeffs):
    """
    Detect ChArUco markers with subpixel refinement.

    Returns:
        charuco_corners: Refined corner positions
        charuco_ids: Corner IDs
        marker_corners: ArUco marker corners
        marker_ids: ArUco marker IDs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    marker_corners, marker_ids = detect_aruco_markers(image, dictionary, camera_matrix, dist_coeffs)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, None, None

    # Interpolate ChArUco corners
    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if charuco_corners is None or num_corners < 4:
        return None, None, marker_corners, marker_ids

    # Apply additional subpixel refinement to ChArUco corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    charuco_corners = cv2.cornerSubPix(
        gray,
        charuco_corners,
        (5, 5),  # winSize
        (-1, -1),  # zeroZone
        criteria
    )

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def estimate_aruco_pose_with_ippe(marker_corners, marker_ids, marker_size_m, camera_matrix, dist_coeffs):
    """
    Estimate pose of a single ArUco marker using IPPE and iterative refinement.

    Args:
        marker_corners: Detected marker corners
        marker_ids: Detected marker IDs
        marker_size_m: Physical size of the marker in meters
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        success: Whether pose estimation succeeded
        rvec: Rotation vector
        tvec: Translation vector
        reprojection_error: Mean reprojection error
        marker_id: ID of the detected marker
    """
    if marker_corners is None or len(marker_corners) == 0:
        return False, None, None, None, None

    # Use the first detected marker
    corners = marker_corners[0].reshape(-1, 2)
    marker_id = marker_ids[0][0]

    # Define 3D object points for a square marker (origin at top-left corner)
    half_size = marker_size_m / 2.0
    obj_points = np.array([
        [-half_size, half_size, 0],   # Top-left
        [half_size, half_size, 0],    # Top-right
        [half_size, -half_size, 0],   # Bottom-right
        [-half_size, -half_size, 0]   # Bottom-left
    ], dtype=np.float64)

    # Initial pose estimation with IPPE
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return False, None, None, None, marker_id

    # Iterative refinement using Levenberg-Marquardt
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points,
        corners,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec
    )

    # Calculate reprojection error
    projected_points, _ = cv2.projectPoints(
        obj_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    reprojection_error = np.mean(
        np.linalg.norm(projected_points - corners, axis=1)
    )

    return True, rvec, tvec, reprojection_error, marker_id


def estimate_pose_with_ippe(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs):
    """
    Estimate pose using IPPE and iterative refinement.

    Returns:
        success: Whether pose estimation succeeded
        rvec: Rotation vector
        tvec: Translation vector
        reprojection_error: Mean reprojection error
    """
    if charuco_corners is None or len(charuco_corners) < 4:
        return False, None, None, None

    # Get 3D object points for the detected ChArUco corners
    obj_points = board.getChessboardCorners()[charuco_ids.flatten()]

    # Initial pose estimation with IPPE
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        charuco_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return False, None, None, None

    # Iterative refinement using Levenberg-Marquardt
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points,
        charuco_corners,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec
    )

    # Calculate reprojection error
    projected_points, _ = cv2.projectPoints(
        obj_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    charuco_corners_2d = charuco_corners.reshape(-1, 2)

    reprojection_error = np.mean(
        np.linalg.norm(projected_points - charuco_corners_2d, axis=1)
    )

    return True, rvec, tvec, reprojection_error


def draw_pose(image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.05):
    """Draw 3D coordinate axes on the image."""
    # Draw axis
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    return image


def rotation_vector_to_euler(rvec):
    """Convert rotation vector to Euler angles (in degrees)."""
    R, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (XYZ convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


class OrbbecCamera:
    """Class to handle Orbbec camera streaming."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.pipeline = None
        self.config = None

    def initialize(self) -> bool:
        """Initialize the Orbbec camera pipeline."""
        try:
            print("Initializing Orbbec camera...")

            # Initialize pipeline
            self.pipeline = Pipeline()

            # Check for connected devices
            device_list = Context().query_devices()
            if len(device_list) == 0:
                print("❌ No Orbbec devices found!")
                return False

            device = device_list[0]
            print(f"✓ Found device: {device.get_device_info().get_name()}")

            # Get available color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

            # Find MJPG profile with desired resolution
            selected_profile = None
            for i in range(profile_list.get_count()):
                profile = profile_list.get_stream_profile_by_index(i)
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    if (vp.get_width() == self.width and
                        vp.get_height() == self.height and
                        vp.get_format() == OBFormat.MJPG):
                        selected_profile = profile
                        print(f"✓ Selected: {self.width}x{self.height} @ {vp.get_fps()}fps MJPG")
                        break

            if selected_profile is None:
                print(f"❌ No suitable profile found for {self.width}x{self.height}")
                return False

            # Configure and start pipeline
            self.config = Config()
            self.config.enable_stream(selected_profile)
            self.pipeline.start(self.config)

            import time
            time.sleep(2.0)

            print("✓ Camera initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        try:
            frames = None
            for attempt in range(3):
                frames = self.pipeline.wait_for_frames(3000)
                if frames is not None:
                    break

            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None

            # Decode MJPG frame
            color_data = np.asanyarray(color_frame.get_data())
            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            else:
                width = color_frame.get_width()
                height = color_frame.get_height()
                color_image = color_data.reshape((height, width, 3))
                if color_frame.get_format() == OBFormat.RGB:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            return color_image

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.pipeline:
                self.pipeline.stop()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Detect ArUco/ChArUco marker position in camera frame using Orbbec camera'
    )
    parser.add_argument(
        '--intrinsics',
        type=str,
        default='./gemini_intrinsics/gemini_355_rgb_intrinsics_20250930_181519.json',
        help='Path to camera intrinsics JSON file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['aruco', 'charuco'],
        default='charuco',
        help='Detection mode: aruco (single marker) or charuco (board) (default: charuco)'
    )
    parser.add_argument(
        '--aruco-size',
        type=float,
        default=0.05,
        help='ArUco marker size in meters (default: 0.05)'
    )
    parser.add_argument(
        '--squares-x',
        type=int,
        default=5,
        help='ChArUco: Number of squares in X direction (default: 5)'
    )
    parser.add_argument(
        '--squares-y',
        type=int,
        default=7,
        help='ChArUco: Number of squares in Y direction (default: 7)'
    )
    parser.add_argument(
        '--square-len',
        type=float,
        default=0.03718,
        help='ChArUco: Square length in meters (default: 0.03718)'
    )
    parser.add_argument(
        '--marker-len',
        type=float,
        default=0.03718 * 0.8,
        help='ChArUco: Marker length in meters (default: 0.029744)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file (if not using live camera)'
    )

    args = parser.parse_args()

    # Load camera intrinsics
    print(f"Loading intrinsics from: {args.intrinsics}")
    camera_matrix, dist_coeffs, intrinsics_data = load_intrinsics(args.intrinsics)

    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients: {dist_coeffs}")

    # Create dictionary and board based on mode
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = None

    if args.mode == 'charuco':
        board, dictionary = create_charuco_board(
            args.squares_x,
            args.squares_y,
            args.square_len,
            args.marker_len
        )
        print(f"\nMode: ChArUco Board")
        print(f"Board size: {args.squares_x}x{args.squares_y}")
        print(f"Square length: {args.square_len*1000:.2f}mm")
        print(f"Marker length: {args.marker_len*1000:.2f}mm")
    else:
        print(f"\nMode: Single ArUco Marker")
        print(f"Marker size: {args.aruco_size*1000:.2f}mm")

    # Process image or video stream
    if args.image:
        # Process single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from {args.image}")
            return

        output_frame = process_frame(
            image, args.mode, board, dictionary, camera_matrix, dist_coeffs,
            aruco_size=args.aruco_size,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_len=args.square_len
        )
        window_title = f'{args.mode.capitalize()} Position Detection'
        cv2.imshow(window_title, output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Process live Orbbec camera stream
        camera = OrbbecCamera(width=1920, height=1080)
        if not camera.initialize():
            return

        print("\nPress 'q' to quit, 's' to save current frame")

        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    continue

                output_frame = process_frame(
                    frame, args.mode, board, dictionary, camera_matrix, dist_coeffs,
                    aruco_size=args.aruco_size,
                    squares_x=args.squares_x,
                    squares_y=args.squares_y,
                    square_len=args.square_len
                )

                window_title = f'{args.mode.capitalize()} Position Detection - Orbbec Camera'
                cv2.imshow(window_title, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_path = f"output/{args.mode}_detection_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(save_path, output_frame)
                    print(f"Saved frame to: {save_path}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            camera.cleanup()
            cv2.destroyAllWindows()


def process_frame(frame, mode, board, dictionary, camera_matrix, dist_coeffs,
                  aruco_size=None, squares_x=None, squares_y=None, square_len=None, save_path=None):
    """Process a single frame to detect and estimate ArUco or ChArUco pose."""
    output_frame = frame.copy()

    if mode == 'aruco':
        # ArUco single marker mode
        marker_corners, marker_ids = detect_aruco_markers(frame, dictionary, camera_matrix, dist_coeffs)

        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(output_frame, marker_corners, marker_ids)

            # Estimate pose with IPPE and refinement
            success, rvec, tvec, reprojection_error, marker_id = estimate_aruco_pose_with_ippe(
                marker_corners, marker_ids, aruco_size, camera_matrix, dist_coeffs
            )

            if success:
                # Draw coordinate axes
                output_frame = draw_pose(output_frame, rvec, tvec, camera_matrix, dist_coeffs)

                # Convert rotation to Euler angles
                euler_angles = rotation_vector_to_euler(rvec)

                # Calculate perpendicular distance to marker plane
                R, _ = cv2.Rodrigues(rvec)
                n_cam = R[:, 2]  # marker normal in camera frame
                d_perp = abs(float(n_cam @ tvec.reshape(3)))

                # For ArUco, center is already at origin (tvec points to center)
                z_center = float(tvec[2])

                # Display pose information
                marker_text = f"ArUco ID: {marker_id} | Size: {aruco_size*1000:.2f}mm"
                position_text = f"Center Pos (m): X={tvec[0][0]:.4f}, Y={tvec[1][0]:.4f}, Z={tvec[2][0]:.4f}"
                distance_text = f"Plane Dist: {d_perp:.4f}m | Center Z: {z_center:.4f}m"
                rotation_text = f"Rotation (deg): Roll={euler_angles[0]:.2f}, Pitch={euler_angles[1]:.2f}, Yaw={euler_angles[2]:.2f}"
                error_text = f"Reproj. Error: {reprojection_error:.4f}px"

                # Draw text on image
                y_offset = 30
                cv2.putText(output_frame, marker_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                y_offset += 30
                cv2.putText(output_frame, position_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(output_frame, distance_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
                cv2.putText(output_frame, rotation_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(output_frame, error_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Print to console
                print("\n" + "="*60)
                print(marker_text)
                print(position_text)
                print(distance_text)
                print(rotation_text)
                print(error_text)
        else:
            cv2.putText(output_frame, "No ArUco marker detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:  # mode == 'charuco'
        # ChArUco board mode
        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_with_refinement(
            frame, board, dictionary, camera_matrix, dist_coeffs
        )

        # Draw detected ArUco markers
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(output_frame, marker_corners, marker_ids)

        # Draw detected ChArUco corners
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(output_frame, charuco_corners, charuco_ids)

            # Estimate pose with IPPE and refinement
            success, rvec, tvec, reprojection_error = estimate_pose_with_ippe(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs
            )

            if success:
                # Draw coordinate axes
                output_frame = draw_pose(output_frame, rvec, tvec, camera_matrix, dist_coeffs)

                # Convert rotation to Euler angles
                euler_angles = rotation_vector_to_euler(rvec)

                # Calculate perpendicular distance to board plane (invariant to X/Y motion)
                R, _ = cv2.Rodrigues(rvec)
                n_cam = R[:, 2]  # board normal expressed in camera frame
                d_perp = abs(float(n_cam @ tvec.reshape(3)))  # orthogonal distance (meters)

                # Calculate Z distance to board center (more stable than corner origin)
                dx = square_len * (squares_x - 1) / 2.0
                dy = square_len * (squares_y - 1) / 2.0
                offset_obj = np.array([dx, dy, 0.0], dtype=np.float64).reshape(3, 1)
                t_center = tvec + R @ offset_obj  # camera -> board-center translation
                z_center = float(t_center[2])

                # Display pose information
                position_text = f"Corner Pos (m): X={tvec[0][0]:.4f}, Y={tvec[1][0]:.4f}, Z={tvec[2][0]:.4f}"
                distance_text = f"Plane Dist: {d_perp:.4f}m | Center Z: {z_center:.4f}m"
                rotation_text = f"Rotation (deg): Roll={euler_angles[0]:.2f}, Pitch={euler_angles[1]:.2f}, Yaw={euler_angles[2]:.2f}"
                error_text = f"Reproj. Error: {reprojection_error:.4f}px"
                corners_text = f"Corners: {len(charuco_corners)}"

                # Draw text on image
                y_offset = 30
                cv2.putText(output_frame, position_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(output_frame, distance_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
                cv2.putText(output_frame, rotation_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(output_frame, error_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
                cv2.putText(output_frame, corners_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Print to console
                print("\n" + "="*60)
                print(position_text)
                print(distance_text)
                print(rotation_text)
                print(error_text)
                print(corners_text)
        else:
            cv2.putText(output_frame, "No ChArUco board detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if save_path:
        cv2.imwrite(save_path, output_frame)
        print(f"Saved output to: {save_path}")

    return output_frame


if __name__ == "__main__":
    main()
