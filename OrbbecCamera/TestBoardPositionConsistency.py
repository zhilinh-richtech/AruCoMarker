#!/usr/bin/env python3
"""
Test Board Position Consistency in Camera Frame

Captures multiple images with camera stationary and board fixed.
Measures how consistently the camera reports the same board position.

Good result: Std dev < 0.5mm translation, < 0.1° rotation
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from pyorbbecsdk import Pipeline, Config, OBSensorType


def pose_to_matrix(rvec, tvec):
    """Convert rvec, tvec to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def pose_difference(T1, T2):
    """Compute position and orientation difference between two poses."""
    # Position difference
    pos_diff = np.linalg.norm(T1[:3, 3] - T2[:3, 3]) * 1000  # in mm

    # Orientation difference
    R_diff = T1[:3, :3].T @ T2[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)

    return pos_diff, angle_deg


def main():
    parser = argparse.ArgumentParser(description='Test board position consistency')
    parser.add_argument('--intrinsics', required=True, help='Intrinsics file (.npz or .json)')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of samples to capture')
    parser.add_argument('--square-size', type=float, default=0.037, help='Square size in meters')
    parser.add_argument('--marker-size', type=float, default=0.037*0.8, help='Marker size in meters')
    parser.add_argument('--board-size', nargs=2, type=int, default=[5, 7], help='Board size')
    args = parser.parse_args()

    # Load intrinsics
    print(f"Loading intrinsics from {args.intrinsics}...")
    if args.intrinsics.endswith('.npz'):
        data = np.load(args.intrinsics)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    elif args.intrinsics.endswith('.json'):
        with open(args.intrinsics, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])

    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion: {dist_coeffs.flatten()}")

    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        tuple(args.board_size),
        args.square_size,
        args.marker_size,
        aruco_dict
    )
    detector_params = cv2.aruco.DetectorParameters()

    # Open Orbbec camera
    print(f"\nOpening Orbbec camera...")
    pipeline = Pipeline()
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        pipeline.start(config)
    except Exception as e:
        print(f"❌ Failed to open Orbbec camera: {e}")
        return

    print("\n" + "="*70)
    print("BOARD POSITION CONSISTENCY TEST")
    print("="*70)
    print("\nInstructions:")
    print("1. Position the ChArUco board in front of the camera")
    print("2. Keep BOTH camera and board completely stationary")
    print("3. Press SPACE to capture samples (or 'q' to start analysis)")
    print(f"4. Target: {args.num_samples} samples")
    print()

    poses = []

    try:
        while len(poses) < args.num_samples:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ChArUco
            corners, ids, _, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

            display = frame.copy()
            success = False

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)

                # Interpolate ChArUco corners
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )

                if response and charuco_corners is not None and len(charuco_corners) > 4:
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

                    # Estimate pose
                    success, rvec, tvec = cv2.solvePnP(
                        board.getChessboardCorners()[charuco_ids.flatten()],
                        charuco_corners,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                        cv2.putText(display, f"Samples: {len(poses)}/{args.num_samples}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display, "SPACE: Capture | Q: Analyze",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Board Position Test', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ids is not None and success:
                T = pose_to_matrix(rvec, tvec)
                poses.append(T)
                print(f"✓ Captured sample {len(poses)}/{args.num_samples}")
            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(poses) < 3:
        print(f"\n❌ Not enough samples: {len(poses)} (need at least 3)")
        return

    # Analyze consistency
    print("\n" + "="*70)
    print(f"ANALYZING {len(poses)} SAMPLES")
    print("="*70)

    # Compute mean pose
    T_mean = np.mean(poses, axis=0)

    # Compute deviations from mean
    pos_errors = []
    rot_errors = []

    for i, T in enumerate(poses):
        pos_err, rot_err = pose_difference(T, T_mean)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)

    print(f"\nPosition Consistency:")
    print(f"  Mean error:   {np.mean(pos_errors):.3f} mm")
    print(f"  Std dev:      {np.std(pos_errors):.3f} mm")
    print(f"  Max error:    {np.max(pos_errors):.3f} mm")
    print(f"  Min error:    {np.min(pos_errors):.3f} mm")

    print(f"\nOrientation Consistency:")
    print(f"  Mean error:   {np.mean(rot_errors):.3f}°")
    print(f"  Std dev:      {np.std(rot_errors):.3f}°")
    print(f"  Max error:    {np.max(rot_errors):.3f}°")
    print(f"  Min error:    {np.min(rot_errors):.3f}°")

    print("\n" + "="*70)
    print("QUALITY ASSESSMENT")
    print("="*70)

    pos_std = np.std(pos_errors)
    rot_std = np.std(rot_errors)

    if pos_std < 0.5 and rot_std < 0.1:
        print("✅ EXCELLENT consistency - Camera measurements are highly accurate")
    elif pos_std < 1.0 and rot_std < 0.2:
        print("✅ GOOD consistency - Suitable for hand-eye calibration")
    elif pos_std < 2.0 and rot_std < 0.5:
        print("⚠️ MODERATE consistency - May affect calibration quality")
    else:
        print("❌ POOR consistency - Check:")
        print("   - Camera/board actually stationary?")
        print("   - Lighting stable?")
        print("   - Intrinsics quality?")

    print(f"\nPosition std: {pos_std:.3f} mm (target: < 0.5 mm)")
    print(f"Rotation std: {rot_std:.3f}° (target: < 0.1°)")


if __name__ == "__main__":
    main()
