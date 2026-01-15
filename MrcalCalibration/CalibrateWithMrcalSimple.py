#!/usr/bin/env python3
"""
Camera calibration using mrcal library with ChArUco board images.
"""

import cv2
import numpy as np
import argparse
import glob
import os
import sys
from pathlib import Path
import json

# ChArUco board parameters
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
SQUARE_LEN_M = 0.03718
MARKER_LEN_M = SQUARE_LEN_M * 0.8
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250


def main():
    parser = argparse.ArgumentParser(description='Camera calibration using mrcal with ChArUco board')
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='mrcal_calibration.json')
    parser.add_argument('--lens-model', type=str, default='LENSMODEL_OPENCV8')
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()

    # Create ChArUco board
    print("Creating ChArUco board...")
    sys.stdout.flush()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    print("  Dictionary created")
    sys.stdout.flush()
    board = cv2.aruco.CharucoBoard((CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), SQUARE_LEN_M, MARKER_LEN_M, aruco_dict)
    print("  Board created")
    sys.stdout.flush()

    # Find images
    image_paths = sorted(glob.glob(os.path.join(args.images_dir, '*.jpg')))
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return 1

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images")
    sys.stdout.flush()

    # Detect ChArUco corners
    all_corners = []
    all_ids = []
    image_shape = None

    print("Creating detector...")
    sys.stdout.flush()
    params = cv2.aruco.DetectorParameters()
    print("  Params created")
    sys.stdout.flush()
    # Don't set cornerRefinementMethod - default is already NONE
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    print("  Detector created")
    sys.stdout.flush()

    print("Detecting ChArUco corners...")
    sys.stdout.flush()
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(image_paths)}")

        image = cv2.imread(image_path)
        if image is None:
            continue

        if image_shape is None:
            image_shape = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            continue

        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=board
        )

        if charuco_corners is None or len(charuco_corners) < 4:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        charuco_corners = cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)

    print(f"Successfully detected ChArUco in {len(all_corners)}/{len(image_paths)} images")

    if len(all_corners) == 0:
        print("No valid detections")
        return 1

    # Import mrcal after detection
    print("\nImporting mrcal...")
    import mrcal

    # Prepare observations
    print("Preparing observations for mrcal...")
    height, width = image_shape
    num_frames = len(all_corners)
    object_points_all = board.getChessboardCorners()

    observations_list = []
    frame_indices_list = []

    for frame_idx, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        for corner, corner_id in zip(corners, ids):
            corner_id_int = int(corner_id[0])
            if corner_id_int < len(object_points_all):
                observations_list.append(corner.flatten().astype(np.float64))
                frame_indices_list.append(frame_idx)

    observations = np.array(observations_list, dtype=np.float64)
    frame_indices = np.array(frame_indices_list, dtype=np.int32)

    print(f"  Frames: {num_frames}, Observations: {len(observations)}")

    # Setup intrinsics seed
    focal_estimate = width
    if args.lens_model == 'LENSMODEL_OPENCV4':
        intrinsics_seed = np.array([focal_estimate, focal_estimate, width/2, height/2, 0, 0, 0, 0], dtype=np.float64)
    elif args.lens_model == 'LENSMODEL_OPENCV5':
        intrinsics_seed = np.array([focal_estimate, focal_estimate, width/2, height/2, 0, 0, 0, 0, 0], dtype=np.float64)
    elif args.lens_model == 'LENSMODEL_OPENCV8':
        intrinsics_seed = np.array([focal_estimate, focal_estimate, width/2, height/2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    else:
        intrinsics_seed = np.array([focal_estimate, focal_estimate, width/2, height/2], dtype=np.float64)

    indices_frame_camera_extrinsics = np.column_stack([
        frame_indices,
        np.zeros(len(frame_indices), dtype=np.int32),
        frame_indices
    ])

    extrinsics_rt_fromref = np.zeros((num_frames, 6), dtype=np.float64)

    # Run mrcal
    print(f"\nRunning mrcal optimization ({args.lens_model})...")
    stats = mrcal.optimize(
        intrinsics=np.array([intrinsics_seed]),
        extrinsics_rt_fromref=extrinsics_rt_fromref,
        frames_rt_toref=None,
        points=None,
        observations_board=observations,
        indices_frame_camintrinsics_camextrinsics=indices_frame_camera_extrinsics,
        observations_point=None,
        indices_point_camintrinsics_camextrinsics=None,
        lensmodel=args.lens_model,
        calobject_warp=None,
        imagersizes=np.array([[width, height]], dtype=np.int32),
        calibration_object_spacing=SQUARE_LEN_M,
        calibration_object_width_n=CHARUCO_SQUARES_X - 1,
        calibration_object_height_n=CHARUCO_SQUARES_Y - 1,
        verbose=False,
        do_optimize_intrinsics_core=True,
        do_optimize_intrinsics_distortions=True,
        do_optimize_extrinsics=True,
        do_optimize_frames=False,
        do_optimize_calobject_warp=False,
    )

    intrinsics_optimized = stats['intrinsics'][0]
    rms_error = np.sqrt(stats['x_squared'] / len(observations))

    print(f"\n✓ Calibration successful!")
    print(f"  RMS error: {rms_error:.4f} pixels")

    # Save results
    output_path = Path(args.output)
    if 'OPENCV' in args.lens_model:
        fx, fy, cx, cy = intrinsics_optimized[:4]
        dist_coeffs = intrinsics_optimized[4:].tolist()

        data = {
            'lens_model': args.lens_model,
            'image_size': [width, height],
            'camera_matrix': [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            'distortion_coefficients': dist_coeffs,
            'rms_reprojection_error': rms_error,
            'num_frames': num_frames,
            'num_observations': len(observations)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
