#!/usr/bin/env python3
"""
Camera calibration using mrcal library with ChArUco board images.

This script detects ChArUco corners in images and uses mrcal for camera calibration,
which provides more sophisticated lens models and optimization compared to OpenCV.
"""

import cv2
import numpy as np
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Import mrcal lazily to avoid early conflicts with OpenCV
mrcal = None

# ChArUco board parameters (matching the project standard)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M = 0.03718      # square side length in meters
MARKER_LEN_M = SQUARE_LEN_M * 0.8  # marker side length in meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250


def create_charuco_board():
    """Create ChArUco board object."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        aruco_dict
    )
    return aruco_dict, board


def detect_charuco_corners(image_path: str, aruco_dict, board, visualize: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect ChArUco corners in an image.

    Args:
        image_path: Path to image file
        aruco_dict: ArUco dictionary
        board: ChArUco board object
        visualize: Show detection visualization

    Returns:
        Tuple of (charuco_corners, charuco_ids) or None if detection failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect ArUco markers (no corner refinement for ChArUco)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Step 2: Interpolate ChArUco corners
        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
            return None

        # Step 3: Subpixel refinement on ChArUco corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        charuco_corners = cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)

        if visualize:
            vis_img = image.copy()
            cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
            cv2.imshow(f"ChArUco Detection - {os.path.basename(image_path)}", vis_img)
            cv2.waitKey(100)

        return charuco_corners, charuco_ids

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def calibrate_with_mrcal(all_corners: List[np.ndarray], all_ids: List[np.ndarray],
                         board, image_shape: Tuple[int, int],
                         lens_model: str = 'LENSMODEL_OPENCV8') -> dict:
    """
    Perform camera calibration using mrcal.

    Args:
        all_corners: List of detected corner arrays per image
        all_ids: List of corner ID arrays per image
        board: ChArUco board object
        image_shape: (height, width) of images
        lens_model: mrcal lens model (OPENCV4, OPENCV5, OPENCV8, etc.)

    Returns:
        Dictionary with calibration results
    """
    try:
        # Import mrcal here to avoid conflicts
        global mrcal
        if mrcal is None:
            import mrcal as mrcal_module
            mrcal = mrcal_module

        print(f"\nPreparing observations for mrcal calibration...")

        height, width = image_shape
        num_frames = len(all_corners)

        # Get object points (3D coordinates of ChArUco corners in board frame)
        object_points_all = board.getChessboardCorners()

        # Build flat arrays of observations and corresponding object points
        observations_list = []
        object_points_list = []
        frame_indices_list = []

        for frame_idx, (corners, ids) in enumerate(zip(all_corners, all_ids)):
            for corner, corner_id in zip(corners, ids):
                corner_id_int = int(corner_id[0])
                if corner_id_int < len(object_points_all):
                    observations_list.append(corner.flatten().astype(np.float64))
                    object_points_list.append(object_points_all[corner_id_int].astype(np.float64))
                    frame_indices_list.append(frame_idx)

        observations = np.array(observations_list, dtype=np.float64)
        object_points = np.array(object_points_list, dtype=np.float64)
        frame_indices = np.array(frame_indices_list, dtype=np.int32)

        print(f"  - Image size: {width}x{height}")
        print(f"  - Number of frames: {num_frames}")
        print(f"  - Total observations: {len(observations)}")
        print(f"  - Lens model: {lens_model}")

        # Initial intrinsics estimate
        focal_length_estimate = width
        if lens_model == 'LENSMODEL_OPENCV4':
            intrinsics_seed = np.array([focal_length_estimate, focal_length_estimate,
                                       width/2, height/2, 0, 0, 0, 0], dtype=np.float64)
        elif lens_model == 'LENSMODEL_OPENCV5':
            intrinsics_seed = np.array([focal_length_estimate, focal_length_estimate,
                                       width/2, height/2, 0, 0, 0, 0, 0], dtype=np.float64)
        elif lens_model == 'LENSMODEL_OPENCV8':
            intrinsics_seed = np.array([focal_length_estimate, focal_length_estimate,
                                       width/2, height/2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        else:
            intrinsics_seed = np.array([focal_length_estimate, focal_length_estimate,
                                       width/2, height/2], dtype=np.float64)

        # Create indices array for mrcal: [frame_idx, camera_idx, extrinsics_idx]
        indices_frame_camera_extrinsics = np.column_stack([
            frame_indices,
            np.zeros(len(frame_indices), dtype=np.int32),  # camera index (all 0)
            frame_indices  # extrinsics index (one per frame)
        ])

        # Initial extrinsics (rt: rotation + translation in axis-angle form)
        extrinsics_rt_fromref = np.zeros((num_frames, 6), dtype=np.float64)

        print(f"\nRunning mrcal optimization...")

        # Run mrcal calibration
        stats = mrcal.optimize(
            intrinsics=np.array([intrinsics_seed]),
            extrinsics_rt_fromref=extrinsics_rt_fromref,
            frames_rt_toref=None,
            points=None,
            observations_board=observations,
            indices_frame_camintrinsics_camextrinsics=indices_frame_camera_extrinsics,
            observations_point=None,
            indices_point_camintrinsics_camextrinsics=None,
            lensmodel=lens_model,
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

        # Extract optimized intrinsics
        intrinsics_optimized = stats['intrinsics'][0]

        # Compute RMS error
        rms_error = np.sqrt(stats['x_squared'] / len(observations))

        print(f"\n✓ Calibration successful!")
        print(f"  - RMS reprojection error: {rms_error:.4f} pixels")

        return {
            'success': True,
            'intrinsics': intrinsics_optimized,
            'lens_model': lens_model,
            'rms_error': rms_error,
            'stats': stats,
            'image_size': (width, height),
            'num_frames': num_frames,
            'num_observations': len(observations)
        }

    except Exception as e:
        print(f"❌ Mrcal calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def save_calibration(results: dict, output_path: str):
    """Save calibration results to file."""
    if not results['success']:
        print("Cannot save failed calibration")
        return

    output_path = Path(output_path)

    # Save as JSON (OpenCV-compatible format if possible)
    if output_path.suffix == '.json':
        # For OPENCV models, extract fx, fy, cx, cy
        if 'OPENCV' in results['lens_model']:
            fx, fy, cx, cy = results['intrinsics'][:4]
            dist_coeffs = results['intrinsics'][4:].tolist()

            data = {
                'lens_model': results['lens_model'],
                'image_size': results['image_size'],
                'camera_matrix': [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ],
                'distortion_coefficients': dist_coeffs,
                'rms_reprojection_error': results['rms_error'],
                'num_frames': results['num_frames'],
                'num_observations': results['num_observations']
            }
        else:
            # Generic format
            data = {
                'lens_model': results['lens_model'],
                'intrinsics': results['intrinsics'].tolist(),
                'image_size': results['image_size'],
                'rms_reprojection_error': results['rms_error'],
                'num_frames': results['num_frames'],
                'num_observations': results['num_observations']
            }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Save as NPZ (NumPy format)
    elif output_path.suffix == '.npz':
        if 'OPENCV' in results['lens_model']:
            fx, fy, cx, cy = results['intrinsics'][:4]
            camera_matrix = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            dist_coeffs = results['intrinsics'][4:]

            np.savez(
                output_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                lens_model=results['lens_model'],
                image_size=np.array(results['image_size']),
                rms_error=results['rms_error']
            )
        else:
            np.savez(
                output_path,
                intrinsics=results['intrinsics'],
                lens_model=results['lens_model'],
                image_size=np.array(results['image_size']),
                rms_error=results['rms_error']
            )

    print(f"✓ Calibration saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Camera calibration using mrcal with ChArUco board')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing calibration images')
    parser.add_argument('--output', type=str, default='mrcal_calibration.json',
                        help='Output calibration file (.json or .npz)')
    parser.add_argument('--lens-model', type=str, default='LENSMODEL_OPENCV8',
                        choices=['LENSMODEL_OPENCV4', 'LENSMODEL_OPENCV5', 'LENSMODEL_OPENCV8',
                                'LENSMODEL_OPENCV12'],
                        help='Mrcal lens model to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize ChArUco detection')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                        help='Image file pattern (default: *.jpg)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to use')

    args = parser.parse_args()

    # Create ChArUco board
    aruco_dict, board = create_charuco_board()

    # Find images
    image_paths = sorted(glob.glob(os.path.join(args.images_dir, args.pattern)))
    if not image_paths:
        print(f"❌ No images found in {args.images_dir} matching {args.pattern}")
        return 1

    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images in {args.images_dir}")

    # Detect ChArUco corners in all images
    all_corners = []
    all_ids = []
    valid_images = []
    image_shape = None

    print("Starting ChArUco detection...")
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            result = detect_charuco_corners(image_path, aruco_dict, board, args.visualize)
            if result is not None:
                corners, ids = result
                all_corners.append(corners)
                all_ids.append(ids)
                valid_images.append(image_path)

                if image_shape is None:
                    img = cv2.imread(image_path)
                    image_shape = img.shape[:2]
        except Exception as e:
            print(f"Exception processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

    if args.visualize:
        cv2.destroyAllWindows()

    if len(all_corners) == 0:
        print("❌ No valid ChArUco detections found")
        return 1

    print(f"\n✓ Successfully detected ChArUco in {len(all_corners)}/{len(image_paths)} images")

    # Calibrate with mrcal
    results = calibrate_with_mrcal(all_corners, all_ids, board, image_shape, args.lens_model)

    if results['success']:
        # Save calibration
        save_calibration(results, args.output)
        return 0
    else:
        print(f"❌ Calibration failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
