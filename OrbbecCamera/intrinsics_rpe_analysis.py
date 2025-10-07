#!/usr/bin/env python3
"""
Simple Intrinsics RPE Analysis

Computes and visualizes reprojection errors per view and per feature
for intrinsics calibration analysis.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


def load_calibration_data(calibration_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera calibration data from JSON or NPZ file."""
    file_path = Path(calibration_file)

    if file_path.suffix == '.npz':
        # Load from NPZ file
        data = np.load(calibration_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    elif file_path.suffix == '.json':
        # Load from JSON file
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .npz")

    return camera_matrix, dist_coeffs


def compute_intrinsics_rpe(calibration_images: List[str],
                          pattern_size: Tuple[int, int],
                          square_size: float,
                          camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray,
                          pattern_type: str = 'charuco',
                          marker_size: float = None,
                          aruco_dict_type: int = cv2.aruco.DICT_4X4_50) -> Dict:
    """
    Compute RPE for each view and feature in intrinsics calibration.

    Args:
        calibration_images: List of image file paths
        pattern_size: (width, height) - for chessboard: inner corners, for charuco: board squares
        square_size: Size of squares in calibration pattern (meters)
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients
        pattern_type: 'chessboard' or 'charuco'
        marker_size: Size of ArUco markers (meters). If None, defaults to 0.75 * square_size
        aruco_dict_type: ArUco dictionary type

    Returns:
        Dictionary with RPE analysis results
    """
    # Prepare based on pattern type
    if pattern_type == 'charuco':
        # Set marker size if not provided
        if marker_size is None:
            marker_size = square_size * 0.8  # Match GenerateOrbbecIntrinsics.py

        # Create ChArUco board (matching GenerateOrbbecIntrinsics.py)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        charuco_board = cv2.aruco.CharucoBoard(
            (pattern_size[0], pattern_size[1]),
            square_size,
            marker_size,
            aruco_dict
        )
        # Create ArUco detector parameters
        aruco_params = cv2.aruco.DetectorParameters()
        print(f"Processing {len(calibration_images)} calibration images with ChArUco board...")
    else:
        # Prepare object points for chessboard
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        print(f"Processing {len(calibration_images)} calibration images with chessboard...")

    # Collect all object and image points
    object_points = []
    image_points = []
    valid_images = []

    for i, img_path in enumerate(calibration_images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path} - could not load")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if pattern_type == 'charuco':
            # Detect ChArUco corners (matching GenerateOrbbecIntrinsics.py method)
            # Step 1: Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=aruco_params)

            if ids is not None and len(ids) > 0:
                # Step 2: Interpolate ChArUco corners from detected markers
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=charuco_board
                )

                # Step 3: Refine corners with subpixel accuracy
                if response and response > 0 and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
                    charuco_corners = cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)

                    # Get corresponding object points for detected corners
                    obj_pts = charuco_board.getChessboardCorners()[charuco_ids.flatten()]

                    object_points.append(obj_pts)
                    image_points.append(charuco_corners)
                    valid_images.append(img_path)

                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(calibration_images)} images ({len(charuco_corners)} corners)")
                else:
                    print(f"  No ChArUco pattern found in {img_path} (too few corners)")
            else:
                print(f"  No ArUco markers found in {img_path}")
        else:
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                object_points.append(objp)
                image_points.append(corners.reshape(-1, 2))
                valid_images.append(img_path)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(calibration_images)} images")
            else:
                print(f"  No chessboard pattern found in {img_path}")
    
    if not valid_images:
        print("No valid calibration images found!")
        return {}
    
    print(f"Found {len(valid_images)} valid images with detected patterns")

    # Check if we should use provided intrinsics or calibrate
    use_provided_intrinsics = camera_matrix is not None and not np.array_equal(camera_matrix, np.eye(3))

    if use_provided_intrinsics:
        print("Using provided intrinsics to evaluate calibration quality...")
        print("Computing pose for each view with fixed intrinsics...")
        camera_matrix_cal = camera_matrix
        dist_coeffs_cal = dist_coeffs

        # Compute pose for each view using provided intrinsics
        rvecs = []
        tvecs = []
        for obj_pts, img_pts in zip(object_points, image_points):
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix_cal, dist_coeffs_cal,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                print("Warning: Pose estimation failed for one view")

        # Calculate overall RMS error manually
        total_error = 0
        total_points = 0
        for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            projected_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix_cal, dist_coeffs_cal)
            projected_points = projected_points.reshape(-1, 2)
            img_pts_reshaped = np.asarray(img_pts).reshape(-1, 2)
            error = np.linalg.norm(img_pts_reshaped - projected_points, axis=1)
            total_error += np.sum(error**2)
            total_points += len(error)

        ret = np.sqrt(total_error / total_points)
        print(f"Provided intrinsics RMS error: {ret:.3f} pixels")

    else:
        # Perform camera calibration from scratch
        print("No intrinsics provided - performing camera calibration...")
        ret, camera_matrix_cal, dist_coeffs_cal, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )

        if not ret:
            print("Calibration failed!")
            return {}

        print(f"Calibration RMS error: {ret:.3f} pixels")
    
    # Compute reprojection errors for each view
    rpe_per_view = []
    rpe_per_feature = []
    all_error_vectors = []
    
    for i, (obj_pts, img_pts, rvec, tvec) in enumerate(zip(object_points, image_points, rvecs, tvecs)):
        # Ensure img_pts is properly shaped (N, 2)
        img_pts = np.asarray(img_pts).reshape(-1, 2)

        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix_cal, dist_coeffs_cal
        )
        projected_points = projected_points.reshape(-1, 2)

        # Compute error vectors (observed - projected)
        error_vectors = img_pts - projected_points
        error_magnitudes = np.linalg.norm(error_vectors, axis=1)

        # Store per-view statistics
        view_rms = np.sqrt(np.mean(error_magnitudes**2))
        view_max = np.max(error_magnitudes)
        view_mean = np.mean(error_magnitudes)

        rpe_per_view.append({
            'image': valid_images[i],
            'rms': float(view_rms),
            'max': float(view_max),
            'mean': float(view_mean),
            'num_features': len(error_magnitudes)
        })

        # Store per-feature errors
        for j in range(len(error_vectors)):
            rpe_per_feature.append({
                'image_idx': i,
                'feature_idx': j,
                'error_x': float(error_vectors[j, 0]),
                'error_y': float(error_vectors[j, 1]),
                'magnitude': float(error_magnitudes[j])
            })
        
        all_error_vectors.append(error_vectors)
    
    # Overall statistics
    all_magnitudes = np.concatenate([np.linalg.norm(ev, axis=1) for ev in all_error_vectors])
    
    results = {
        'calibration_rms': float(ret),
        'camera_matrix': camera_matrix_cal,
        'distortion_coefficients': dist_coeffs_cal,
        'rpe_per_view': rpe_per_view,
        'rpe_per_feature': rpe_per_feature,
        'overall_stats': {
            'mean_rpe': float(np.mean(all_magnitudes)),
            'std_rpe': float(np.std(all_magnitudes)),
            'max_rpe': float(np.max(all_magnitudes)),
            'min_rpe': float(np.min(all_magnitudes))
        },
        'num_views': len(valid_images),
        'total_features': len(rpe_per_feature)
    }
    
    return results


def plot_rpe_per_view(results: Dict, output_dir: Path):
    """Plot RPE statistics per view."""
    rpe_per_view = results['rpe_per_view']
    
    # Extract data
    view_indices = list(range(len(rpe_per_view)))
    rms_errors = [view['rms'] for view in rpe_per_view]
    max_errors = [view['max'] for view in rpe_per_view]
    mean_errors = [view['mean'] for view in rpe_per_view]
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # RMS errors per view
    ax1.bar(view_indices, rms_errors, alpha=0.7, color='blue')
    ax1.set_xlabel('View Index')
    ax1.set_ylabel('RMS RPE (pixels)')
    ax1.set_title('RMS Reprojection Error per View')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(rms_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rms_errors):.3f}')
    ax1.legend()
    
    # Max errors per view
    ax2.bar(view_indices, max_errors, alpha=0.7, color='orange')
    ax2.set_xlabel('View Index')
    ax2.set_ylabel('Max RPE (pixels)')
    ax2.set_title('Maximum Reprojection Error per View')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(max_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(max_errors):.3f}')
    ax2.legend()
    
    # Mean errors per view
    ax3.bar(view_indices, mean_errors, alpha=0.7, color='green')
    ax3.set_xlabel('View Index')
    ax3.set_ylabel('Mean RPE (pixels)')
    ax3.set_title('Mean Reprojection Error per View')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(mean_errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(mean_errors):.3f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rpe_per_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ RPE per view plot saved to {output_dir / 'rpe_per_view.png'}")


def plot_rpe_per_feature(results: Dict, output_dir: Path):
    """Plot RPE statistics per feature."""
    rpe_per_feature = results['rpe_per_feature']

    # Extract data and ensure they're flat lists of floats
    magnitudes = np.array([float(feature['magnitude']) for feature in rpe_per_feature])
    error_x = np.array([float(feature['error_x']) for feature in rpe_per_feature])
    error_y = np.array([float(feature['error_y']) for feature in rpe_per_feature])
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of error magnitudes
    ax1.hist(magnitudes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('RPE Magnitude (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of RPE Magnitudes')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=np.mean(magnitudes), color='red', linestyle='--',
                label=f'Mean: {np.mean(magnitudes):.3f}')
    ax1.legend()
    
    # Error X vs Error Y scatter
    ax2.scatter(error_x, error_y, alpha=0.6, s=1)
    ax2.set_xlabel('Error X (pixels)')
    ax2.set_ylabel('Error Y (pixels)')
    ax2.set_title('Reprojection Error Vectors')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Box plot of error magnitudes
    ax3.boxplot(magnitudes)
    ax3.set_ylabel('RPE Magnitude (pixels)')
    ax3.set_title('RPE Magnitude Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_mags = np.sort(magnitudes)
    cumulative = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags)
    ax4.plot(sorted_mags, cumulative, linewidth=2)
    ax4.set_xlabel('RPE Magnitude (pixels)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of RPE Magnitudes')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rpe_per_feature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ RPE per feature plot saved to {output_dir / 'rpe_per_feature.png'}")


def plot_rpe_by_feature_index(results: Dict, output_dir: Path):
    """Plot average RPE per feature index across all views."""
    rpe_per_feature = results['rpe_per_feature']

    # Group by feature_idx and calculate average RPE
    from collections import defaultdict
    feature_rpe_groups = defaultdict(list)

    for feature in rpe_per_feature:
        feature_rpe_groups[feature['feature_idx']].append(feature['magnitude'])

    # Calculate statistics per feature index
    feature_indices = sorted(feature_rpe_groups.keys())
    avg_rpe = [np.mean(feature_rpe_groups[idx]) for idx in feature_indices]
    std_rpe = [np.std(feature_rpe_groups[idx]) for idx in feature_indices]
    max_rpe = [np.max(feature_rpe_groups[idx]) for idx in feature_indices]
    min_rpe = [np.min(feature_rpe_groups[idx]) for idx in feature_indices]
    counts = [len(feature_rpe_groups[idx]) for idx in feature_indices]

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Average RPE with error bars
    ax1.errorbar(feature_indices, avg_rpe, yerr=std_rpe, fmt='o-',
                 capsize=3, markersize=4, alpha=0.7, color='blue')
    ax1.set_xlabel('Feature Index (Corner ID)')
    ax1.set_ylabel('Average RPE (pixels)')
    ax1.set_title('Average Reprojection Error by Feature Index (with std dev)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(avg_rpe), color='red', linestyle='--',
                label=f'Overall Mean: {np.mean(avg_rpe):.3f}')
    ax1.legend()

    # Plot 2: Min/Max range
    ax2.fill_between(feature_indices, min_rpe, max_rpe, alpha=0.3, color='orange')
    ax2.plot(feature_indices, avg_rpe, 'b-', linewidth=2, label='Average')
    ax2.plot(feature_indices, max_rpe, 'r--', linewidth=1, label='Max', alpha=0.7)
    ax2.plot(feature_indices, min_rpe, 'g--', linewidth=1, label='Min', alpha=0.7)
    ax2.set_xlabel('Feature Index (Corner ID)')
    ax2.set_ylabel('RPE (pixels)')
    ax2.set_title('RPE Range by Feature Index')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Observation count per feature
    ax3.bar(feature_indices, counts, alpha=0.7, color='green')
    ax3.set_xlabel('Feature Index (Corner ID)')
    ax3.set_ylabel('Number of Observations')
    ax3.set_title('Feature Detection Frequency Across All Views')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(counts), color='red', linestyle='--',
                label=f'Mean: {np.mean(counts):.1f}')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'rpe_by_feature_index.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ RPE by feature index plot saved to {output_dir / 'rpe_by_feature_index.png'}")

    # Save feature index statistics to JSON
    feature_stats = {
        'feature_index_stats': []
    }

    for idx in feature_indices:
        feature_stats['feature_index_stats'].append({
            'feature_idx': int(idx),
            'avg_rpe': float(np.mean(feature_rpe_groups[idx])),
            'std_rpe': float(np.std(feature_rpe_groups[idx])),
            'max_rpe': float(np.max(feature_rpe_groups[idx])),
            'min_rpe': float(np.min(feature_rpe_groups[idx])),
            'observation_count': int(len(feature_rpe_groups[idx]))
        })

    with open(output_dir / 'feature_index_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2)

    print(f"✓ Feature index statistics saved to {output_dir / 'feature_index_stats.json'}")


def plot_rpe_direction_analysis(results: Dict, output_dir: Path):
    """Plot RPE error direction and magnitude using quiver plots and heatmaps."""
    from collections import defaultdict
    rpe_per_feature = results['rpe_per_feature']

    # Group errors by feature index
    feature_errors = defaultdict(lambda: {'x': [], 'y': [], 'mag': []})

    for feature in rpe_per_feature:
        idx = feature['feature_idx']
        feature_errors[idx]['x'].append(feature['error_x'])
        feature_errors[idx]['y'].append(feature['error_y'])
        feature_errors[idx]['mag'].append(feature['magnitude'])

    # Calculate average error direction per feature
    feature_indices = sorted(feature_errors.keys())
    avg_error_x = [np.mean(feature_errors[idx]['x']) for idx in feature_indices]
    avg_error_y = [np.mean(feature_errors[idx]['y']) for idx in feature_indices]
    avg_mag = [np.mean(feature_errors[idx]['mag']) for idx in feature_indices]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Quiver plot showing average error direction per feature
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.plasma(np.array(avg_mag) / max(avg_mag))
    quiver = ax1.quiver(feature_indices, np.zeros_like(feature_indices),
                        avg_error_x, avg_error_y,
                        avg_mag, cmap='plasma', scale=5, width=0.003, alpha=0.8)
    ax1.set_xlabel('Feature Index (Corner ID)')
    ax1.set_ylabel('Error Direction')
    ax1.set_title('Average Reprojection Error Direction by Feature Index')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    cbar1 = plt.colorbar(quiver, ax=ax1, label='Error Magnitude (pixels)')

    # Plot 2: Error X component by feature index
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(feature_indices, avg_error_x, alpha=0.7, color='blue')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Average Error X (pixels)')
    ax2.set_title('Horizontal Error Component')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linewidth=1, linestyle='--')

    # Plot 3: Error Y component by feature index
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(feature_indices, avg_error_y, alpha=0.7, color='green')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Average Error Y (pixels)')
    ax3.set_title('Vertical Error Component')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linewidth=1, linestyle='--')

    # Plot 4: 2D scatter showing error direction for all features
    ax4 = fig.add_subplot(gs[2, 0])
    all_error_x = [feature['error_x'] for feature in rpe_per_feature]
    all_error_y = [feature['error_y'] for feature in rpe_per_feature]
    all_mag = [feature['magnitude'] for feature in rpe_per_feature]

    scatter = ax4.scatter(all_error_x, all_error_y, c=all_mag,
                         cmap='plasma', alpha=0.5, s=10)
    ax4.set_xlabel('Error X (pixels)')
    ax4.set_ylabel('Error Y (pixels)')
    ax4.set_title('All Reprojection Error Vectors (colored by magnitude)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.axvline(x=0, color='black', linewidth=0.5)
    ax4.set_aspect('equal')
    cbar2 = plt.colorbar(scatter, ax=ax4, label='Error Magnitude (pixels)')

    # Plot 5: Average error direction on 2D scatter
    ax5 = fig.add_subplot(gs[2, 1])
    quiver2 = ax5.quiver(np.zeros_like(avg_error_x), np.zeros_like(avg_error_y),
                         avg_error_x, avg_error_y, avg_mag,
                         cmap='plasma', scale=3, width=0.005, alpha=0.8)
    ax5.set_xlabel('Error X (pixels)')
    ax5.set_ylabel('Error Y (pixels)')
    ax5.set_title('Average Error Direction per Feature (from origin)')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.axvline(x=0, color='black', linewidth=0.5)
    ax5.set_aspect('equal')

    # Add labels for features with highest errors
    top_n = 5
    top_indices = np.argsort(avg_mag)[-top_n:]
    for i in top_indices:
        ax5.annotate(f'F{feature_indices[i]}',
                    (avg_error_x[i], avg_error_y[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

    plt.savefig(output_dir / 'rpe_direction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ RPE direction analysis plot saved to {output_dir / 'rpe_direction_analysis.png'}")

    # Save direction statistics
    direction_stats = {
        'feature_direction_stats': []
    }

    for idx in feature_indices:
        direction_stats['feature_direction_stats'].append({
            'feature_idx': int(idx),
            'avg_error_x': float(np.mean(feature_errors[idx]['x'])),
            'avg_error_y': float(np.mean(feature_errors[idx]['y'])),
            'avg_magnitude': float(np.mean(feature_errors[idx]['mag'])),
            'std_error_x': float(np.std(feature_errors[idx]['x'])),
            'std_error_y': float(np.std(feature_errors[idx]['y'])),
            'error_angle_deg': float(np.degrees(np.arctan2(
                np.mean(feature_errors[idx]['y']),
                np.mean(feature_errors[idx]['x'])))),
            'observation_count': int(len(feature_errors[idx]['mag']))
        })

    with open(output_dir / 'rpe_direction_stats.json', 'w') as f:
        json.dump(direction_stats, f, indent=2)

    print(f"✓ RPE direction statistics saved to {output_dir / 'rpe_direction_stats.json'}")


def identify_outliers(results: Dict, output_dir: Path) -> Dict:
    """Identify outlier features based on statistical thresholds."""
    rpe_per_feature = results['rpe_per_feature']
    overall_stats = results['overall_stats']

    # Extract magnitudes
    magnitudes = np.array([feature['magnitude'] for feature in rpe_per_feature])
    mean_rpe = overall_stats['mean_rpe']
    std_rpe = overall_stats['std_rpe']

    # Define outlier thresholds
    threshold_1std = mean_rpe + std_rpe
    threshold_2std = mean_rpe + 2 * std_rpe
    threshold_3std = mean_rpe + 3 * std_rpe

    # Find outliers at different levels
    outliers_1std = []
    outliers_2std = []
    outliers_3std = []

    for feature in rpe_per_feature:
        mag = feature['magnitude']
        if mag > threshold_3std:
            outliers_3std.append(feature)
        elif mag > threshold_2std:
            outliers_2std.append(feature)
        elif mag > threshold_1std:
            outliers_1std.append(feature)

    outlier_info = {
        'thresholds': {
            '1_std': float(threshold_1std),
            '2_std': float(threshold_2std),
            '3_std': float(threshold_3std)
        },
        'counts': {
            'beyond_1std': len(outliers_1std) + len(outliers_2std) + len(outliers_3std),
            'beyond_2std': len(outliers_2std) + len(outliers_3std),
            'beyond_3std': len(outliers_3std)
        },
        'features_beyond_1std': outliers_1std,
        'features_beyond_2std': outliers_2std,
        'features_beyond_3std': outliers_3std
    }

    # Save outliers to separate file for easy access
    with open(output_dir / 'outliers.json', 'w') as f:
        json.dump(outlier_info, f, indent=2)

    # Print outlier summary
    print(f"\n{'='*60}")
    print("OUTLIER ANALYSIS")
    print(f"{'='*60}")
    print(f"Mean RPE: {mean_rpe:.3f} pixels")
    print(f"Std RPE:  {std_rpe:.3f} pixels")
    print(f"\nOutlier Thresholds:")
    print(f"  1σ (68%): {threshold_1std:.3f} pixels")
    print(f"  2σ (95%): {threshold_2std:.3f} pixels")
    print(f"  3σ (99.7%): {threshold_3std:.3f} pixels")
    print(f"\nOutlier Counts:")
    print(f"  Beyond 1σ: {outlier_info['counts']['beyond_1std']} features ({100*outlier_info['counts']['beyond_1std']/len(rpe_per_feature):.1f}%)")
    print(f"  Beyond 2σ: {outlier_info['counts']['beyond_2std']} features ({100*outlier_info['counts']['beyond_2std']/len(rpe_per_feature):.1f}%)")
    print(f"  Beyond 3σ: {outlier_info['counts']['beyond_3std']} features ({100*outlier_info['counts']['beyond_3std']/len(rpe_per_feature):.1f}%)")

    if len(outliers_3std) > 0:
        print(f"\n⚠️ Found {len(outliers_3std)} severe outliers (>3σ)")
        print("  Check outliers.json for details")

    return outlier_info


def print_analysis_summary(results: Dict):
    """Print analysis summary."""
    print("\n" + "="*60)
    print("INTRINSICS RPE ANALYSIS SUMMARY")
    print("="*60)
    
    overall = results['overall_stats']
    print(f"Calibration RMS Error: {results['calibration_rms']:.3f} pixels")
    print(f"Number of Views: {results['num_views']}")
    print(f"Total Features: {results['total_features']}")
    print(f"Mean RPE: {overall['mean_rpe']:.3f} pixels")
    print(f"Std RPE: {overall['std_rpe']:.3f} pixels")
    print(f"Max RPE: {overall['max_rpe']:.3f} pixels")
    print(f"Min RPE: {overall['min_rpe']:.3f} pixels")
    
    # View analysis
    rpe_per_view = results['rpe_per_view']
    view_rms = [view['rms'] for view in rpe_per_view]
    view_max = [view['max'] for view in rpe_per_view]
    
    print(f"\nPer-View Analysis:")
    print(f"  Best view RMS: {min(view_rms):.3f} pixels")
    print(f"  Worst view RMS: {max(view_rms):.3f} pixels")
    print(f"  Best view max: {min(view_max):.3f} pixels")
    print(f"  Worst view max: {max(view_max):.3f} pixels")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if overall['mean_rpe'] < 0.5:
        print("✅ Excellent calibration quality")
    elif overall['mean_rpe'] < 1.0:
        print("⚠️ Good calibration quality")
    elif overall['mean_rpe'] < 2.0:
        print("⚠️ Moderate calibration quality - room for improvement")
    else:
        print("❌ Poor calibration quality - needs improvement")
    
    if max(view_rms) > 2 * min(view_rms):
        print("⚠️ High variation between views - check image quality and coverage")
    
    if overall['max_rpe'] > 5.0:
        print("⚠️ Some features have high errors - check corner detection")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple Intrinsics RPE Analysis')
    parser.add_argument('--images', '-i', required=True,
                       help='Path to directory containing calibration images')
    parser.add_argument('--intrinsics', type=str, default=None,
                       help='Path to intrinsics file (.json or .npz). If not provided, calibration will be computed.')
    parser.add_argument('--output', '-o', default='intrinsics_rpe_analysis',
                       help='Output directory for results')
    parser.add_argument('--pattern-type', type=str, choices=['chessboard', 'charuco'], default='charuco',
                       help='Type of calibration pattern (chessboard or charuco)')
    parser.add_argument('--pattern-size', nargs=2, type=int, default=[5, 7],
                       help='For chessboard: inner corners (width height). For charuco: board squares (width height)')
    parser.add_argument('--square-size', type=float, default=0.037,
                       help='Size of squares in calibration pattern (meters)')
    parser.add_argument('--marker-size', type=float, default=None,
                       help='Size of ArUco markers (meters). Defaults to 0.8 * square_size. Only for charuco.')
    parser.add_argument('--aruco-dict', type=str, default='DICT_4X4_250',
                       help='ArUco dictionary type (e.g., DICT_4X4_50, DICT_5X5_100). Only for charuco.')
    parser.add_argument('--max-images', type=int, default=300,
                       help='Maximum number of images to process')

    args = parser.parse_args()

    # Parse ArUco dictionary
    aruco_dict_type = getattr(cv2.aruco, args.aruco_dict, cv2.aruco.DICT_4X4_250)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Find calibration images
    image_dir = Path(args.images)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No calibration images found in {args.images}")
        return
    
    # Limit number of images
    image_files = image_files[:args.max_images]
    print(f"Processing {len(image_files)} calibration images...")

    # Load intrinsics if provided
    if args.intrinsics:
        print(f"Loading intrinsics from {args.intrinsics}...")
        camera_matrix, dist_coeffs = load_calibration_data(args.intrinsics)
        print(f"Loaded camera matrix:\n{camera_matrix}")
        print(f"Loaded distortion coefficients: {dist_coeffs}")
    else:
        print("No intrinsics file provided - will compute calibration from images")
        camera_matrix = np.eye(3)
        dist_coeffs = np.zeros(5)

    # Compute RPE analysis
    results = compute_intrinsics_rpe(
        [str(f) for f in image_files],
        tuple(args.pattern_size),
        args.square_size,
        camera_matrix,
        dist_coeffs,
        pattern_type=args.pattern_type,
        marker_size=args.marker_size,
        aruco_dict_type=aruco_dict_type
    )
    
    if not results:
        print("RPE analysis failed!")
        return
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_rpe_per_view(results, output_dir)
    plot_rpe_per_feature(results, output_dir)
    plot_rpe_by_feature_index(results, output_dir)
    plot_rpe_direction_analysis(results, output_dir)
    
    # Print summary
    print_analysis_summary(results)
    
    # Identify outliers
    outliers = identify_outliers(results, output_dir)

    # Save results
    with open(output_dir / 'rpe_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif key == 'rpe_per_feature':
                # Convert feature data
                json_results[key] = []
                for feature in value:
                    json_results[key].append({
                        'image_idx': int(feature['image_idx']),
                        'feature_idx': int(feature['feature_idx']),
                        'error_x': float(feature['error_x']),
                        'error_y': float(feature['error_y']),
                        'magnitude': float(feature['magnitude'])
                    })
            elif key == 'rpe_per_view':
                # Convert view data
                json_results[key] = []
                for view in value:
                    json_results[key].append({
                        'image': str(view['image']),
                        'rms': float(view['rms']),
                        'max': float(view['max']),
                        'mean': float(view['mean']),
                        'num_features': int(view['num_features'])
                    })
            else:
                json_results[key] = value

        # Add outlier information
        json_results['outliers'] = outliers

        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("  - rpe_per_view.png: RPE statistics per view")
    print("  - rpe_per_feature.png: RPE distribution and patterns")
    print("  - rpe_by_feature_index.png: Average RPE by corner position")
    print("  - rpe_direction_analysis.png: Error direction vectors and components")
    print("  - rpe_results.json: Detailed numerical results")
    print("  - feature_index_stats.json: Statistics per feature index")
    print("  - rpe_direction_stats.json: Error direction statistics per feature")
    print("  - outliers.json: List of outlier features beyond 1σ, 2σ, and 3σ")


if __name__ == "__main__":
    main()
