#!/usr/bin/env python3
"""
RPE Gradient Map Generator for Extrinsics Calibration Analysis

This script generates reprojection error (RPE) gradient maps to visualize
the magnitude and direction of reprojection errors across calibration images.
This helps identify systematic biases, distortion modeling issues, and 
coordinate convention errors.

Author: Generated for extrinsics calibration analysis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml


class RPEGradientMapper:
    """
    Generates RPE gradient maps for extrinsics calibration analysis.
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize the RPE gradient mapper.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_size = None
        
    def compute_reprojection_errors(self, 
                                  object_points: List[np.ndarray],
                                  image_points: List[np.ndarray],
                                  rvecs: List[np.ndarray],
                                  tvecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reprojection errors for all calibration images.
        
        Args:
            object_points: List of 3D object points for each image
            image_points: List of 2D image points for each image
            rvecs: List of rotation vectors for each image
            tvecs: List of translation vectors for each image
            
        Returns:
            Tuple of (error_vectors, error_magnitudes)
        """
        all_error_vectors = []
        all_error_magnitudes = []
        
        for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Compute error vectors (observed - projected)
            error_vectors = img_pts - projected_points
            error_magnitudes = np.linalg.norm(error_vectors, axis=1)
            
            all_error_vectors.append(error_vectors)
            all_error_magnitudes.append(error_magnitudes)
            
        return np.vstack(all_error_vectors), np.concatenate(all_error_magnitudes)
    
    def create_gradient_heatmap(self, 
                              error_vectors: np.ndarray,
                              image_size: Tuple[int, int],
                              grid_size: int = 50) -> np.ndarray:
        """
        Create a gradient heatmap showing RPE magnitude distribution.
        
        Args:
            error_vectors: Array of error vectors (N, 2)
            image_size: (width, height) of the image
            grid_size: Size of the grid for interpolation
            
        Returns:
            Heatmap array
        """
        width, height = image_size
        x_coords = np.linspace(0, width, grid_size)
        y_coords = np.linspace(0, height, grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Get error magnitudes
        error_magnitudes = np.linalg.norm(error_vectors, axis=1)
        
        # Create grid for interpolation
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # For simplicity, we'll create a density-based heatmap
        # In practice, you might want to use more sophisticated interpolation
        heatmap = np.zeros((grid_size, grid_size))
        
        # Bin the error vectors by location
        for i, (x, y) in enumerate(points):
            # Find nearby error vectors
            distances = np.sqrt((error_vectors[:, 0] - x)**2 + (error_vectors[:, 1] - y)**2)
            nearby_mask = distances < (width / grid_size)
            
            if np.any(nearby_mask):
                # Average magnitude of nearby errors
                heatmap[i // grid_size, i % grid_size] = np.mean(error_magnitudes[nearby_mask])
        
        return heatmap
    
    def plot_quiver_map(self, 
                       error_vectors: np.ndarray,
                       image_points: np.ndarray,
                       image_size: Tuple[int, int],
                       max_arrows: int = 1000,
                       scale_factor: float = 1.0,
                       title: str = "RPE Gradient Map") -> plt.Figure:
        """
        Create a quiver plot showing RPE direction and magnitude.
        
        Args:
            error_vectors: Array of error vectors (N, 2)
            image_points: Array of image points where errors occur (N, 2)
            image_size: (width, height) of the image
            max_arrows: Maximum number of arrows to display
            scale_factor: Scale factor for arrow lengths
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Subsample if too many points
        if len(error_vectors) > max_arrows:
            indices = np.random.choice(len(error_vectors), max_arrows, replace=False)
            error_vectors = error_vectors[indices]
            image_points = image_points[indices]
        
        # Create quiver plot
        x, y = image_points[:, 0], image_points[:, 1]
        u, v = error_vectors[:, 0], error_vectors[:, 1]
        
        # Scale arrows for visibility
        magnitude = np.sqrt(u**2 + v**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        scale = scale_factor * 50 / max_mag
        
        quiver = ax.quiver(x, y, u, v, magnitude, 
                          cmap='viridis', scale=scale, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('RPE Magnitude (pixels)')
        
        # Set up the plot
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(image_size[1], 0)  # Flip Y axis for image coordinates
        ax.set_aspect('equal')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_heatmap(self, 
                    heatmap: np.ndarray,
                    image_size: Tuple[int, int],
                    title: str = "RPE Magnitude Heatmap") -> plt.Figure:
        """
        Create a heatmap showing RPE magnitude distribution.
        
        Args:
            heatmap: 2D array of error magnitudes
            image_size: (width, height) of the image
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(heatmap, cmap='hot', origin='upper', 
                      extent=[0, image_size[0], image_size[1], 0])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average RPE Magnitude (pixels)')
        
        # Set up the plot
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def analyze_error_patterns(self, 
                             error_vectors: np.ndarray,
                             image_points: np.ndarray) -> Dict:
        """
        Analyze error patterns to detect systematic biases.
        
        Args:
            error_vectors: Array of error vectors (N, 2)
            image_points: Array of image points (N, 2)
            
        Returns:
            Dictionary with analysis results
        """
        # Compute statistics
        error_magnitudes = np.linalg.norm(error_vectors, axis=1)
        mean_magnitude = np.mean(error_magnitudes)
        std_magnitude = np.std(error_magnitudes)
        max_magnitude = np.max(error_magnitudes)
        
        # Check for systematic patterns
        mean_error_vector = np.mean(error_vectors, axis=0)
        
        # Radial pattern detection (outward/inward bias)
        # Convert to polar coordinates
        distances = np.linalg.norm(image_points, axis=1)
        radial_errors = np.sum(error_vectors * image_points, axis=1) / (distances + 1e-8)
        
        # Tangential pattern detection
        # Rotate image points 90 degrees for tangential direction
        tangential_dirs = np.column_stack([-image_points[:, 1], image_points[:, 0]])
        tangential_errors = np.sum(error_vectors * tangential_dirs, axis=1) / (distances + 1e-8)
        
        analysis = {
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'max_magnitude': max_magnitude,
            'mean_error_vector': mean_error_vector,
            'radial_bias': np.mean(radial_errors),
            'tangential_bias': np.mean(tangential_errors),
            'total_points': len(error_vectors)
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print analysis results in a readable format."""
        print("\n" + "="*50)
        print("RPE GRADIENT MAP ANALYSIS")
        print("="*50)
        print(f"Total points analyzed: {analysis['total_points']}")
        print(f"Mean RPE magnitude: {analysis['mean_magnitude']:.3f} pixels")
        print(f"Std RPE magnitude: {analysis['std_magnitude']:.3f} pixels")
        print(f"Max RPE magnitude: {analysis['max_magnitude']:.3f} pixels")
        print(f"Mean error vector: [{analysis['mean_error_vector'][0]:.3f}, {analysis['mean_error_vector'][1]:.3f}]")
        print(f"Radial bias: {analysis['radial_bias']:.3f} pixels")
        print(f"Tangential bias: {analysis['tangential_bias']:.3f} pixels")
        
        print("\n" + "-"*30)
        print("INTERPRETATION:")
        print("-"*30)
        
        if analysis['mean_magnitude'] < 0.5:
            print("✅ Good: Low overall RPE magnitude")
        elif analysis['mean_magnitude'] < 1.0:
            print("⚠️  Moderate: RPE magnitude could be improved")
        else:
            print("❌ Poor: High RPE magnitude indicates calibration issues")
        
        if abs(analysis['radial_bias']) > 0.2:
            if analysis['radial_bias'] > 0:
                print("⚠️  Systematic outward bias detected - check radial distortion model")
            else:
                print("⚠️  Systematic inward bias detected - check radial distortion model")
        else:
            print("✅ Good: No significant radial bias")
        
        if abs(analysis['tangential_bias']) > 0.2:
            print("⚠️  Systematic tangential bias detected - check coordinate conventions")
        else:
            print("✅ Good: No significant tangential bias")


def load_calibration_data(calibration_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration data from file.
    
    Args:
        calibration_file: Path to calibration file (JSON or YAML)
        
    Returns:
        Tuple of (camera_matrix, dist_coeffs)
    """
    if calibration_file.endswith('.json'):
        with open(calibration_file, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
    elif calibration_file.endswith(('.yaml', '.yml')):
        with open(calibration_file, 'r') as f:
            data = yaml.safe_load(f)
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(data['distortion_coefficients']['data'])
    else:
        raise ValueError("Unsupported file format. Use JSON or YAML.")
    
    return camera_matrix, dist_coeffs


def main():
    """Main function to generate RPE gradient maps."""
    parser = argparse.ArgumentParser(description='Generate RPE gradient maps for extrinsics calibration')
    parser.add_argument('--calibration', '-c', required=True,
                       help='Path to camera calibration file (JSON or YAML)')
    parser.add_argument('--images', '-i', required=True,
                       help='Path to directory containing calibration images')
    parser.add_argument('--output', '-o', default='rpe_analysis',
                       help='Output directory for results')
    parser.add_argument('--pattern-size', type=float, default=1.0,
                       help='Size of calibration pattern squares')
    parser.add_argument('--pattern-corners', nargs=2, type=int, default=[9, 6],
                       help='Number of inner corners in pattern (width, height)')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load camera calibration
    print("Loading camera calibration...")
    camera_matrix, dist_coeffs = load_calibration_data(args.calibration)
    print(f"Camera matrix shape: {camera_matrix.shape}")
    print(f"Distortion coefficients: {dist_coeffs}")
    
    # Initialize RPE mapper
    mapper = RPEGradientMapper(camera_matrix, dist_coeffs)
    
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
    
    print(f"Found {len(image_files)} calibration images")
    
    # Prepare calibration pattern
    pattern_size = (args.pattern_corners[0], args.pattern_corners[1])
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.pattern_size
    
    # Collect object and image points
    object_points = []
    image_points = []
    rvecs = []
    tvecs = []
    valid_images = []
    
    print("Processing calibration images...")
    for i, img_file in enumerate(image_files[:args.max_images]):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Solve PnP
            ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            
            if ret:
                object_points.append(objp)
                image_points.append(corners.reshape(-1, 2))
                rvecs.append(rvec)
                tvecs.append(tvec)
                valid_images.append(img_file)
                
                if len(valid_images) % 10 == 0:
                    print(f"Processed {len(valid_images)} valid images...")
    
    if not valid_images:
        print("No valid calibration images found!")
        return
    
    print(f"Successfully processed {len(valid_images)} images")
    
    # Compute reprojection errors
    print("Computing reprojection errors...")
    error_vectors, error_magnitudes = mapper.compute_reprojection_errors(
        object_points, image_points, rvecs, tvecs
    )
    
    # Get image size from first image
    first_img = cv2.imread(str(valid_images[0]))
    image_size = (first_img.shape[1], first_img.shape[0])
    mapper.image_size = image_size
    
    # Analyze error patterns
    print("Analyzing error patterns...")
    analysis = mapper.analyze_error_patterns(error_vectors, np.vstack(image_points))
    mapper.print_analysis(analysis)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Quiver plot
    fig_quiver = mapper.plot_quiver_map(
        error_vectors, np.vstack(image_points), image_size,
        title=f"RPE Gradient Map ({len(valid_images)} images)"
    )
    fig_quiver.savefig(output_dir / 'rpe_quiver_map.png', dpi=300, bbox_inches='tight')
    plt.close(fig_quiver)
    
    # Heatmap
    heatmap = mapper.create_gradient_heatmap(error_vectors, image_size)
    fig_heatmap = mapper.plot_heatmap(
        heatmap, image_size,
        title=f"RPE Magnitude Heatmap ({len(valid_images)} images)"
    )
    fig_heatmap.savefig(output_dir / 'rpe_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_heatmap)
    
    # Save analysis results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v.tolist() if isinstance(v, np.ndarray) else v 
                  for k, v in analysis.items()}, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("  - rpe_quiver_map.png: Quiver plot showing error directions")
    print("  - rpe_heatmap.png: Heatmap showing error magnitudes")
    print("  - analysis_results.json: Quantitative analysis results")


if __name__ == "__main__":
    main()
