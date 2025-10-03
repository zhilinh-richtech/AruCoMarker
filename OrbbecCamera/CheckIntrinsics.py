#!/usr/bin/env python3
"""
Check Orbbec Camera Intrinsics

This script loads the calibration results and visualizes the effect of
undistortion on captured images. It also computes reprojection error
to verify calibration quality.

Usage:
    python CheckIntrinsics.py --calibration ../output/orbbec_calibration.npz --images ./calib_intrinsics_images/
    python CheckIntrinsics.py --mode reproj  # Compute reprojection error only
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse


def load_calibration(calib_file):
    """Load camera calibration from file (supports .npz and .json formats)"""
    try:
        # Check file extension
        if calib_file.endswith('.json'):
            # Load JSON format
            import json
            with open(calib_file, 'r') as f:
                calib = json.load(f)

            # Check if it has camera_matrix directly or individual fx/fy/cx/cy fields
            if "camera_matrix" in calib:
                camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
                dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float64)
                reprojection_error = calib.get("reprojection_error", None)
                board_params = None
            else:
                # Handle Orbbec intrinsics format (nested structure)
                # Find the first camera key
                camera_key = next(iter(calib.keys()))
                cam_data = calib[camera_key]

                # Build camera matrix from fx, fy, cx, cy
                fx = cam_data["fx"]
                fy = cam_data["fy"]
                cx = cam_data["cx"]
                cy = cam_data["cy"]

                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.array(cam_data["distortion"], dtype=np.float64)
                reprojection_error = None
                board_params = None

        else:
            # Load NPZ format
            calib = np.load(calib_file)
            camera_matrix = calib["camera_matrix"]
            dist_coeffs = calib["dist_coeffs"]
            reprojection_error = calib.get("reprojection_error", None)

            # Try to load ChArUco board parameters for reprojection test
            board_params = None
            if all(k in calib for k in ["charuco_squares_x", "charuco_squares_y", "square_len_m", "marker_len_m"]):
                board_params = {
                    "squares_x": int(calib["charuco_squares_x"]),
                    "squares_y": int(calib["charuco_squares_y"]),
                    "square_len": float(calib["square_len_m"]),
                    "marker_len": float(calib["marker_len_m"])
                }

        print(f"✓ Loaded calibration from: {calib_file}")
        print(f"\nCamera Matrix:")
        print(camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(dist_coeffs.flatten())
        print(f"\nFocal Length (fx, fy): {camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f}")
        print(f"Principal Point (cx, cy): {camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f}")

        if reprojection_error is not None:
            print(f"Reprojection Error (from file): {reprojection_error:.4f} pixels")

        return camera_matrix, dist_coeffs, board_params

    except Exception as e:
        print(f"❌ Failed to load calibration: {e}")
        return None, None, None


def load_images(images_dir, max_images=50):
    """Load images from directory"""
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
    image_files.sort()

    # Limit number of images to avoid memory issues
    if len(image_files) > max_images:
        print(f"Found {len(image_files)} images, loading first {max_images}")
        image_files = image_files[:max_images]
    else:
        print(f"Found {len(image_files)} images")

    images = []
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            images.append(img)

    print(f"✓ Loaded {len(images)} images")
    return images, image_files


def show_undistortion_comparison(images, camera_matrix, dist_coeffs, image_id=0):
    """Show side-by-side comparison of raw and undistorted image"""
    if image_id >= len(images):
        print(f"⚠️  Image ID {image_id} out of range (0-{len(images)-1})")
        return

    frame = images[image_id]

    # Undistort image
    img_undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None)

    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_undist_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    # Create figure with two subplots
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title(f"Raw Image (ID: {image_id})", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_undist_rgb)
    plt.title(f"Undistorted Image (ID: {image_id})", fontsize=16)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_grid_comparison(images, camera_matrix, dist_coeffs, num_samples=6):
    """Show grid of multiple images comparing raw vs undistorted"""
    num_samples = min(num_samples, len(images))

    # Select evenly spaced images
    indices = np.linspace(0, len(images)-1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples*3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        frame = images[idx]
        img_undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_undist_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

        # Raw image
        axes[i, 0].imshow(frame_rgb)
        axes[i, 0].set_title(f"Raw (ID: {idx})")
        axes[i, 0].axis("off")

        # Undistorted image
        axes[i, 1].imshow(img_undist_rgb)
        axes[i, 1].set_title(f"Undistorted (ID: {idx})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def compute_reprojection_error(images, camera_matrix, dist_coeffs, board_params):
    """
    Compute reprojection error by detecting ChArUco board in images
    and reprojecting the 3D points back to image coordinates.
    """
    if board_params is None:
        print("⚠️  Board parameters not available in calibration file")
        return None

    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (board_params["squares_x"], board_params["squares_y"]),
        board_params["square_len"],
        board_params["marker_len"],
        aruco_dict
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_objpoints = []
    all_imgpoints = []
    rvecs_list = []
    tvecs_list = []
    valid_images = 0

    print("\n" + "="*60)
    print("Computing Reprojection Error")
    print("="*60)

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco corners
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) >= 4:
            # Get object points for detected corners
            obj_points = board.getChessboardCorners()[charuco_ids.flatten()]

            # Estimate pose
            ret, rvec, tvec = cv2.solvePnP(
                obj_points,
                charuco_corners,
                camera_matrix,
                dist_coeffs
            )

            if ret:
                all_objpoints.append(obj_points)
                all_imgpoints.append(charuco_corners)
                rvecs_list.append(rvec)
                tvecs_list.append(tvec)
                valid_images += 1

    if valid_images == 0:
        print("❌ No valid ChArUco boards detected in images")
        return None

    print(f"✓ Detected ChArUco board in {valid_images}/{len(images)} images")

    # Compute reprojection error
    mean_error = 0
    per_image_errors = []

    for i in range(len(all_objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            all_objpoints[i],
            rvecs_list[i],
            tvecs_list[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(all_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_image_errors.append(error)
        mean_error += error

    mean_error = mean_error / len(all_objpoints)

    print(f"\n📊 Reprojection Error Statistics:")
    print(f"  Mean Error: {mean_error:.4f} pixels")
    print(f"  Min Error:  {min(per_image_errors):.4f} pixels")
    print(f"  Max Error:  {max(per_image_errors):.4f} pixels")
    print(f"  Std Dev:    {np.std(per_image_errors):.4f} pixels")
    print("="*60)

    return {
        "mean_error": mean_error,
        "per_image_errors": per_image_errors,
        "min_error": min(per_image_errors),
        "max_error": max(per_image_errors),
        "std_error": np.std(per_image_errors),
        "num_valid_images": valid_images
    }


def show_distortion_difference(images, camera_matrix, dist_coeffs, image_id=0,
                               scale_factor=10.0):
    """Show the difference between raw and undistorted image (amplified)"""
    if image_id >= len(images):
        print(f"⚠️  Image ID {image_id} out of range (0-{len(images)-1})")
        return

    frame = images[image_id]
    img_undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None)

    # Calculate difference
    diff = cv2.absdiff(frame, img_undist)

    # Amplify difference for visualization
    diff_amplified = np.clip(diff.astype(float) * scale_factor, 0, 255).astype(np.uint8)

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_undist_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB)

    # Create figure with three subplots
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(frame_rgb)
    plt.title(f"Raw Image (ID: {image_id})", fontsize=14)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_undist_rgb)
    plt.title(f"Undistorted Image (ID: {image_id})", fontsize=14)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff_rgb)
    plt.title(f"Difference (×{scale_factor})", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def interactive_viewer(images, camera_matrix, dist_coeffs, board_params=None):
    """Interactive viewer to browse through images"""
    print("\n" + "="*60)
    print("Interactive Viewer")
    print("="*60)
    print("Commands:")
    print("  [number] - Show image by ID")
    print("  'n' - Next image")
    print("  'p' - Previous image")
    print("  'g' - Show grid of multiple images")
    print("  'd' - Show difference map")
    if board_params is not None:
        print("  'r' - Compute reprojection error")
    print("  'q' - Quit")
    print("="*60)

    current_id = 0

    while True:
        cmd = input(f"\nCurrent ID: {current_id} (0-{len(images)-1}) > ").strip().lower()

        if cmd == 'q':
            break
        elif cmd == 'n':
            current_id = min(current_id + 1, len(images) - 1)
            show_undistortion_comparison(images, camera_matrix, dist_coeffs, current_id)
        elif cmd == 'p':
            current_id = max(current_id - 1, 0)
            show_undistortion_comparison(images, camera_matrix, dist_coeffs, current_id)
        elif cmd == 'g':
            show_grid_comparison(images, camera_matrix, dist_coeffs)
        elif cmd == 'd':
            show_distortion_difference(images, camera_matrix, dist_coeffs, current_id)
        elif cmd == 'r':
            if board_params is not None:
                compute_reprojection_error(images, camera_matrix, dist_coeffs, board_params)
            else:
                print("⚠️  Reprojection test not available (board parameters missing)")
        elif cmd.isdigit():
            img_id = int(cmd)
            if 0 <= img_id < len(images):
                current_id = img_id
                show_undistortion_comparison(images, camera_matrix, dist_coeffs, current_id)
            else:
                print(f"⚠️  Invalid image ID. Valid range: 0-{len(images)-1}")
        elif cmd == '':
            show_undistortion_comparison(images, camera_matrix, dist_coeffs, current_id)
        else:
            print(f"⚠️  Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Check camera intrinsics by visualizing undistortion")
    parser.add_argument("--calibration", default="../output/orbbec_calibration_20251003_112009.npz",
                       help="Path to calibration file (default: ../output/orbbec_calibration.npz)")
    parser.add_argument("--images", default="./new_intriniscs_image/",
                       help="Directory containing images (default: ./new_intriniscs_image/)")
    parser.add_argument("--image-id", type=int, default=20,
                       help="Image ID to display (default: 20)")
    parser.add_argument("--mode", choices=["single", "grid", "diff", "reproj", "interactive"],
                       default="interactive",
                       help="Display mode (default: interactive)")
    parser.add_argument("--max-images", type=int, default=50,
                       help="Maximum number of images to load (default: 50)")

    args = parser.parse_args()

    print("🔍 Orbbec Camera Intrinsics Checker")
    print("="*60)

    # Load calibration
    camera_matrix, dist_coeffs, board_params = load_calibration(args.calibration)
    if camera_matrix is None:
        return

    print()

    # Load images
    images, image_files = load_images(args.images, args.max_images)
    if len(images) == 0:
        print("❌ No images found!")
        return

    print()

    # Display based on mode
    if args.mode == "single":
        print(f"Displaying single image (ID: {args.image_id})...")
        show_undistortion_comparison(images, camera_matrix, dist_coeffs, args.image_id)
    elif args.mode == "grid":
        print("Displaying grid comparison...")
        show_grid_comparison(images, camera_matrix, dist_coeffs)
    elif args.mode == "diff":
        print(f"Displaying difference map (ID: {args.image_id})...")
        show_distortion_difference(images, camera_matrix, dist_coeffs, args.image_id)
    elif args.mode == "reproj":
        print("Computing reprojection error...")
        compute_reprojection_error(images, camera_matrix, dist_coeffs, board_params)
    elif args.mode == "interactive":
        interactive_viewer(images, camera_matrix, dist_coeffs, board_params)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
