#!/usr/bin/env python3
"""
Generate Orbbec camera intrinsics using captured ChArUco board images.

This script reads images captured from the eye-in-hand calibration and uses
OpenCV's camera calibration functions to compute camera intrinsics and
distortion coefficients.
"""

import cv2
import numpy as np
import os
import glob
from datetime import datetime
import argparse

# ChArUco board parameters (should match the ones used in EyeInHand.py)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M = 0.037        # square side length in meters
MARKER_LEN_M = SQUARE_LEN_M * 0.8       # marker side length in meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250  # ArUco dictionary

class OrbbecIntrinsicsGenerator:
    def __init__(self, images_dir="./calib_intrinsics_images", max_images=None, visualize_refinement=False):
        self.images_dir = images_dir
        self.max_images = max_images
        self.visualize_refinement = visualize_refinement
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_flags = 0
        self.reprojection_error = None
        self.image_width = None
        self.image_height = None

    @staticmethod
    def forward_distort_points(pts_ud_px, K, D):
        """
        Re-distort points from undistorted image space back to raw distorted image space.

        Args:
            pts_ud_px: Nx2 undistorted pixel coordinates
            K: 3x3 camera matrix used to undistort
            D: distortion coefficients (k1,k2,p1,p2,k3[,...]) used to undistort

        Returns:
            Nx2 distorted pixel coordinates (raw image space)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        k = np.zeros(8, dtype=np.float64)
        d = np.ravel(D).astype(np.float64)
        k[:len(d)] = d  # support 5, 8, etc. distortion coefficients

        # Undistorted pixel -> normalized camera coordinates
        x = (pts_ud_px[:, 0] - cx) / fx
        y = (pts_ud_px[:, 1] - cy) / fy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        k1, k2, p1, p2, k3, k4, k5, k6 = k

        # Radial distortion (Brown-Conrady model)
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

        # Apply distortion
        x_d = x * radial + x_tan
        y_d = y * radial + y_tan

        # Distorted normalized -> distorted pixel coordinates
        u_d = fx * x_d + cx
        v_d = fy * y_d + cy

        return np.column_stack([u_d, v_d]).astype(np.float32)

    def create_charuco_board(self):
        """Create the ChArUco board object"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            SQUARE_LEN_M, MARKER_LEN_M, aruco_dict
        )
        return board, aruco_dict

    def get_marker_corners_ideal(self, board, marker_id):
        """
        Get the ideal 3D positions of a marker's 4 corners on the board.
        Returns positions in board coordinate system (meters).
        """
        # Get marker object points (3D coordinates on board)
        marker_obj_points = board.getObjPoints()

        # marker_obj_points is a list where each entry corresponds to a marker
        # Each marker has 4 corners in order: top-left, top-right, bottom-right, bottom-left
        if marker_id < len(marker_obj_points):
            return marker_obj_points[marker_id]  # Returns 4x3 array of 3D points
        return None
    
    def load_images(self):
        """Load captured images"""
        image_files = glob.glob(os.path.join(self.images_dir, "*.jpg"))
        image_files.sort()  # Sort to ensure consistent order

        print(f"Found {len(image_files)} images in {self.images_dir}")

        # Limit number of images if specified
        if self.max_images is not None and self.max_images > 0:
            image_files = image_files[:self.max_images]
            print(f"Using first {len(image_files)} images")

        images = []
        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
                print(f"  Loaded: {os.path.basename(img_file)} ({img.shape})")
            else:
                print(f"  Failed to load: {img_file}")

        return images
    
    def detect_charuco_corners(self, images):
        """Detect ChArUco corners in all images following the classic Charuco pipeline."""
        board, aruco_dict = self.create_charuco_board()

        # Classic DetectorParameters for legacy-style API
        aruco_params = cv2.aruco.DetectorParameters()

        all_corners = []
        all_ids = []

        # Subpixel corner detection criteria optimized for ChArUco calibration
        # Window size (5,5): Balance between gradient info and single corner guarantee
        # Criteria: 200 max iterations, 0.00001 epsilon for high accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.00001)
        winSize = (5, 5)
        zeroZone = (-1, -1)

        print("\n🔍 Detecting ChArUco corners...")

        for i, img in enumerate(images):
            print(f"  Processing image {i+1}/{len(images)}...", end=" ")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers (classic API)
            corners, ids, _ = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=aruco_params)

            if ids is not None and len(ids) > 0:
                # Step 1: DO NOT refine ArUco marker corners before interpolation
                # Marker corner refinement can bias ChArUco corner interpolation due to
                # chessboard-square adjacency and image resampling effects

                # Step 2: Interpolate ChArUco corners directly from detected marker corners (classic API)
                response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board
                )

                # Step 3: Apply additional subpixel refinement to interpolated ChArUco corners
                # This gives maximum accuracy for the actual calibration points
                if response and response > 0 and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                    charuco_corners = cv2.cornerSubPix(gray, charuco_corners, winSize, zeroZone, criteria)
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    print(f"✓ Found {len(charuco_corners)} corners")
                elif charuco_corners is not None and len(charuco_corners) > 0:
                    print(f"⚠ Found {len(charuco_corners)} corners (too few, skipping)")
                    # Optional: Save visualization
                    vis_img = img.copy()
                    cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
                    cv2.putText(vis_img, f"Corners: {len(charuco_corners)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    vis_file = os.path.join(self.images_dir, f"vis_{i+1:03d}.jpg")
                    cv2.imwrite(vis_file, vis_img)
                else:
                    print("✗ No Charuco corners interpolated")
            else:
                print("✗ No markers found")

        print(f"\n✓ Successfully detected corners in {len(all_corners)} images")
        return all_corners, all_ids, board
    
    def calibrate_camera(self, all_corners, all_ids, board, images):
        """Perform camera calibration using ChArUco board (classic calibrateCameraCharuco)."""
        print("\n🎯 Performing camera calibration...")

        # Determine image size from first valid image
        for img in images:
            if img is not None:
                self.image_height, self.image_width = img.shape[:2]
                break
        if self.image_width is None or self.image_height is None:
            print("❌ Could not determine image size from inputs")
            return False
        imsize = (self.image_width, self.image_height)

        print(f"  Using {len(all_corners)} images for calibration")
        print(f"  Image resolution: {self.image_width}x{self.image_height}")

        try:
            # Initial calibration
            flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
            print("\n  Phase 1: Initial calibration...")
            ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=board,
                imageSize=imsize,
                cameraMatrix=None,
                distCoeffs=None,
                flags = flags
            )

            print(f"  Initial RMS error: {ret:.4f}")

            # Iterative refinement with canonical plane warping
            print("\n  Phase 2: Iterative refinement with canonical plane warping...")
            aruco_dict = board.getDictionary()

            # Subpixel criteria for refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.00001)

            for iteration in range(2):
                refined_corners = []
                refined_ids = []
                skipped_count = 0
                insufficient_corners_count = 0
                pixel_deltas = []

                for img_idx, img in enumerate(images):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # 1. Undistort image using current calibration
                    undistorted = cv2.undistort(gray, K, D)

                    # 2. Detect markers in undistorted image
                    aruco_params = cv2.aruco.DetectorParameters()
                    corners, ids, _ = cv2.aruco.detectMarkers(undistorted, aruco_dict, parameters=aruco_params)

                    if ids is None or len(ids) == 0:
                        continue

                    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners, markerIds=ids, image=undistorted, board=board
                    )

                    # Require at least 15 corners for stable calibration
                    if charuco_corners is not None and len(charuco_corners) >= 15:
                        # 3a. Refine charuco corners in undistorted space before computing homography
                        cv2.cornerSubPix(undistorted, charuco_corners, (5, 5), (-1, -1), criteria)

                        # 3b. Warp to canonical plane using global homography
                        # Note: For maximum accuracy with high distortion, local homographies per corner
                        # using nearest markers would be ideal, but global homography with RANSAC
                        # provides good practical results for most calibration scenarios

                        # High resolution canonical plane (200 pixels per square) for sharp edges
                        canonical_width = CHARUCO_SQUARES_X * 200
                        canonical_height = CHARUCO_SQUARES_Y * 200

                        # Destination "ideal" 2D points in canonical pixel coordinates
                        dst_points = []
                        for id in charuco_ids.flatten():
                            x, y, _ = board.getChessboardCorners()[id]
                            # Convert from meters to canonical pixels (200 pixels per square)
                            dst_points.append([x / SQUARE_LEN_M * 200, y / SQUARE_LEN_M * 200])
                        dst_points = np.array(dst_points, dtype=np.float32)

                        # Use RANSAC to reject outliers and stabilize homography
                        H, inlier_mask = cv2.findHomography(charuco_corners, dst_points, cv2.RANSAC, 1.0)

                        if H is None or inlier_mask is None or inlier_mask.sum() < 15:
                            skipped_count += 1
                            continue

                        # Use cubic interpolation for higher quality warp (keeps edges crisp)
                        canonical = cv2.warpPerspective(undistorted, H, (canonical_width, canonical_height),
                                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                        # Optional: Visualize undistorted and canonical warped images
                        if self.visualize_refinement:
                            # Create side-by-side comparison
                            h_undist, w_undist = undistorted.shape[:2]
                            h_canon, w_canon = canonical.shape[:2]

                            # Resize for display if needed
                            max_height = 800
                            if h_undist > max_height or h_canon > max_height:
                                scale_undist = max_height / h_undist
                                scale_canon = max_height / h_canon
                                undist_display = cv2.resize(undistorted, None, fx=scale_undist, fy=scale_undist)
                                canon_display = cv2.resize(canonical, None, fx=scale_canon, fy=scale_canon)
                            else:
                                undist_display = undistorted.copy()
                                canon_display = canonical.copy()

                            # Create side-by-side visualization
                            h_max = max(undist_display.shape[0], canon_display.shape[0])
                            combined = np.zeros((h_max, undist_display.shape[1] + canon_display.shape[1]), dtype=np.uint8)
                            combined[:undist_display.shape[0], :undist_display.shape[1]] = undist_display
                            combined[:canon_display.shape[0], undist_display.shape[1]:] = canon_display

                            # Add labels
                            combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                            cv2.putText(combined_color, "Undistorted", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(combined_color, "Canonical Fronto-Parallel", (undist_display.shape[1] + 10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                            warp_vis_file = os.path.join(self.images_dir, f"warp_iter{iteration+1}_img{img_idx+1:03d}.jpg")
                            cv2.imwrite(warp_vis_file, combined_color)

                        # 4. Redetect corners in canonical plane with subpixel refinement enabled
                        aruco_params_canonical = cv2.aruco.DetectorParameters()
                        aruco_params_canonical.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                        corners2, ids2, _ = cv2.aruco.detectMarkers(canonical, aruco_dict, parameters=aruco_params_canonical)

                        if ids2 is None or len(ids2) == 0:
                            continue

                        _, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=corners2, markerIds=ids2, image=canonical, board=board
                        )

                        # Need at least 6 corners for pose estimation in calibration
                        if charuco_corners2 is not None and len(charuco_corners2) >= 6:
                            # 5. Refine subpixel in canonical plane (more accurate due to frontal view)
                            cv2.cornerSubPix(canonical, charuco_corners2, (5, 5), (-1, -1), criteria)

                            # 6. Map refined corners back to undistorted image space
                            H_inv = np.linalg.inv(H)
                            pts_h = cv2.convertPointsToHomogeneous(charuco_corners2)[:, 0, :].T
                            pts_back_ud = (H_inv @ pts_h).T
                            pts_back_ud = pts_back_ud[:, :2] / pts_back_ud[:, 2, None]

                            # 7. Re-distort points back to raw image space (calibration expects distorted coords)
                            pts_back_distorted = self.forward_distort_points(pts_back_ud, K, D)

                            refined_corners_reshaped = pts_back_distorted.reshape(-1, 1, 2).astype(np.float32)
                            refined_corners.append(refined_corners_reshaped)
                            refined_ids.append(charuco_ids2)

                            # Calculate mean pixel delta for sanity check
                            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            corners_orig, ids_orig, _ = cv2.aruco.detectMarkers(gray_orig, aruco_dict)
                            if ids_orig is not None and len(ids_orig) > 0:
                                _, charuco_corners_orig, charuco_ids_orig = cv2.aruco.interpolateCornersCharuco(
                                    markerCorners=corners_orig, markerIds=ids_orig, image=gray_orig, board=board
                                )
                                if charuco_corners_orig is not None and len(charuco_corners_orig) == len(pts_back_distorted):
                                    delta = np.linalg.norm(charuco_corners_orig.reshape(-1, 2) - pts_back_distorted, axis=1)
                                    mean_delta = np.mean(delta)
                                    pixel_deltas.append(mean_delta)

                            # Optional: Visualize refined corners vs original detection
                            if self.visualize_refinement and charuco_corners_orig is not None:
                                vis_img = img.copy()
                                # Draw original detected corners in green
                                for pt in charuco_corners_orig.reshape(-1, 2):
                                    cv2.circle(vis_img, tuple(pt.astype(int)), 8, (0, 255, 0), 2)

                                # Draw refined re-distorted corners in red
                                for pt in pts_back_distorted:
                                    cv2.circle(vis_img, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

                                # Display mean pixel delta
                                if len(charuco_corners_orig) == len(pts_back_distorted):
                                    cv2.putText(vis_img, f"Mean delta: {mean_delta:.3f}px", (10, 30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                                vis_file = os.path.join(self.images_dir, f"refine_iter{iteration+1}_img{img_idx+1:03d}.jpg")
                                cv2.imwrite(vis_file, vis_img)
                        else:
                            insufficient_corners_count += 1
                    else:
                        insufficient_corners_count += 1

                # Recalibrate with refined corners - use intrinsic guess and early stop if no improvement
                if len(refined_corners) > 0:
                    total_corners = sum(len(c) for c in refined_corners)

                    # Store previous calibration for potential early stop
                    prev_ret, prevK, prevD = ret, K.copy(), D.copy()
                    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS
                    # Re-optimize with intrinsic guess (keeps solution close to previous, which we used for re-distortion)
                    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                        charucoCorners=refined_corners,
                        charucoIds=refined_ids,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=K,
                        distCoeffs=D,
                        flags=cv2.CALIB_USE_INTRINSIC_GUESS
                    )

                    # Calculate mean pixel delta across all images
                    mean_pixel_delta = np.mean(pixel_deltas) if pixel_deltas else 0.0

                    # Early stop if no improvement
                    if ret > prev_ret:
                        ret, K, D = prev_ret, prevK, prevD
                        print(f"  Iteration {iteration+1}: Early stop - no RMS improvement (was {prev_ret:.4f}, got {ret:.4f}), reverting to previous intrinsics")
                        break
                    else:
                        print(f"  Iteration {iteration+1}: RMS = {ret:.4f} | Refined: {len(refined_corners)} imgs, {total_corners} corners | Skipped: {skipped_count} | Mean Δ: {mean_pixel_delta:.3f}px")
                else:
                    print(f"  Iteration {iteration+1}: No corners refined, keeping previous calibration")

            self.camera_matrix = K
            self.dist_coeffs = D
            self.reprojection_error = ret

            print("✓ Camera calibration with iterative refinement successful!")
            print(f"  Final RMS error: {ret:.4f}")
            return True

        except Exception as e:
            print(f"❌ Calibration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_reprojection_error(self, all_corners, all_ids, board, images):
        """Calculate reprojection error to assess calibration quality"""
        print("\n📊 Calculating reprojection error...")

        total_error = 0
        total_points = 0
        errors = []

        for i, (corners, ids) in enumerate(zip(all_corners, all_ids)):
            if len(corners) > 0:
                # OFFICIAL PATH: Use matchImagePoints to ensure correct correspondence
                # This avoids subtle corner-indexing mismatches
                obj_pts, img_pts = board.matchImagePoints(corners, ids)

                if obj_pts is None or len(obj_pts) == 0:
                    print(f"  Image {i+1}: matchImagePoints failed")
                    continue

                # HIGH ACCURACY: Use IPPE + Levenberg-Marquardt refinement for planar targets
                # Step 1: IPPE method (best initial estimate for planar objects like ChArUco)
                ok_ippe, rvecs_ippe, tvecs_ippe, _ = cv2.solvePnPGeneric(
                    obj_pts, img_pts,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )

                if not ok_ippe or len(rvecs_ippe) == 0:
                    print(f"  Image {i+1}: IPPE failed")
                    continue

                # Select best IPPE solution (positive Z for board in front of camera)
                best_rvec_ippe = None
                best_tvec_ippe = None
                for j in range(len(rvecs_ippe)):
                    if tvecs_ippe[j][2] > 0:
                        best_rvec_ippe = rvecs_ippe[j]
                        best_tvec_ippe = tvecs_ippe[j]
                        break

                # Fallback to first solution if no positive Z found
                if best_rvec_ippe is None:
                    best_rvec_ippe = rvecs_ippe[0]
                    best_tvec_ippe = tvecs_ippe[0]

                # Step 2: Levenberg-Marquardt refinement (same as ArUco pose estimation)
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts,
                    self.camera_matrix, self.dist_coeffs,
                    rvec=best_rvec_ippe,
                    tvec=best_tvec_ippe,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE  # Uses Levenberg-Marquardt internally
                )

                if not success:
                    print(f"  Image {i+1}: LM refinement failed")
                    continue

                # Project 3D points using refined pose
                projected_points, _ = cv2.projectPoints(
                    obj_pts, rvec, tvec,
                    self.camera_matrix, self.dist_coeffs
                )

                # Calculate per-image error using OpenCV's standard method
                # error = cv.norm(imgpoints, imgpoints2, cv.NORM_L2) / len(imgpoints2)
                error = cv2.norm(img_pts, projected_points, cv2.NORM_L2) / len(img_pts)

                total_error += error
                total_points += 1  # Count images, not individual points
                errors.append(error)

                print(f"  Image {i+1}: {error:.4f} pixels per point (IPPE+LM)")

        mean_error = total_error / total_points if total_points > 0 else 0

        # Add statistics
        if errors:
            errors = np.array(errors)
            print(f"✓ Mean reprojection error: {mean_error:.4f} pixels per point")
            print(f"  Max error: {np.max(errors):.4f} pixels")
            print(f"  Min error: {np.min(errors):.4f} pixels")
            print(f"  Std dev: {np.std(errors):.4f} pixels")

            # Store mean error for saving (following OpenCV's convention)
            self.reprojection_error = mean_error
        else:
            self.reprojection_error = float('inf')

        return mean_error
    
    def save_results(self):
        """Save calibration results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as NPZ file (compatible with existing code)
        npz_file = f"../output/orbbec_calibration_{timestamp}.npz"
        np.savez(npz_file,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 calibration_flags=self.calibration_flags,
                 reprojection_error=self.reprojection_error,
                 timestamp=timestamp,
                 charuco_squares_x=CHARUCO_SQUARES_X,
                 charuco_squares_y=CHARUCO_SQUARES_Y,
                 square_len_m=SQUARE_LEN_M,
                 marker_len_m=MARKER_LEN_M)
        
        print(f"✓ Saved calibration to: {npz_file}")
        
        # Also save as the standard filename for compatibility
        standard_file = "../output/orbbec_calibration.npz"
        np.savez(standard_file,
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 calibration_flags=self.calibration_flags,
                 reprojection_error=self.reprojection_error,
                 timestamp=timestamp,
                 charuco_squares_x=CHARUCO_SQUARES_X,
                 charuco_squares_y=CHARUCO_SQUARES_Y,
                 square_len_m=SQUARE_LEN_M,
                 marker_len_m=MARKER_LEN_M)
        
        print(f"✓ Saved calibration to: {standard_file}")
        
        # Save as JSON for human readability
        json_file = f"../output/orbbec_calibration_{timestamp}.json"
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reprojection_error": float(self.reprojection_error) if self.reprojection_error is not None else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": float(self.camera_matrix[0, 0]),
            "fy": float(self.camera_matrix[1, 1]),
            "cx": float(self.camera_matrix[0, 2]),
            "cy": float(self.camera_matrix[1, 2]),
            "timestamp": timestamp,
            "charuco_board": {
                "squares_x": CHARUCO_SQUARES_X,
                "squares_y": CHARUCO_SQUARES_Y,
                "square_len_m": SQUARE_LEN_M,
                "marker_len_m": MARKER_LEN_M
            }
        }
        
        import json
        with open(json_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"✓ Saved calibration to: {json_file}")
    
    def print_results(self):
        """Print calibration results"""
        print("\n" + "="*50)
        print("📊 CALIBRATION RESULTS")
        print("="*50)
        
        print(f"Camera Matrix:")
        print(self.camera_matrix)
        print(f"\nFocal Length (fx, fy): {self.camera_matrix[0,0]:.2f}, {self.camera_matrix[1,1]:.2f}")
        print(f"Principal Point (cx, cy): {self.camera_matrix[0,2]:.2f}, {self.camera_matrix[1,2]:.2f}")
        
        print(f"\nDistortion Coefficients:")
        print(self.dist_coeffs)

        print(f"\nImage Resolution: {self.image_width}x{self.image_height}")
        print(f"Calibration Flags: {self.calibration_flags}")
    
    def run(self):
        """Main calibration process"""
        print("🎥 Orbbec Camera Intrinsics Generation")
        print("="*50)
        
        # Load images
        images = self.load_images()
        if len(images) == 0:
            print("❌ No images found!")
            return False
        
        # Detect ChArUco corners
        all_corners, all_ids, board = self.detect_charuco_corners(images)
        if len(all_corners) < 10:
            print(f"❌ Not enough images with detected corners ({len(all_corners)}). Need at least 10.")
            return False
        
        # Perform calibration
        if not self.calibrate_camera(all_corners, all_ids, board, images):
            return False
        
        # Calculate reprojection error
        reproj_error = self.calculate_reprojection_error(all_corners, all_ids, board, images)
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        print(f"\n✅ Calibration complete! Reprojection error: {reproj_error:.4f} pixels per point")

        if reproj_error < 0.5:
            print("🎉 Excellent calibration quality!")
        elif reproj_error < 1.0:
            print("✅ Good calibration quality")
        else:
            print("⚠️  Calibration quality could be improved")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Generate Orbbec camera intrinsics from captured images")
    parser.add_argument("--images-dir", default="./new_intriniscs_image/",
                        help="Directory containing captured images")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to use (uses first N images in sorted order)")
    parser.add_argument("--visualize-refinement", action="store_true",
                        help="Save visualization images comparing original and refined corners")
    args = parser.parse_args()

    generator = OrbbecIntrinsicsGenerator(args.images_dir, max_images=args.max_images,
                                         visualize_refinement=args.visualize_refinement)
    success = generator.run()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Use the generated orbbec_calibration.npz file in your applications")
        print("2. The calibration is compatible with the existing EyeInHand.py script")
        print("3. For best results, ensure good lighting and clear ChArUco board visibility")
    else:
        print("\n❌ Calibration failed. Please check your images and try again.")

if __name__ == "__main__":
    main()
