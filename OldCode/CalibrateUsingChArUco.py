import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os

# ---------- Charuco board parameters ----------
squares_x = 7           # Number of chessboard squares in X direction
squares_y = 9              # Number of chessboard squares in Y direction
square_length = 0.029     # Square size (meters)
marker_length = 0.023      # Marker size inside each square (meters)

charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, charuco_dict)
# board.setLegacyPattern(True)
# ---------- Calibration images ----------
images = glob.glob('ArUcoBoardcalib_images/*.jpg')
if not images:
    print("❌ No images found in 'ArUcoBoardcalib_images/'")
    exit()

all_corners = []
all_ids = []
image_size = None

# Create output folder
os.makedirs("output", exist_ok=True)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, charuco_dict)

    img_copy = img.copy()

    if ids is not None and len(ids) > 0:
        # Draw detected markers
        aruco.drawDetectedMarkers(img_copy, corners, ids)

        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

        print(f"{os.path.basename(fname)}: markers = {len(ids)}, charuco corners = {retval}")

        if retval and retval > 0:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

            # Draw Charuco corners
            aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids)

            # Save annotated image
            output_path = os.path.join("output", f"charuco_ok_{os.path.basename(fname)}")
            cv2.imwrite(output_path, img_copy)
        else:
            print(f"⚠️ Not enough Charuco corners in {fname} (retval={retval})")
            # Save image showing only markers
            output_path = os.path.join("output", f"charuco_fail_{os.path.basename(fname)}")
            cv2.imwrite(output_path, img_copy)
    else:
        print(f"⚠️ No markers detected in {fname}")
        output_path = os.path.join("output", f"no_markers_{os.path.basename(fname)}")
        cv2.imwrite(output_path, img_copy)

    if image_size is None:
        image_size = gray.shape[::-1]

# Destroy windows
cv2.destroyAllWindows()

# ---------- Run calibration if enough valid images ----------
if len(all_corners) < 5:
    print("❌ Not enough valid images for calibration. Need at least 5.")
else:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None
    )

    print("\n✅ Calibration successful!")
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    print("\n rvecs:")
    print(rvecs)
    print("\n tvecs:")
    print(tvecs)
    # Correct reprojection error for ChArUco
    total_error = 0
    total_points = 0

    for i in range(len(all_corners)):
        imgpoints2, _ = cv2.projectPoints(
            board.getChessboardCorners()[all_ids[i].flatten()],  # select only detected corners
            rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(all_corners[i], imgpoints2, cv2.NORM_L2)
        total_error += error**2
        total_points += len(all_corners[i])

    mean_error = np.sqrt(total_error / total_points)
    print("Mean reprojection error:", mean_error)
    # Save calibration data
    np.savez("output/charuco_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)

    print("\n📁 Calibration data saved to 'output/charuco_calibration.npz'")
