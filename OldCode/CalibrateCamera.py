import cv2
import numpy as np
import glob
import os

# Parameters
chessboard_size = (10, 7)  # number of inner corners (columns, rows)
square_size = 0.023  # size of one square in meters (adjust to match your printout)

# Prepare object points, e.g., (0,0,0), (0.025,0,0), ..., for all corners
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D real-world points
imgpoints = []  # 2D image plane points

# Load calibration images
images = glob.glob('calib_images/*.jpg')
if not images:
    print("❌ No images found in 'calib_images/'")
    exit()

# Detect corners in each image
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)

        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Show detected corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)
    else:
        print(f"⚠️ Chessboard not found in image: {fname}")

cv2.destroyAllWindows()

# Run calibration
img_size = gray.shape[::-1]  # width, height
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# Print results
print("\n✅ Calibration successful!")
print("\nCamera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

# Save to file
os.makedirs("output", exist_ok=True)
np.savez("output/calibration_data.npz",
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs)



print("\n📁 Calibration data saved to 'output/calibration_data.npz'")