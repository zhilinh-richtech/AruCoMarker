import pyrealsense2 as rs
import numpy as np

# Start pipeline
pipeline = rs.pipeline()
config = rs.config()
# ⚠️ Use the same resolution you will run detection at!
config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 1280, 800, rs.format.z16, 30)
pipeline.start(config)

# Get active profiles
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color)

# Extract color intrinsics
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# Build camera matrix
K = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])

# Distortion coefficients (numpy array)
dist = np.array(color_intrinsics.coeffs)

# Save intrinsics to .npz
np.savez("../output/realsense_calibration.npz",
         camera_matrix=K,
         dist_coeffs=dist)

print("Saved color intrinsics to output/realsense_calibration.npz")
print("K =\n", K)
print("dist =", dist)

pipeline.stop()
