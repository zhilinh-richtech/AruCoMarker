#!/usr/bin/env python3
"""
Create test extrinsics matrix as .npy file
"""
import numpy as np

# Test extrinsics 4x4 camera-to-gripper transformation matrix (translations in mm)
test_T_cam2gripper_mm = np.array([[-0.02829711 ,-0.99954692 , 0.01025778 , 0.07082251],
 [ 0.99923424 ,-0.02856269, -0.0267413,   0.00216768],
 [ 0.02702217,  0.00949323 , 0.99958976 ,-0.14288646],
 [ 0.,          0.,          0.,          1.,        ]]
)

# Convert translation components from mm to meters
test_T_cam2gripper = test_T_cam2gripper_mm.copy()
test_T_cam2gripper[:3, 3] = test_T_cam2gripper_mm[:3, 3]

# Save to .npy file
np.save('test_extrinsics.npy', test_T_cam2gripper)
print("✓ Saved test extrinsics to test_extrinsics.npy")
print(f"Matrix shape: {test_T_cam2gripper.shape}")
print(f"Matrix (in meters):\n{test_T_cam2gripper}")
