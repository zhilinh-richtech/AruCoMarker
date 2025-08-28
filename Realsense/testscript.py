import numpy as np

T = np.array([
    [0.51097,  -0.559244,  0.652806, -1.189456],
    [0.742469, -0.095573, -0.663027, -1.593206],
    [0.433185,  0.823476,  0.366386,  0.600184],
    [0.0,       0.0,       0.0,       1.0]
], dtype=np.float64)

# 180° rotation around Y axis
R_y180 = np.array([
    [-1.0, 0.0,  0.0 , 0],
    [ 0.0, 1.0,  0.0, 0],
    [ 0.0, 0.0, -1.0, 0],
    [0 , 0, 0, 1]
])

# Apply flip to rotation part
T_flipped = T.copy()
T_flipped[:3, :3] = T[:3, :3] @ R_y180  # local pitch flip

print(T_flipped)
