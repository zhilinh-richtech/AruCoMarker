import numpy as np


'''
EXAMPLE INPUT
# P = predicted points from camera transformed to base frame (Nx3)
P = np.array([
    [48.99, 527.71, -1.84],   # corner 1
    [49.01, 490.54, -2.24],   # corner 2
    [49.03, 453.36, -2.63],   # corner 3
    [49.05, 416.18, -3.03],   # corner 4
    [11.82, 527.70, -2.17],   # corner 5
    [11.83, 490.52, -2.57],   # corner 6
    [11.85, 453.34, -2.96],   # corner 7
    [11.87, 416.17, -3.36],   # corner 8
    [-25.36, 527.69, -2.50],  # corner 9
    [-25.35, 490.51, -2.90],  # corner 10
    [-25.33, 453.33, -3.29],  # corner 11
    [-25.31, 416.15, -3.69],  # corner 12
    [-62.54, 527.67, -2.83],  # corner 13
    [-62.52, 490.49, -3.23],  # corner 14
    [-62.51, 453.32, -3.62],  # corner 15
    [-62.49, 416.14, -4.02],  # corner 16
    [-99.72, 527.66, -3.16],  # corner 17
    [-99.70, 490.48, -3.56],  # corner 18
    [-99.68, 453.30, -3.95],  # corner 19
    [-99.67, 416.12, -4.35],  # corner 20
    [-136.90, 527.64, -3.49], # corner 21
    [-136.88, 490.47, -3.89], # corner 22
    [-136.86, 453.29, -4.28], # corner 23
    [-136.85, 416.11, -4.68]  # corner 24
])

# Corresponding ground-truth points in base frame
Q = np.array([
    [49.3,526,-1.4],  # corner 1
    [48.8, 489.3, -2.5],   # corner 2
    [47.5, 452.3, -3.2],   # corner 3
    [47, 415.9, -3.8],   # corner 4
    [11.1, 527.40, -2.2],   # corner 5
    [11.3,490.2,-2.6],   # corner 6
    [10.9,453,-3],   # corner 7
    [9.3,416.8,-4.1],   # corner 8
    [-25.6,527.3,-2.6],  # corner 9
    [-25.8,491.1,-3],  # corner 10
    [-26.1,453.6,-3.4],  # corner 11
    [-27.2,416.6,-4] #corner 12
    [-62.5,528.7,-2.4],  # corner 13
    [-62.8,491.8,-2.9] #corner 14
    [-63.2, 454.6, -3.6] #corner 15
    [-63.9, 418, -4.2] #corner 16
    [-99.1, 529.8, -2.5] #corner 17
    [-99.8, 492.8, -3.2] #corner 18
    [-100.6, 456.4, -4.3] #corner 19
    [-101.4, 418.3, -4.5] #corner 20
    [-136.8, 530.6, -2.5] #corner 21
    [-137.2, 494.1, -3.2] #corner 22
    [-137.7, 456.5, -3.8] #corner 23
    [-138, 420.1, -4.5] #corner 24

])
'''


def to_homogeneous(R, t):
    """Convert rotation matrix R and translation vector t to 4x4 homogeneous matrix"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t).flatten()
    return T


def from_homogeneous(T):
    """Extract rotation matrix R and translation vector t from 4x4 homogeneous matrix"""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def compute_icp_transformation(P, Q):
    """
    Compute the transformation matrix deltaX using Iterative Closest Point (ICP) algorithm.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Nx3 array of predicted xyz points
    Q : numpy.ndarray
        Nx3 array of matching known xyz points
        
    Returns:
    --------
    deltaX : numpy.ndarray
        4x4 transformation matrix [R|t; 0 0 0 1]
    Rdelta : numpy.ndarray
        3x3 rotation matrix
    tdelta : numpy.ndarray
        3x1 translation vector
    """
    # Convert to numpy arrays if not already
    P = np.array(P)
    Q = np.array(Q)
    
    # Ensure P and Q are Nx3
    assert P.shape == Q.shape, "P and Q must have the same shape"
    assert P.shape[1] == 3, "Points must be 3D (Nx3)"
    
    N = P.shape[0]
    
    # Step 1: Compute centroids (Pbar, Qbar)
    Pbar = (1/N) * np.sum(P, axis=0)  # Shape: (3,)
    Qbar = (1/N) * np.sum(Q, axis=0)  # Shape: (3,)
    
    # Step 2: Compute centered points (pi' and qi')
    P_centered = P - Pbar  # Shape: (N, 3)
    Q_centered = Q - Qbar  # Shape: (N, 3)
    
    # Step 3: Compute H matrix (3x3)
    # H = sum(pi' * qi'^T) for i=1 to N
    H = np.zeros((3, 3))
    for i in range(N):
        pi_prime = P_centered[i, :].reshape(3, 1)  # Column vector (3, 1)
        qi_prime = Q_centered[i, :].reshape(3, 1)  # Column vector (3, 1)
        H += pi_prime @ qi_prime.T  # Outer product
    
    # Alternative vectorized computation:
    # H = P_centered.T @ Q_centered
    
    # Step 4: Perform SVD on H
    # H = U * Σ * V^T
    U, Sigma, VT = np.linalg.svd(H)
    V = VT.T  # V is the transpose of VT
    
    # Step 5: Compute rotation matrix Rdelta
    Rdelta = V @ U.T
    
    # Check for reflection (determinant should be +1 for proper rotation)
    if np.linalg.det(Rdelta) < 0:
        # Correct for reflection by flipping the sign of the last column of V
        V[:, -1] *= -1
        Rdelta = V @ U.T
    
    # Step 6: Compute translation vector tdelta
    tdelta = Qbar - Rdelta @ Pbar  # Shape: (3,)
    
    # Step 7: Form the 4x4 transformation matrix deltaX
    deltaX = np.eye(4)
    deltaX[:3, :3] = Rdelta
    deltaX[:3, 3] = tdelta
    
    return deltaX, Rdelta, tdelta


def apply_transformation(X0, deltaX):
    """
    Apply transformation deltaX to X0 to get X1.
    X0 = np.array([0.01714663619779156, -0.9998386143440384, -0.005355757874287381, 72.53234054573493],
                  [0.999798127854351, 0.01720156004482209, -0.010383153228154418, 0.05973784764155832],
                  [0.01047360520716499, -0.005176640544964832, 0.9999317506643304, -141.86480233899346],
                  [0.0, 0.0, 0.0, 1.0])
    Parameters:
    -----------
    X0 : numpy.ndarray
        4x4 initial transformation matrix (e.g., Tgrip_cam)
    deltaX : numpy.ndarray
        4x4 transformation matrix from ICP
        
    Returns:
    --------
    X1 : numpy.ndarray
        4x4 resulting transformation matrix (X1 = deltaX * X0)
    """
    X0 = np.array(X0)
    deltaX = np.array(deltaX)
    
    X1 = deltaX @ X0
    return X1


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Roll-Pitch-Yaw angles to rotation matrix.
    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    
    Parameters:
    -----------
    roll, pitch, yaw : float
        Angles in degrees
        
    Returns:
    --------
    R : numpy.ndarray
        3x3 rotation matrix
    """
    # Convert to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # Rotation around X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation around Y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def compute_corrected_extrinsics(X0, deltaX, T_base_gripper):
    """
    Properly compute the corrected camera-to-gripper extrinsics X1 when P was
    computed through the full robot chain: P = T_base_gripper @ X0 @ P_camera
    
    ICP found: Q ≈ deltaX @ P = deltaX @ T_base_gripper @ X0 @ P_camera
    We want: Q ≈ T_base_gripper @ X1 @ P_camera
    
    Therefore: X1 = T_base_gripper^(-1) @ deltaX @ T_base_gripper @ X0
    
    Parameters:
    -----------
    X0 : numpy.ndarray
        4x4 original Tgrip_cam transformation (in same units as P/Q)
    deltaX : numpy.ndarray
        4x4 ICP correction in base frame
    T_base_gripper : numpy.ndarray
        4x4 robot forward kinematics (base to gripper, in same units)
        
    Returns:
    --------
    X1_corrected : numpy.ndarray
        4x4 properly corrected Tgrip_cam transformation
    """
    X0 = np.array(X0)
    deltaX = np.array(deltaX)
    T_base_gripper = np.array(T_base_gripper)
    
    # Compute inverse of T_base_gripper
    T_gripper_base = np.linalg.inv(T_base_gripper)
    
    # Proper correction: X1 = T_gripper_base @ deltaX @ T_base_gripper @ X0
    X1_corrected = T_gripper_base @ deltaX @ T_base_gripper @ X0
    
    return X1_corrected


def icp_iteration(P, Q, X0=None, max_iterations=50, tolerance=1e-6):
    """
    Perform full ICP with iterations (useful if point correspondences need to be updated).
    
    Parameters:
    -----------
    P : numpy.ndarray
        Nx3 array of predicted xyz points
    Q : numpy.ndarray
        Nx3 array of matching known xyz points
    X0 : numpy.ndarray, optional
        4x4 initial transformation matrix
    max_iterations : int
        Maximum number of ICP iterations
    tolerance : float
        Convergence tolerance for transformation change
        
    Returns:
    --------
    deltaX : numpy.ndarray
        4x4 final transformation matrix
    X1 : numpy.ndarray or None
        Final transformation (if X0 provided)
    """
    # Compute the transformation for the given correspondences
    deltaX, Rdelta, tdelta = compute_icp_transformation(P, Q)
    
    # If X0 is provided, compute X1
    X1 = None
    if X0 is not None:
        X1 = apply_transformation(X0, deltaX)
    
    return deltaX, X1


def analyze_point_quality(P, Q):
    """
    Analyze the quality and diversity of point correspondences.
    """
    print("\n" + "=" * 70)
    print("POINT QUALITY ANALYSIS")
    print("=" * 70)
    
    # Check point differences
    diff = Q - P
    print(f"\nPoint-wise differences (Q - P):")
    print(f"{'Point':<8} {'ΔX':<12} {'ΔY':<12} {'ΔZ':<12} {'Distance':<12}")
    print("-" * 70)
    for i in range(len(P)):
        dist = np.linalg.norm(diff[i])
        print(f"{i+1:<8} {diff[i,0]:>11.2f} {diff[i,1]:>11.2f} {diff[i,2]:>11.2f} {dist:>11.2f}")
    
    print(f"\nMean difference: {np.mean(diff, axis=0)}")
    print(f"Std deviation: {np.std(diff, axis=0)}")
    
    # Check if points are coplanar
    if len(P) >= 4:
        # Fit plane to P points
        P_centered = P - np.mean(P, axis=0)
        _, _, Vt = np.linalg.svd(P_centered)
        normal = Vt[-1, :]  # Normal to best-fit plane
        
        # Calculate distance of each point from plane
        distances = np.abs(P_centered @ normal)
        planarity = np.max(distances)
        
        print(f"\n⚠️  Planarity check:")
        print(f"   Max deviation from best-fit plane: {planarity:.2f} mm")
        if planarity < 50:
            print(f"   WARNING: Points are nearly coplanar! This gives poor constraints.")
            print(f"   Recommendation: Add points at different depths (±100mm range)")
    
    # Check spatial spread
    P_range = np.ptp(P, axis=0)
    Q_range = np.ptp(Q, axis=0)
    print(f"\nSpatial spread:")
    print(f"   P range: X={P_range[0]:.1f}mm, Y={P_range[1]:.1f}mm, Z={P_range[2]:.1f}mm")
    print(f"   Q range: X={Q_range[0]:.1f}mm, Y={Q_range[1]:.1f}mm, Z={Q_range[2]:.1f}mm")


if __name__ == "__main__":
    # User's actual data
    print("=" * 70)
    print("ICP CALIBRATION RESULTS")
    print("=" * 70)
    
    # P = predicted points from camera transformed to base frame
    P = np.array([
        [48.99, 527.71, -1.84],   # corner 1
        [49.01, 490.54, -2.24],   # corner 2
        [49.03, 453.36, -2.63],   # corner 3
        [49.05, 416.18, -3.03],   # corner 4
        [11.82, 527.70, -2.17],   # corner 5
        [11.83, 490.52, -2.57],   # corner 6
        [11.85, 453.34, -2.96],   # corner 7
        [11.87, 416.17, -3.36],   # corner 8
        [-25.36, 527.69, -2.50],  # corner 9
        [-25.35, 490.51, -2.90],  # corner 10
        [-25.33, 453.33, -3.29],  # corner 11
        [-25.31, 416.15, -3.69],  # corner 12
        [-62.54, 527.67, -2.83],  # corner 13
        [-62.52, 490.49, -3.23],  # corner 14
        [-62.51, 453.32, -3.62],  # corner 15
        [-62.49, 416.14, -4.02],  # corner 16
        [-99.72, 527.66, -3.16],  # corner 17
        [-99.70, 490.48, -3.56],  # corner 18
        [-99.68, 453.30, -3.95],  # corner 19
        [-99.67, 416.12, -4.35],  # corner 20
        [-136.90, 527.64, -3.49], # corner 21
        [-136.88, 490.47, -3.89], # corner 22
        [-136.86, 453.29, -4.28], # corner 23
        [-136.85, 416.11, -4.68]  # corner 24
    ])
    
    # Q = Corresponding ground-truth points in base frame
    Q = np.array([
        [49.3, 526, -1.4],          # corner 1
        [48.8, 489.3, -2.5],        # corner 2
        [47.5, 452.3, -3.2],        # corner 3
        [47, 415.9, -3.8],          # corner 4
        [11.1, 527.40, -2.2],       # corner 5
        [11.3, 490.2, -2.6],        # corner 6
        [10.9, 453, -3],            # corner 7
        [9.3, 416.8, -4.1],         # corner 8
        [-25.6, 527.3, -2.6],       # corner 9
        [-25.8, 491.1, -3],         # corner 10
        [-26.1, 453.6, -3.4],       # corner 11
        [-27.2, 416.6, -4],         # corner 12
        [-62.5, 528.7, -2.4],       # corner 13
        [-62.8, 491.8, -2.9],       # corner 14
        [-63.2, 454.6, -3.6],       # corner 15
        [-63.9, 418, -4.2],         # corner 16
        [-99.1, 529.8, -2.5],       # corner 17
        [-99.8, 492.8, -3.2],       # corner 18
        [-100.6, 456.4, -4.3],      # corner 19
        [-101.4, 418.3, -4.5],      # corner 20
        [-136.8, 530.6, -2.5],      # corner 21
        [-137.2, 494.1, -3.2],      # corner 22
        [-137.7, 456.5, -3.8],      # corner 23
        [-138, 420.1, -4.5]         # corner 24
    ])
    
    # X0 = Original Tgrip_cam transformation (in mm)
    X0 = np.array([
        [0.01714663619779156, -0.9998386143440384, -0.005355757874287381, 72.53234054573493],
        [0.999798127854351, 0.01720156004482209, -0.010383153228154418, 0.05973784764155832],
        [0.01047360520716499, -0.005176640544964832, 0.9999317506643304, -141.86480233899346],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # T_base_gripper = Robot pose when P was collected
    # Translation: (-52.5, 386.1, 154.5) mm, RPY: (180, 0, 90) degrees
    R_base_gripper = rpy_to_rotation_matrix(180, 0, 90)
    T_base_gripper = np.eye(4)
    T_base_gripper[:3, :3] = R_base_gripper
    T_base_gripper[:3, 3] = [-52.5, 386.1, 154.5]
    
    # First, analyze the point quality
    analyze_point_quality(P, Q)
    
    print("\nInput Data:")
    print("-" * 70)
    print(f"P (Predicted points):\n{P}\n")
    print(f"Q (Ground truth points):\n{Q}\n")
    print(f"X0 (Original Tgrip_cam):\n{X0}\n")
    
    # Compute deltaX
    deltaX, Rdelta, tdelta = compute_icp_transformation(P, Q)
    
    print("\nComputed Transformation:")
    print("-" * 70)
    print(f"deltaX (4x4 transformation matrix):\n{deltaX}\n")
    print(f"Rdelta (3x3 rotation matrix):\n{Rdelta}\n")
    print(f"tdelta (translation vector): {tdelta}\n")
    
    # Compute X1_naive = deltaX * X0 (WRONG when P includes robot transform!)
    X1_naive = apply_transformation(X0, deltaX)
    
    # Compute X1_corrected (CORRECT method accounting for robot kinematics)
    X1_corrected = compute_corrected_extrinsics(X0, deltaX, T_base_gripper)
    
    print("\nT_base_gripper (Robot pose when P was collected):")
    print("-" * 70)
    print(f"{T_base_gripper}\n")
    
    print("\n❌ X1_naive = deltaX * X0 (WRONG - ignores robot kinematics):")
    print("-" * 70)
    print(f"{X1_naive}\n")
    
    print("\n✅ X1_corrected (CORRECT - accounts for robot transform):")
    print("-" * 70)
    print(f"{X1_corrected}\n")
    print("Use X1_corrected for your robot!\n")
    
    # Verification: Transform P using deltaX and compare to Q
    P_homogeneous = np.hstack([P, np.ones((P.shape[0], 1))])
    P_transformed = (deltaX @ P_homogeneous.T).T[:, :3]
    
    print("\nVerification:")
    print("-" * 70)
    print(f"P transformed by deltaX:\n{P_transformed}\n")
    print(f"Q (target):\n{Q}\n")
    
    # Calculate errors
    errors = np.linalg.norm(P_transformed - Q, axis=1)
    print(f"Per-point errors (mm): {errors}")
    print(f"Mean error: {np.mean(errors):.4f} mm")
    print(f"Max error: {np.max(errors):.4f} mm")
    print(f"RMS error: {np.sqrt(np.mean(errors**2)):.4f} mm")
    
    # Convert X1_corrected to meters for robot (if needed)
    X1_corrected_meters = X1_corrected.copy()
    X1_corrected_meters[:3, 3] /= 1000  # Convert mm to meters
    
    print("\n" + "=" * 70)
    print("FINAL RESULT FOR YOUR ROBOT")
    print("=" * 70)
    print(f"X1_corrected in MILLIMETERS:\n{X1_corrected}\n")
    print(f"X1_corrected in METERS (if your robot uses meters):\n{X1_corrected_meters}\n")
    print(f"Translation change from X0 (mm): {X1_corrected[:3, 3] - X0[:3, 3]}\n")
    
    print("=" * 70)
    print("⚠️  NOTES:")
    print(f"   - Used {len(P)} calibration points")
    print("   - Mean residual error: {:.2f} mm".format(np.mean(errors)))
    print("   - This correction accounts for robot kinematics!")
    if np.max(np.abs(P[:, 2] - np.mean(P[:, 2]))) < 50:
        print("   - ⚠️  Points are coplanar - add depth diversity for robustness")
    print("=" * 70)

