import numpy as np
import cv2

def flip(T: np.ndarray, direction: str, degree: float) -> np.ndarray:
    """
    Apply a rotation (flip) to a 4x4 homogeneous transformation matrix in WORLD frame.
    
    Parameters:
        T (np.ndarray): 4x4 homogeneous transform
        direction (str): 'roll' (X), 'pitch' (Y), or 'yaw' (Z)
        degree (float): rotation angle in degrees (e.g. 180)
    
    Returns:
        np.ndarray: new 4x4 homogeneous transform
    """
    # Convert degrees to radians
    theta = np.radians(degree)

    if direction.lower() == "roll":  # X axis
        R = np.array([
            [1, 0,           0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    elif direction.lower() == "pitch":  # Y axis
        R = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0           ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif direction.lower() == "yaw":  # Z axis
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
    else:
        raise ValueError("direction must be 'roll', 'pitch', or 'yaw'")

    # Build 4x4 rotation
    R4 = np.eye(4)
    R4[:3, :3] = R

    # Pre-multiply to rotate in WORLD frame
    return R4 @ T


def draw_axes_custom(img, K, dist, R_cm, tvec, axis_len=0.08):
    """
    Draw custom axes at the marker origin using existing pose (Camera ← Marker):
      red (x):   right  = +X_marker
      green (y): up     = -Y_marker
      blue (z):  out    = +Z_marker
    """
    # Directions in the marker frame
    dir_x_m = np.array([ 1.0,  0.0,  0.0], dtype=np.float64)  # right
    dir_y_m = np.array([ 0.0, 1.0,  0.0], dtype=np.float64)  # up (note: Y)
    dir_z_m = np.array([ 0.0,  0.0,  1.0], dtype=np.float64)  # out of marker plane

    # Convert to camera frame
    o_cam = tvec.reshape(3)
    x_cam = o_cam + R_cm @ (dir_x_m * axis_len)
    y_cam = o_cam + R_cm @ (dir_y_m * axis_len)
    z_cam = o_cam + R_cm @ (dir_z_m * axis_len)

    # Project (use identity extrinsics since points are already in camera frame)
    pts_cam = np.stack([o_cam, x_cam, y_cam, z_cam], axis=0).reshape(-1, 1, 3)
    rvec0 = np.zeros(3, dtype=np.float64)
    tvec0 = np.zeros(3, dtype=np.float64)
    pts_px, _ = cv2.projectPoints(pts_cam, rvec0, tvec0, K, dist)
    O, Xp, Yp, Zp = pts_px.reshape(-1, 2).astype(int)

    # Draw: red (x), green (y), blue (z) in BGR
    cv2.line(img, tuple(O), tuple(Xp), (0,   0, 255), 2)  # red
    cv2.line(img, tuple(O), tuple(Yp), (0, 255,   0), 2)  # green
    cv2.line(img, tuple(O), tuple(Zp), (255, 0,   0), 2)  # blue

    # optional end dots
    cv2.circle(img, tuple(Xp), 3, (0, 0, 255), -1)
    cv2.circle(img, tuple(Yp), 3, (0, 255, 0), -1)
    cv2.circle(img, tuple(Zp), 3, (255, 0, 0), -1)


def to_homogeneous(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.reshape(3)
    return T

def invert_se3(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti

def matrix_to_rpy(R):
    # Accept 3x3 rotation matrix or 4x4 homogeneous matrix
    if R.shape == (4, 4):
        R = R[:3, :3]
    # ZYX (yaw-pitch-roll) -> returns (roll, pitch, yaw) in degrees
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])

def rpy_to_matrix(roll, pitch, yaw):
    roll  = np.radians(roll)
    pitch = np.radians(pitch)
    yaw   = np.radians(yaw)
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def rot_angle_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = np.clip(v, -1.0, 1.0)
    return np.degrees(np.arccos(v))

def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

def invert_rt(R, t):
    Rt = R.T
    return Rt, -Rt @ t

def to_cv_lists(R_list, t_list):
    Rcv = [np.asarray(R, dtype=np.float64) for R in R_list]
    tcv = [np.asarray(t, dtype=np.float64).reshape(3, 1) for t in t_list]
    return Rcv, tcv

def rel_motion(R_a, t_a, R_b, t_b):
    T_a = to_homogeneous(R_a, t_a)
    T_b = to_homogeneous(R_b, t_b)
    T_ab = T_b @ np.linalg.inv(T_a)
    return T_ab[:3, :3], T_ab[:3, 3]

def choose_marker_index(corners, ids, target_marker_id=None):
    """Pick a specific marker ID if provided; otherwise choose the largest-area marker."""
    if ids is None or len(ids) == 0:
        return None
    if target_marker_id is not None:
        matches = np.where(ids.flatten() == target_marker_id)[0]
        if len(matches) > 0:
            return int(matches[0])
        # fall through to largest if target not found
    # choose by area
    areas = []
    for i, c in enumerate(corners):
        # c shape: (1, 4, 2)
        poly = c.reshape(4, 2).astype(np.float32)
        areas.append(cv2.contourArea(poly))
    return int(np.argmax(areas))