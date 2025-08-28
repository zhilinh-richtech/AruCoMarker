#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
#import pyzed.sl as sl

# ...existing imports...
import pyrealsense2 as rs
import os
from utils import rpy_to_matrix, rot_angle_deg, to_homogeneous, invert_rt, to_cv_lists, rel_motion

# =========================
# CONFIG
# =========================
XARM_IP = "192.168.10.201"
USE_DEG = True

REALSENSE_WIDTH = 1280
REALSENSE_HEIGHT = 800
REALSENSE_FPS = 30

# calib.io ChArUco board (you said: rows=4, columns=6)
CHARUCO_SQUARES_X = 5       # columns (X across)
CHARUCO_SQUARES_Y = 7       # rows    (Y down)
SQUARE_LEN_M      = 0.034  # measure your printed square side (meters)
MARKER_LEN_RATIO  = 0.81     # calib.io default unless you changed it
MARKER_LEN_M      = MARKER_LEN_RATIO * SQUARE_LEN_M  # measure your printed marker side (meters)

# If you KNOW these, set them; otherwise leave None to auto-lock from the image
ARUCO_DICT_ID    = None     # e.g. cv2.aruco.DICT_4X4_250
FIRST_MARKER_ID  = None     # e.g. 17

# Capture gating (encourage diverse robot poses)
MIN_ANGLE_DEG    = 8.0
MIN_TRANS_M      = 0.03
MIN_SAMPLES      = 3
TARGET_SAMPLES   = 20

AXIS_LEN_M       = 0.08
SAVE_DIR         = "../output/poses"  # Directory to save pose pairs

# Load RealSense intrinsics/distortion from calibration file
calib = np.load("../output/realsense_calibration.npz")
K = calib["camera_matrix"]
dist = calib["dist_coeffs"]


# =========================
# Utilities
# =========================
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    # Delegate to shared utility. utils.rpy_to_matrix expects degrees.
    if not degrees:
        roll = np.degrees(roll); pitch = np.degrees(pitch); yaw = np.degrees(yaw)
    return rpy_to_matrix(roll, pitch, yaw)


# Use shared to_homogeneous from utils instead of local se3


 


# =========================
# RealSense source
# =========================
class RealSenseSource:
    def __init__(self, width=1280, height=800, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def close(self):
        self.pipeline.stop()


# =========================
# ChArUco with auto dict + firstMarkerId lock-in (no board mutation)
# =========================
def _dict_size(dict_id):
    m = {
        cv2.aruco.DICT_4X4_50: 50,    cv2.aruco.DICT_4X4_100: 100,
        cv2.aruco.DICT_4X4_250: 250,  cv2.aruco.DICT_4X4_1000: 1000,
        cv2.aruco.DICT_5X5_50: 50,    cv2.aruco.DICT_5X5_100: 100,
        cv2.aruco.DICT_5X5_250: 250,  cv2.aruco.DICT_5X5_1000: 1000,
    }
    return m.get(dict_id, 250)

def _make_board_and_detector(dict_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # Robust-ish defaults; good for prints/screens
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.detectInvertedMarker = True
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LEN_M, MARKER_LEN_M, aruco_dict
    )
    return board, detector

_COMMON_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
]

def make_charuco_state():
    state = {
        'locked': False,
        'dict_id': None,
        'first_off': None,   # firstMarkerId offset
        'board': None,
        'detector': None,
        'candidates': {},    # dict_id -> (board, detector)
        'last_markers': 0,
        'last_charuco': 0,
    }
    if ARUCO_DICT_ID is not None:
        board, det = _make_board_and_detector(ARUCO_DICT_ID)
        state.update({'locked': True,
                      'dict_id': ARUCO_DICT_ID,
                      'first_off': (FIRST_MARKER_ID or 0),
                      'board': board,
                      'detector': det})
    return state

def _adjust_ids(ids, offset, dict_sz):
    # ids is shape (N,1); keep shape
    ids_i = ids.astype(np.int32)
    ids_adj = ((ids_i - int(offset)) % dict_sz).astype(np.int32)
    return ids_adj

def estimate_charuco_pose(img_bgr, K, dist, debug_img=None, state=None):
    """
    Auto-lock dictionary and firstMarkerId from observed IDs.
    Returns (R, t) or None. Updates state['last_markers'], state['last_charuco'].
    """
    assert state is not None

    def draw_counts(img, markers, charuco):
        cv2.putText(img, f"markers:{markers}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(img, f"charuco:{charuco}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if not state['locked']:
        # try each common dict
        for did in _COMMON_DICTS if ARUCO_DICT_ID is None else [ARUCO_DICT_ID]:
            if did not in state['candidates']:
                state['candidates'][did] = _make_board_and_detector(did)
            board, det = state['candidates'][did]

            corners, ids, _ = det.detectMarkers(img_bgr)
            state['last_markers'] = 0 if ids is None else int(len(ids))
            if debug_img is not None and ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

            if ids is None or len(ids) < 4:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], 0)
                continue

            dict_sz = _dict_size(did)
            guess_off = int(np.min(ids)) if FIRST_MARKER_ID is None else int(FIRST_MARKER_ID)
            ids_adj = _adjust_ids(ids, guess_off, dict_sz)

            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids_adj, img_bgr, board)
            state['last_charuco'] = 0 if (retval is None) else int(retval)
            if retval is None or retval < 4:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, dist, None, None)
            if not ok:
                if debug_img is not None:
                    draw_counts(debug_img, state['last_markers'], state['last_charuco'])
                continue

            # Lock in
            state.update({'locked': True, 'dict_id': did, 'first_off': guess_off,
                          'board': board, 'detector': det})
            print(f"[ChArUco] Locked dict={did}, firstMarkerId={guess_off}, "
                  f"markers={len(ids)}, charuco={int(retval)}")

            if debug_img is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug_img, ch_corners, ch_ids)
                draw_counts(debug_img, state['last_markers'], state['last_charuco'])

            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)

        return None

    # locked path: reuse board/detector and adjust ids each time
    board, det = state['board'], state['detector']
    corners, ids, _ = det.detectMarkers(img_bgr)
    state['last_markers'] = 0 if ids is None else int(len(ids))
    if debug_img is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

    if ids is None or len(ids) < 4:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], 0)
        return None

    dict_sz = _dict_size(state['dict_id'])
    ids_adj = _adjust_ids(ids, state['first_off'], dict_sz)

    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids_adj, img_bgr, board)
    state['last_charuco'] = 0 if (retval is None) else int(retval)
    if retval is None or retval < 4:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], state['last_charuco'])
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, K, dist, None, None)
    if not ok:
        if debug_img is not None:
            draw_counts(debug_img, state['last_markers'], state['last_charuco'])
        return None

    if debug_img is not None:
        cv2.aruco.drawDetectedCornersCharuco(debug_img, ch_corners, ch_ids)
        draw_counts(debug_img, state['last_markers'], state['last_charuco'])

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)

# =========================
# xArm client
# =========================
class XArmClient:
    def __init__(self, ip):
        self.arm = XArmAPI(ip)
        self.arm.connect()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def get_base_to_gripper(self):
        code, pos = self.arm.get_position(is_radian=not USE_DEG)
        if code != 0:
            raise RuntimeError(f"xArm get_position failed, code={code}")
        x, y, z, roll, pitch, yaw = pos
        t_bg = np.array([x, y, z], dtype=np.float64) / 1000.0  # mm -> m
        R_bg = euler_rpy_to_R(roll, pitch, yaw, degrees=USE_DEG)
        return R_bg, t_bg

    def close(self):
        try: self.arm.disconnect()
        except Exception: pass

# =========================
# Residual diagnostics
# =========================
def handeye_residuals(Rg, tg, Rt, tt, R_cam2base, t_cam2base):
    """
    Calculate residuals for eye-in-hand hand-eye calibration.
    Rg: gripper poses in base frame (camera poses in base frame)
    tg: gripper translations in base frame
    Rt: target poses in camera frame
    tt: target translations in camera frame
    R_cam2base, t_cam2base: camera to base transformation
    """
    X = to_homogeneous(R_cam2base, t_cam2base)
    rots, trans = [], []
    for i in range(len(Rg) - 1):
        RA, tA = rel_motion(Rg[i], tg[i], Rg[i+1], tg[i+1])
        RB, tB = rel_motion(Rt[i+1], tt[i+1], Rt[i], tt[i])  # inverse order
        L = to_homogeneous(RA, tA) @ X
        Rhs = X @ to_homogeneous(RB, tB)
        dR = L[:3,:3].T @ Rhs[:3,:3]
        dtheta = rot_angle_deg(dR)
        dt = np.linalg.norm(L[:3,3] - Rhs[:3,3])
        rots.append(dtheta); trans.append(dt)
    if not rots: return None
    def stats(a): a=np.array(a); return dict(mean=float(np.mean(a)),
                                            median=float(np.median(a)),
                                            p95=float(np.percentile(a,95)))
    return dict(rot_deg=stats(rots), trans_m=stats(trans))
    
    # Keeping residuals only; defer detailed validation to MoveArm3.py
    
# =========================
# Main
# =========================
def main():
    print("\n=== Instructions ===")
    print("• Mount RealSense rigidly on the gripper (eye-in-hand).")
    print("• Fix ChArUco board rigidly in the environment.")
    print("• Move to varied poses (large rotations + translations).")
    print("• Press [SPACE] to capture, [q] to finish.\n")

    print("Opening RealSense...")
    rs_cam = RealSenseSource(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)

    print("Connecting xArm...")
    xarm = XArmClient(XARM_IP)

    ch_state = make_charuco_state()
    R_g2b_list, t_g2b_list = [], []  # gripper to base (camera pose)
    R_t2c_list, t_t2c_list = [], []  # target to camera
    captured_images = []  # Store captured images for saving

    last_Rg, last_tg = None, None

    try:
        while True:
            ok, color = rs_cam.read()
            if not ok or color is None:
                print("Camera read failed")
                break

            vis = color.copy()
            det = estimate_charuco_pose(color, K, dist, debug_img=vis, state=ch_state)

            if det is not None:
                R, t = det
                cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(R)[0], t.reshape(3,1), AXIS_LEN_M)
                cv2.putText(vis, "Board detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(vis, "No board", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cap_txt = f"Captures: {len(R_g2b_list)} / {TARGET_SAMPLES}"
            cv2.putText(vis, cap_txt, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("RealSense", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                if det is None:
                    print("⚠️  Need the board detected. Adjust and try again.")
                    continue
                try:
                    R_bg, t_bg = xarm.get_base_to_gripper()
                except Exception as e:
                    print(f"⚠️  xArm read failed: {e}")
                    continue

                # In eye-in-hand: camera is on gripper, so gripper pose = camera pose
                R_g2b = R_bg  # gripper to base (same as camera to base)
                t_g2b = t_bg

                accept = True
                if last_Rg is not None:
                    dR, dt = rel_motion(last_Rg, last_tg, R_g2b, t_g2b)
                    ang = rot_angle_deg(dR); d = np.linalg.norm(dt)
                    if ang < MIN_ANGLE_DEG and d < MIN_TRANS_M:
                        print(f"Pose too similar (Δang={ang:.1f}°, Δt={d*1000:.1f} mm); move more.")
                        accept = False

                if accept:
                    R_g2b_list.append(R_g2b); t_g2b_list.append(t_g2b)
                    R, t = det
                    R_t2c_list.append(R); t_t2c_list.append(t)
                    captured_images.append(color.copy())  # Store the captured image
                    last_Rg, last_tg = R_g2b, t_g2b
                    
                    # Save pose pair immediately
                    pose_num = len(R_g2b_list)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    # Convert rotation matrices to Euler angles for saving
                    R_g2b_euler = cv2.Rodrigues(R_g2b)[0]
                    R_t2c_euler = cv2.Rodrigues(R)[0]  # Use R from det
                    
                    # Calculate base to gripper transformation (inverse of gripper to base)
                    R_base2gripper, t_base2gripper = invert_rt(R_g2b, t_g2b)
                    
                    # Save as JPG
                    img_name = f"{SAVE_DIR}/pose{pose_num:03d}.jpg"
                    cv2.imwrite(img_name, captured_images[-1])
                    
                    # Save as NPY with base to gripper transformation
                    np.save(f"{SAVE_DIR}/pose{pose_num:03d}.npy", {
                        "R_base2gripper": R_base2gripper,      # Base to gripper rotation
                        "t_base2gripper": t_base2gripper,      # Base to gripper translation
                        "R_gripper2base": R_g2b,               # Original gripper to base (for reference)
                        "t_gripper2base": t_g2b,               # Original gripper to base (for reference)
                        "R_target2cam": R,                      # Target to camera (from det)
                        "t_target2cam": t,                      # Target to camera (from det)
                        "R_base2gripper_euler": cv2.Rodrigues(R_base2gripper)[0],  # Base to gripper in Euler angles
                        "R_target2cam_euler": R_t2c_euler,                         # Target to camera in Euler angles
                        "pose_number": pose_num,
                        "timestamp": pose_num  # You could add actual timestamp here if needed
                    }, allow_pickle=True)
                    
                    
                    print(f"Captured #{pose_num}  (markers:{ch_state['last_markers']}, charuco:{ch_state['last_charuco']})")
                    print(f"Saved → {img_name}")
                    print(f"Saved → {SAVE_DIR}/pose{pose_num:03d}.npy")
    finally:
        cv2.destroyAllWindows()
        rs_cam.close()
        xarm.close()

    n = len(R_g2b_list)
    if n < MIN_SAMPLES:
        print(f"Not enough samples ({n}). Aim for {TARGET_SAMPLES}+ varied poses.")
        return

    Rg, tg = to_cv_lists(R_g2b_list, t_g2b_list)
    Rt, tt = to_cv_lists(R_t2c_list, t_t2c_list)
    print("\nRunning hand-eye calibration (eye-in-hand)...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=Rg, t_gripper2base=tg,  # Camera poses in base frame
        R_target2cam=Rt,  t_target2cam=tt,     # Target poses in camera frame
        method=cv2.CALIB_HAND_EYE_PARK
    )
    t_cam2base = t_cam2base.reshape(3)

    # For eye-in-hand: camera is mounted on gripper
    # We want the transformation from camera frame to gripper frame
    # This is the fixed offset between camera and gripper TCP
    # We can calculate this from the calibration results
    
    # Since we have gripper poses in base frame and camera poses in base frame,
    # we can find the camera-to-gripper transformation
    # T_gripper2base = T_cam2base * T_gripper2cam
    # Therefore: T_gripper2cam = inv(T_cam2base) * T_gripper2base
    # And: T_cam2gripper = inv(T_gripper2cam)
    
    # For now, let's use the first pose to calculate this relationship
    if len(R_g2b_list) > 0:
        R_gripper2base = R_g2b_list[0]  # First gripper pose
        t_gripper2base = t_g2b_list[0]
        
        # T_gripper2base = T_cam2base * T_gripper2cam
        # T_gripper2cam = inv(T_cam2base) * T_gripper2base
        T_cam2base_4x4 = np.eye(4)
        T_cam2base_4x4[:3,:3] = R_cam2base
        T_cam2base_4x4[:3,3] = t_cam2base
        
        T_gripper2base_4x4 = np.eye(4)
        T_gripper2base_4x4[:3,:3] = R_gripper2base
        T_gripper2base_4x4[:3,3] = t_gripper2base
        
        # T_gripper2cam = inv(T_cam2base) * T_gripper2base
        T_gripper2cam = np.linalg.inv(T_cam2base_4x4) @ T_gripper2base_4x4
        
        # T_cam2gripper = inv(T_gripper2cam)
        T_cam2gripper = np.linalg.inv(T_gripper2cam)
        
        R_cam2gripper = T_cam2gripper[:3,:3]
        t_cam2gripper = T_cam2gripper[:3,3]
    else:
        # Fallback if no poses captured
        R_cam2gripper = np.eye(3)
        t_cam2gripper = np.zeros(3)
        T_cam2gripper = np.eye(4)

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== T_cam2gripper (camera to gripper transformation) ===")
    print(T_cam2base_4x4)
    print("\n=== t_cam2gripper (translation vector) ===")
    print(t_cam2base)
    np.save(f"{SAVE_DIR}/result.npy", {
        "t_cam2grip": t_cam2base,
        "R_cam2grip": R_cam2base,
        "T_cam2grip": T_cam2base_4x4
    }, allow_pickle=True)
    np.savez("../output/markercalibration.npz",
    last_mark = T_cam2gripper)

    res = handeye_residuals(Rg, tg, Rt, tt, R_cam2base, t_cam2base)
    if res:
        print("\nResiduals: "
              f"rot mean={res['rot_deg']['mean']:.3f}°, med={res['rot_deg']['median']:.3f}°, p95={res['rot_deg']['p95']:.3f}°; "
              f"trans mean={res['trans_m']['mean']:.4f} m, med={res['trans_m']['median']:.4f} m, p95={res['trans_m']['p95']:.4f} m")

    print(f"\n=== Eye-in-hand calibration complete ===")
    print(f"Total poses captured: {len(R_g2b_list)}")
    print(f"Pose pairs saved to: {SAVE_DIR}")
    print(f"Calibration result saved to: ../output/markercalibration.npz")

if __name__ == "__main__":
    main()