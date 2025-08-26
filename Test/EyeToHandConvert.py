#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
#import pyzed.sl as sl

# ...existing imports...
import pyrealsense2 as rs
import os

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
SQUARE_LEN_M      = 0.02474  # measure your printed square side (meters)
MARKER_LEN_RATIO  = 0.78     # calib.io default unless you changed it
MARKER_LEN_M      = MARKER_LEN_RATIO * SQUARE_LEN_M  # measure your printed marker side (meters)

# If you KNOW these, set them; otherwise leave None to auto-lock from the image
ARUCO_DICT_ID    = None     # e.g. cv2.aruco.DICT_4X4_250
FIRST_MARKER_ID  = None     # e.g. 17

# Capture gating (encourage diverse robot poses)
MIN_ANGLE_DEG    = 8.0
MIN_TRANS_M      = 0.03
MIN_SAMPLES      = 12
TARGET_SAMPLES   = 20

AXIS_LEN_M       = 0.08
SAVE_NPY         = "handeye_result_stereo.npy"

# Incremental-capture controls
BATCH_SIZE       = 1
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "../output/captures")

# Load RealSense intrinsics/distortion from calibration file
calib = np.load("../output/realsense_calibration.npz")
K = calib["camera_matrix"]
dist = calib["dist_coeffs"]


# =========================
# Utilities
# =========================
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    if degrees:
        roll = np.deg2rad(roll); pitch = np.deg2rad(pitch); yaw = np.deg2rad(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx  # yaw-pitch-roll (ZYX)

def invert_rt(R, t):
    Rt = R.T
    return Rt, -Rt @ t

def to_cv_lists(R_list, t_list):
    Rcv = [np.asarray(R, dtype=np.float64) for R in R_list]
    tcv = [np.asarray(t, dtype=np.float64).reshape(3,1) for t in t_list]
    return Rcv, tcv

def se3(R, t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t; return T

def rot_angle_deg(R):
    val = (np.trace(R) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))

def rel_motion(A1, t1, A2, t2):
    T1 = se3(A1, t1); T2 = se3(A2, t2)
    T12 = T2 @ np.linalg.inv(T1)
    return T12[:3,:3], T12[:3,3]


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
    def get_position(self):
        code, pos = self.arm.get_position(is_radian=not USE_DEG)
        if code != 0:
            raise RuntimeError(f"xArm get_position failed, code={code}")
        return pos

# =========================
# Simple helpers
# =========================

def _to_homogeneous(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3]  = v3(t)           # <- handles (3,), (3,1), or (1,3)
    return T


def _invert_se3(T):
    R = T[:3,:3]; t = T[:3,3]
    RT = R.T
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = RT
    Ti[:3,3] = -RT @ t
    return Ti

def _matrix_to_rpy(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])
def v3(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(3) if x.size == 3 else x.squeeze().reshape(3)

def get_abs(listOfThree):
    try:
        result = abs(listOfThree[0]) + abs(listOfThree[1]) + abs(listOfThree[2])
        return result
    except Exception as e:
        print("Failed with get abs",e)
    
# =========================
# Main
# =========================
def main():
    print("\n=== Instructions ===")
    print("• Fix RealSense rigidly in the environment (eye-to-hand).")
    print("• Mount ChArUco board rigidly on the gripper.")
    print("• Move to varied poses (large rotations + translations).")
    print("• Press [SPACE] to capture, [q] to finish.\n")

    print("Opening RealSense...")
    rs_cam = RealSenseSource(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)

    print("Connecting xArm...")
    xarm = XArmClient(XARM_IP)

    ch_state = make_charuco_state()
    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    # Ensure output dir for captured images exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track saved image filenames to allow discarding the last batch
    saved_images = []

    last_Rg, last_tg = None, None

    try:
        prev_xyz_result = None
        prev_rpy_result = None
        current = 0
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

                R_gb, t_gb = invert_rt(R_bg, t_bg)

                accept = True
                if last_Rg is not None:
                    dR, dt = rel_motion(last_Rg, last_tg, R_gb, t_gb)
                    ang = rot_angle_deg(dR); d = np.linalg.norm(dt)
                    if ang < MIN_ANGLE_DEG and d < MIN_TRANS_M:
                        print(f"Pose too similar (Δang={ang:.1f}°, Δt={d*1000:.1f} mm); move more.")
                        accept = False

                if accept:
                    # Save capture image
                    capture_idx = len(R_g2b_list) + 1
                    img_path = os.path.join(OUTPUT_DIR, f"capture_{capture_idx:02d}.png")
                    try:
                        cv2.imwrite(img_path, color)
                        saved_images.append(img_path)
                    except Exception as e:
                        print(f"⚠️  Failed to save image: {e}")

                    # Accumulate hand-eye data
                    R_g2b_list.append(R_gb); t_g2b_list.append(t_gb)
                    R, t = det
                    R_t2c_list.append(R); t_t2c_list.append(t)
                    last_Rg, last_tg = R_gb, t_gb
                    print(f"Captured #{len(R_g2b_list)}  (markers:{ch_state['last_markers']}, charuco:{ch_state['last_charuco']})  → saved {os.path.basename(img_path)}")
                    current += 1
                    # After every BATCH_SIZE captures, run an incremental calibration preview
                    if len(R_g2b_list) >= BATCH_SIZE and (len(R_g2b_list) % BATCH_SIZE == 0) and current > 3:
                        try:
                            Rg_prev, tg_prev = to_cv_lists(R_g2b_list, t_g2b_list)
                            Rt_prev, tt_prev = to_cv_lists(R_t2c_list, t_t2c_list)
                            # --- preview block ---
                            R_cam2base_prev, t_cam2base_prev = cv2.calibrateHandEye(
                                R_gripper2base=Rg_prev, t_gripper2base=tg_prev,
                                R_target2cam=Rt_prev,  t_target2cam=tt_prev,
                                method=cv2.CALIB_HAND_EYE_PARK
                            )
                            t_cam2base_prev = v3(t_cam2base_prev)

                            T_base_cam_prev = np.eye(4)
                            T_base_cam_prev[:3,:3] = R_cam2base_prev
                            T_base_cam_prev[:3,3]  = t_cam2base_prev  # <- column expects (3,), not (3,1)

                            np.set_printoptions(precision=6, suppress=True)
                            print("\n=== Preview T_base_cam after {0} captures ===".format(len(R_g2b_list)))
                            print(T_base_cam_prev)
                            # MoveArm3-style prints (match labels and values)
                            try:
                                charucoToCam = _to_homogeneous(R_t2c_list[-1], t_t2c_list[-1])  # Camera ← Board
                                gripperToCam = T_base_cam_prev @ charucoToCam                 # (label matches MoveArm3)

                                # Actual Base ← EE from robot
                                R_bg_last, t_bg_last = invert_rt(R_g2b_list[-1], t_g2b_list[-1])
                                T_base_gripper = _to_homogeneous(R_bg_last, t_bg_last)

                                print("gripper to base")
                                print(T_base_gripper)
                                print("calculated gripper to cam")
                                print(gripperToCam)

                                actual_rpy = _matrix_to_rpy(T_base_gripper)
                                actual_xyz = T_base_gripper[:3, 3] * 1000
                                estimated_rpy = _matrix_to_rpy(gripperToCam)
                                estimated_xyz = gripperToCam[:3, 3] * 1000
                                print(actual_xyz)
                                print(estimated_xyz)
                                print(actual_xyz - estimated_xyz)
                                print(actual_rpy - estimated_rpy)
                                if prev_xyz_result is not None and prev_rpy_result is not None:
                                    print("into our statement")
                                    print("this is the absoule vaule of prev xyz result", get_abs(prev_xyz_result))
                                    print("this is the absoule vaule of prev rpy result", get_abs(prev_rpy_result))
                                    print("this is the absoule vaule of curr xyz result", get_abs(actual_xyz - estimated_xyz))
                                    print("this is the absoule vaule of curr rpy result", get_abs(actual_rpy - estimated_rpy))

                                
                            except Exception as e:
                                print(f"⚠️  Preview evaluation failed: {e}")
                            # Save the preview so MoveArm3 can be run immediately for evaluation
                            try:
                                np.savez(os.path.join(os.path.dirname(__file__), "../output/markercalibration.npz"),
                                         last_mark=T_base_cam_prev)
                            except Exception as e:
                                print(f"⚠️  Failed to save preview calibration: {e}")

                            def _vec(x):
                                return np.atleast_1d(np.asarray(x)).reshape(-1)

                            # Compute current absolute errors (vectors)
                            cur_xyz_err = np.abs(_vec(actual_xyz) - _vec(estimated_xyz))
                            cur_rpy_err = np.abs(_vec(actual_rpy) - _vec(estimated_rpy))

                            # Determine if we should discard the last batch
                            if (prev_xyz_result is not None) and (prev_rpy_result is not None):
                                prev_xyz_err = np.abs(_vec(prev_xyz_result))
                                prev_rpy_err = np.abs(_vec(prev_rpy_result))
                                # "worse or equal" in every component -> discard
                                should_discard = np.all(prev_xyz_err <= cur_xyz_err) and np.all(prev_rpy_err <= cur_rpy_err)
                            else:
                                should_discard = False

                            if should_discard:
                                # Remove last batch from memory and delete image files
                                for _ in range(BATCH_SIZE):
                                    if R_g2b_list:
                                        R_g2b_list.pop(); t_g2b_list.pop()
                                        R_t2c_list.pop(); t_t2c_list.pop()
                                    if saved_images:
                                        path_to_remove = saved_images.pop()
                                        try:
                                            if os.path.exists(path_to_remove):
                                                os.remove(path_to_remove)
                                                print(f"Removed {os.path.basename(path_to_remove)}")
                                        except Exception as e:
                                            print(f"⚠️  Failed removing image: {e}")

                                # Update last_Rg/last_tg to new tail
                                last_Rg, last_tg = (R_g2b_list[-1], t_g2b_list[-1]) if R_g2b_list else (None, None)
                                print(f"Discarded last {BATCH_SIZE} captures. Current count: {len(R_g2b_list)}")
                            else:
                                # Update prev_* only if meaningfully changed
                                new_xyz = _vec(actual_xyz) - _vec(estimated_xyz)
                                new_rpy = _vec(actual_rpy) - _vec(estimated_rpy)
                                if (prev_xyz_result is None) or (not np.allclose(prev_xyz_result, new_xyz, rtol=1e-6, atol=1e-9)):
                                    prev_xyz_result = new_xyz
                                if (prev_rpy_result is None) or (not np.allclose(prev_rpy_result, new_rpy, rtol=1e-6, atol=1e-9)):
                                    prev_rpy_result = new_rpy
                                print("Kept captures.")

                        except Exception as e:
                            print(f"⚠️  Preview calibration failed: {e}")
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
    print("\nRunning hand-eye (Park)...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=Rg, t_gripper2base=tg,
        R_target2cam=Rt,  t_target2cam=tt,
        method=cv2.CALIB_HAND_EYE_PARK
    )
    t_cam2base = t_cam2base.reshape(3)
    T_base_cam = np.eye(4); T_base_cam[:3,:3]=R_cam2base; T_base_cam[:3,3]=t_cam2base

    np.set_printoptions(precision=6, suppress=True)
    print("\n=== T_base_cam (camera pose in base frame) ===")
    print(T_base_cam)
    np.savez("../output/markercalibration.npz",
    last_mark = T_base_cam)

    # MoveArm3-style prints at the end using the last capture
    try:
        charucoToCam = _to_homogeneous(R_t2c_list[-1], t_t2c_list[-1])
        gripperToCam = T_base_cam @ charucoToCam
        R_bg_last, t_bg_last = invert_rt(R_g2b_list[-1], t_g2b_list[-1])
        T_base_gripper = _to_homogeneous(R_bg_last, t_bg_last)
        print("gripper to base")
        print(T_base_gripper)
        print("calculated gripper to cam")
        print(gripperToCam)
        actual_rpy = _matrix_to_rpy(T_base_gripper)
        actual_xyz = T_base_gripper[:3, 3] * 1000
        estimated_rpy = _matrix_to_rpy(gripperToCam)
        estimated_xyz = gripperToCam[:3, 3] * 1000
        print(actual_xyz)
        print(estimated_xyz)
        print(actual_xyz - estimated_xyz)
        print(actual_rpy - estimated_rpy)
    except Exception as e:
        print(f"⚠️  Final evaluation failed: {e}")

    np.save(SAVE_NPY, {
        "resolution": (REALSENSE_WIDTH, REALSENSE_HEIGHT),
        "charuco": {
            "squares_xy": (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            "square_len_m": SQUARE_LEN_M,
            "marker_len_m": MARKER_LEN_M,
            "dict_id": int(ARUCO_DICT_ID) if ARUCO_DICT_ID is not None else (int(ch_state['dict_id']) if ch_state['dict_id'] is not None else None),
            "first_marker_id": int(FIRST_MARKER_ID) if FIRST_MARKER_ID is not None else (int(ch_state['first_off']) if ch_state['first_off'] is not None else None),
        },
        "K": K, "dist": dist,
        "R_base_cam": R_cam2base, "t_base_cam": t_cam2base,
        "T_base_cam": T_base_cam,
        "residuals": None,
        "captures": len(R_g2b_list)
    }, allow_pickle=True)
    print(f"\nSaved → {SAVE_NPY}")

if __name__ == "__main__":
    main()