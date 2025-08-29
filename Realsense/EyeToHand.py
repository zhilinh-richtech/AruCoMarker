#!/usr/bin/env python3
import cv2
import numpy as np
from xarm.wrapper import XArmAPI
#import pyzed.sl as sl

# ...existing imports...
import pyrealsense2 as rs
from camera import create_camera
from utils import (
    rpy_to_matrix,
    rot_angle_deg,
    to_homogeneous,
    invert_rt,
    to_cv_lists,
    rel_motion,
)
from terminal_display import Display, draw_axes_ascii_friendly
import argparse
import sys
import time
import shutil
import select
import termios
import tty
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

# Load RealSense intrinsics/distortion from calibration file
calib = np.load("../output/realsense_calibration.npz")
K = calib["camera_matrix"]
dist = calib["dist_coeffs"]


# =========================
# Utilities
# =========================
def euler_rpy_to_R(roll, pitch, yaw, degrees=True):
    if not degrees:
        roll = np.degrees(roll); pitch = np.degrees(pitch); yaw = np.degrees(yaw)
    return rpy_to_matrix(roll, pitch, yaw)

 


# =========================
# RealSense source
# =========================
class RealSenseSource:
    def __init__(self, width=1280, height=800, fps=30, camera_kind: str = "auto"):
        self.cam = create_camera(kind=camera_kind, width=width, height=height, fps=fps)

    def read(self):
        return self.cam.read()

    def close(self):
        self.cam.close()


# =========================
# Terminal/ASCII helpers moved to terminal_display.Display
# =========================

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
    parser = argparse.ArgumentParser(description="Eye-to-Hand hand-eye calibration (terminal-friendly)")
    parser.add_argument("--mode", choices=["gui", "ascii", "ascii_hi", "headless"],
                        default=("gui" if os.environ.get("DISPLAY") else "ascii"),
                        help="Display mode: OpenCV GUI, ASCII in terminal, or headless")
    parser.add_argument("--camera", choices=["auto", "realsense", "opencv"], default="auto",
                        help="Camera backend to use: RealSense (if available) or OpenCV UVC")
    args = parser.parse_args()

    print("\n=== Instructions ===")
    print("• Fix RealSense rigidly in the environment (eye-to-hand).")
    print("• Mount ChArUco board rigidly on the gripper.")
    print("• Move to varied poses (large rotations + translations).")
    if args.mode == "gui":
        print("• Press [SPACE] to capture, [q] to finish in the GUI window.\n")
    else:
        print("• In terminal, press SPACE to capture, q to finish.\n")

    print("Opening camera...")
    rs_cam = RealSenseSource(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS, camera_kind=args.camera)

    print("Connecting xArm...")
    xarm = XArmClient(XARM_IP)

    ch_state = make_charuco_state()
    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    last_Rg, last_tg = None, None

    display = Display(args.mode)
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
                if args.mode in ("ascii", "ascii_hi", "headless"):
                    draw_axes_ascii_friendly(vis, K, dist, R, t, AXIS_LEN_M, thickness=6)
                else:
                    cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(R)[0], t.reshape(3,1), AXIS_LEN_M)
                cv2.putText(vis, "Board detected", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(vis, "No board", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cap_txt = f"Captures: {len(R_g2b_list)} / {TARGET_SAMPLES}"
            cv2.putText(vis, cap_txt, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            status = f"Captures: {len(R_g2b_list)}/{TARGET_SAMPLES}  |  {'Board detected' if det is not None else 'No board'}  |  [SPACE]=capture, q=quit"
            key = display.update(vis, status)

            if key == 'q':
                break
            if key == ' ':
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
                    R_g2b_list.append(R_gb); t_g2b_list.append(t_gb)
                    R, t = det
                    R_t2c_list.append(R); t_t2c_list.append(t)
                    last_Rg, last_tg = R_gb, t_gb
                    print(f"Captured #{len(R_g2b_list)}  (markers:{ch_state['last_markers']}, charuco:{ch_state['last_charuco']})")
    finally:
        display.close()
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

    res = handeye_residuals(Rg, tg, Rt, tt, R_cam2base, t_cam2base)
    if res:
        print("\nResiduals: "
              f"rot mean={res['rot_deg']['mean']:.3f}°, med={res['rot_deg']['median']:.3f}°, p95={res['rot_deg']['p95']:.3f}°; "
              f"trans mean={res['trans_m']['mean']:.4f} m, med={res['trans_m']['median']:.4f} m, p95={res['trans_m']['p95']:.4f} m")

    # Detailed validation is handled in MoveArm3.py

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
        "residuals": res,
        "captures": len(R_g2b_list)
    }, allow_pickle=True)
    print(f"\nSaved → {SAVE_NPY}")

if __name__ == "__main__":
    main()