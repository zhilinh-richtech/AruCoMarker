#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from datetime import datetime

# ----------------- Config -----------------
XARM_IP = '192.168.10.201'
ARUCO_DICT = cv2.aruco.DICT_4X4_250
MARKER_LENGTH = 0.063  # meters
save_dir = 'ArUcoBoardcalib_images_left'
os.makedirs(save_dir, exist_ok=True)

def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Load RealSense intrinsics/distortion from calibration file
calib = np.load("../output/realsense_calibration.npz")
camera_matrix = calib["camera_matrix"]
dist_coeffs   = calib["dist_coeffs"]

# ----------------- Utility Functions -----------------
def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rpy_to_matrix(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                   [ np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# ----------------- Connect to xArm -----------------
arm = XArmAPI(XARM_IP)
arm.motion_enable(True)

# ----------------- RealSense color source -----------------
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
        try:
            self.pipeline.stop()
        except Exception:
            pass

# ----------------- ArUco detector -----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


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

# ----------------- Lists for pose pairs -----------------
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# -------- Auto-load previous samples (optional) --------
sample_path = os.path.join(save_dir, 'handeye_samples.npz')
if os.path.exists(sample_path):
    data = np.load(sample_path, allow_pickle=True)
    R_gripper2base = list(data['R_gripper2base'])
    t_gripper2base = list(data['t_gripper2base'])
    R_target2cam   = list(data['R_target2cam'])
    t_target2cam   = list(data['t_target2cam'])
    print(f"[INFO] Loaded {len(R_gripper2base)} previous samples.")
else:
    print("[INFO] No previous samples found. Starting fresh.")

pose_count = len(R_gripper2base)
print("[INFO] Move robot to various poses.")
print("[INFO] SPACE=capture JPG, s=save sample (id==0), q=calibrate & quit")

cam = RealSenseSource()

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # Detect ArUco
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs
            )
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], 0.03)

        # HUD
        cv2.putText(frame, "SPACE=capture JPG  s=save sample (id0)  q=calibrate&quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow('Aruco Detection', frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:  # SPACE -> capture JPG
            img_path = os.path.join(save_dir, f"color_{ts()}.jpg")
            ok = cv2.imwrite(img_path, frame)
            print(f"[CAPTURE] Saved {img_path}" if ok else "[ERROR] Failed to save JPG")

        elif key == ord('s'):
            # Save a sample ONLY if exactly one marker with id==0 is visible
            if ids is not None and len(ids) == 1 and int(ids[0][0]) == 0:
                R_marker_cam, _ = cv2.Rodrigues(rvecs[0])
                t_marker_cam = tvecs[0].flatten()

                code, pose = arm.get_position()  # [x(mm), y(mm), z(mm), roll, pitch, yaw]
                if code != 0 or pose is None or len(pose) < 6:
                    print("[ERROR] Failed to get robot pose from xArm.")
                    continue

                t_gripper_base = np.array(pose[:3], dtype=float) / 1000.0
                R_gripper_base = rpy_to_matrix(pose[3], pose[4], pose[5])

                R_gripper2base.append(R_gripper_base)
                t_gripper2base.append(t_gripper_base)
                R_target2cam.append(R_marker_cam)
                t_target2cam.append(t_marker_cam)

                pose_count += 1
                print(f"[INFO] Pose #{pose_count} saved.")

                # Persist samples immediately
                np.savez(sample_path,
                         R_gripper2base=np.array(R_gripper2base, dtype=object),
                         t_gripper2base=np.array(t_gripper2base, dtype=object),
                         R_target2cam=np.array(R_target2cam, dtype=object),
                         t_target2cam=np.array(t_target2cam, dtype=object))
                print(f"[INFO] Samples saved to {sample_path}")
            else:
                found = 0 if ids is None else len(ids)
                ids_list = [] if ids is None else ids.flatten().tolist()
                print(f"[ERROR] Need exactly one marker with id==0. "
                      f"Detected {found}, ids={ids_list}")

finally:
    cv2.destroyAllWindows()
    cam.close()

# ----------------- Solve Eye-in-Hand Calibration -----------------
if pose_count >= 3:
    print("\n[INFO] Running eye-in-hand calibration...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam
    )

    T_cam_gripper = to_homogeneous(R_cam2gripper, t_cam2gripper)
    T_gripper_cam = np.linalg.inv(T_cam_gripper)

    print("\n=== Calibration Result (Eye-in-Hand) ===")
    print("T_cam_gripper (Camera → Gripper):\n", T_cam_gripper)
    print("\nT_gripper_cam (Gripper → Camera):\n", T_gripper_cam)

    out_calib = os.path.join(save_dir, 'eye_to_hand_calibration.npz')
    np.savez(out_calib, T_cam_gripper=T_cam_gripper, T_gripper_cam=T_gripper_cam)
    print(f"\n✅ Saved: {out_calib}")
else:
    print("❌ Not enough pose samples! You need at least 3.")

# ----------------- Cleanup -----------------
arm.disconnect()
