#!/usr/bin/env python3
"""
Press SPACE to print the xArm TCP pose.
Press 'q' to quit.

Requires: ufactory xArm Python SDK (pip install xarm==1.* if needed)
"""

import sys
import time
import math
from datetime import datetime

# ---------- Keyboard helpers (cross-platform) ----------
def _getch():
    """Return a single key press (non-echo), blocking."""
    try:
        # Windows
        import msvcrt
        ch = msvcrt.getch()
        # Handle arrow/function keys which return a prefix + code
        if ch in (b'\x00', b'\xe0'):
            msvcrt.getch()  # discard second byte
            return ''
        try:
            return ch.decode('utf-8', errors='ignore')
        except Exception:
            return ''
    except ImportError:
        # POSIX
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

def wait_for_keypress(valid=(' ', 'q')):
    """Block until one of the valid keys is pressed; return the key."""
    while True:
        ch = _getch()
        if ch in valid:
            return ch

# ---------- xArm setup ----------
from xarm.wrapper import XArmAPI

XARM_IP = "192.168.10.201"  # <-- change this to your robot's IP

def deg2rad(d): return d * math.pi / 180.0

def main():
    print("Connecting to xArm at", XARM_IP, "...")
    arm = XArmAPI(XARM_IP, do_not_open=False)
    # Optional: ensure API is ready
    arm.motion_enable(enable=True)
    # You don't need to change mode/state to read pose.

    print("\nReady.")
    print("Press SPACE to print TCP pose; press 'q' to quit.\n")

    try:
        while True:
            key = wait_for_keypress(valid=(' ', 'q'))
            if key == 'q':
                print("Exiting.")
                break

            # SPACE pressed: read pose
            code, pose = arm.get_position(is_radian=False)  # [x(mm), y, z, roll(deg), pitch(deg), yaw(deg)]
            if code != 0 or pose is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] get_position failed: code={code}")
                continue

            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = pose[:6]

            x_m, y_m, z_m = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
            rx_rad, ry_rad, rz_rad = deg2rad(rx_deg), deg2rad(ry_deg), deg2rad(rz_deg)

            # Pretty print
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{ts}] TCP Pose")
            print(f"  Position:  x={x_m:+.6f} m  y={y_m:+.6f} m  z={z_m:+.6f} m")
            print(f"  RPY (deg): roll={rx_deg:+.3f}  pitch={ry_deg:+.3f}  yaw={rz_deg:+.3f}")
            print(f"  RPY (rad): roll={rx_rad:+.6f}  pitch={ry_rad:+.6f}  yaw={rz_rad:+.6f}")

            # If you also want the 4x4 homogeneous transform, uncomment below:
            # (ZYX intrinsic, which matches common xArm RPY convention)
            # import numpy as np
            # cr, sr = math.cos(rx_rad), math.sin(rx_rad)
            # cp, sp = math.cos(ry_rad), math.sin(ry_rad)
            # cy, sy = math.cos(rz_rad), math.sin(rz_rad)
            # Rz = np.array([[cy, -sy, 0],
            #                [sy,  cy, 0],
            #                [ 0,   0, 1]])
            # Ry = np.array([[ cp, 0, sp],
            #                [  0, 1,  0],
            #                [-sp, 0, cp]])
            # Rx = np.array([[1,  0,   0],
            #                [0, cr, -sr],
            #                [0, sr,  cr]])
            # R = Rz @ Ry @ Rx
            # T = np.eye(4)
            # T[:3, :3] = R
            # T[:3, 3] = [x_m, y_m, z_m]
            # np.set_printoptions(precision=6, suppress=True)
            # print("  Homogeneous T:\n", T)

            # Small debounce so holding space doesn't spam too fast (tweak as needed)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
