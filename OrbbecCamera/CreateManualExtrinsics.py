#!/usr/bin/env python3
"""
Create Manual Eye-in-Hand Extrinsics Calibration

This script creates an EyeInHand.npz file with manually specified extrinsics.
Useful for testing or when you have precise measurements of your camera-to-gripper transformation.

Usage:
    python CreateManualExtrinsics.py --x 0.0742 --y 0 --z -0.1389 --roll 0 --pitch 0 --yaw 90
    python CreateManualExtrinsics.py --x 74.2 --y 0 --z -138.9 --roll 0 --pitch 0 --yaw 90 --units mm
"""

import numpy as np
import argparse
from scipy.spatial.transform import Rotation as Rsc
import os


def create_manual_extrinsics(x, y, z, roll, pitch, yaw,
                             camera_calib_path='../output/orbbec_calibration.npz',
                             output_path='./calibrate_result/EyeInHand_manual.npz',
                             set_as_main=True):
    """
    Create a manual extrinsics calibration file.

    Args:
        x, y, z: Camera-to-Gripper translation in meters
        roll, pitch, yaw: Camera-to-Gripper rotation in degrees (ZYX Euler)
        camera_calib_path: Path to camera intrinsics calibration
        output_path: Where to save the manual extrinsics
        set_as_main: If True, also save as EyeInHand.npz
    """

    print("🔧 Creating manual extrinsics calibration file")
    print("=" * 70)
    print(f"\n📍 Manual Camera→Gripper Transformation:")
    print(f"  Translation (m): [{x:.4f}, {y:.4f}, {z:.4f}]")
    print(f"  Rotation (RPY deg): [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]")

    # Convert to rotation matrix using ZYX Euler convention
    r = Rsc.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    R_cam2gripper = r.as_matrix()
    t_cam2gripper = np.array([x, y, z])

    # Create homogeneous transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper

    # Get quaternion representation
    quat_xyzw = r.as_quat()  # x, y, z, w format

    # Compute inverse (Gripper→Camera)
    R_gc = R_cam2gripper.T
    t_gc = -R_gc @ t_cam2gripper
    r_gc = Rsc.from_matrix(R_gc)
    quat_gc = r_gc.as_quat()
    rpy_gc_zyx_deg = r_gc.as_euler('zyx', degrees=True)

    print(f"\n📐 Computed representations:")
    print(f"  Camera→Gripper:")
    print(f"    Rotation matrix:\n{R_cam2gripper}")
    print(f"    Quaternion (x,y,z,w): [{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, {quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]")
    print(f"\n  Gripper→Camera (inverse, for URDF):")
    print(f"    Translation: [{t_gc[0]:.6f}, {t_gc[1]:.6f}, {t_gc[2]:.6f}]")
    print(f"    Quaternion (x,y,z,w): [{quat_gc[0]:.6f}, {quat_gc[1]:.6f}, {quat_gc[2]:.6f}, {quat_gc[3]:.6f}]")
    print(f"    RPY ZYX (deg): [{rpy_gc_zyx_deg[0]:.3f}, {rpy_gc_zyx_deg[1]:.3f}, {rpy_gc_zyx_deg[2]:.3f}]")

    # Load camera intrinsics
    if os.path.exists(camera_calib_path):
        calib = np.load(camera_calib_path)
        K = calib['camera_matrix']
        D = calib['dist_coeffs']
        print(f"\n📷 Loaded camera intrinsics from: {camera_calib_path}")
        print(f"  Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        print(f"  Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    else:
        print(f"\n⚠️  Camera calibration not found at: {camera_calib_path}")
        print(f"  Creating extrinsics without camera intrinsics")
        K = np.eye(3)
        D = np.zeros(5)

    # Create the output data structure
    output_data = {
        'R_cam2gripper': R_cam2gripper,
        't_cam2gripper': t_cam2gripper,
        'T_cam2gripper': T_cam2gripper,
        'quat_cg_xyzw': quat_xyzw,
        'rpy_cg_zyx_deg': np.array([yaw, pitch, roll]),
        'R_gc': R_gc,
        't_gc': t_gc,
        'T_gc': np.linalg.inv(T_cam2gripper),
        'quat_gc_xyzw': quat_gc,
        'rpy_gc_zyx_deg': rpy_gc_zyx_deg,
        'selected_method': 'MANUAL',
        'num_poses': 0,
        'valid_pairs': ['manual_entry'],
        'camera_matrix': K,
        'dist_coeffs': D,
        'charuco_board': {
            'squares_x': 5,
            'squares_y': 7,
            'square_len_m': 0.03705,
            'marker_len_m': 0.02964
        }
    }

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to file
    np.savez(output_path, **output_data)
    print(f"\n✅ Saved manual extrinsics to: {output_path}")

    # Also create as main calibration if requested
    if set_as_main:
        main_path = os.path.join(os.path.dirname(output_path), 'EyeInHand.npz')
        np.savez(main_path, **output_data)
        print(f"✅ Saved as main calibration: {main_path}")

    print("\n" + "=" * 70)
    print("✅ Manual extrinsics calibration file created successfully!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Create manual eye-in-hand extrinsics calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Meters (default)
  python CreateManualExtrinsics.py --x 0.0742 --y 0 --z -0.1389 --roll 0 --pitch 0 --yaw 90

  # Millimeters
  python CreateManualExtrinsics.py --x 74.2 --y 0 --z -138.9 --roll 0 --pitch 0 --yaw 90 --units mm

  # Don't set as main calibration
  python CreateManualExtrinsics.py --x 0.074 --y 0 --z -0.14 --yaw 90 --no-set-main
        """
    )

    parser.add_argument('--x', type=float, required=True,
                       help='X translation (camera to gripper)')
    parser.add_argument('--y', type=float, required=True,
                       help='Y translation (camera to gripper)')
    parser.add_argument('--z', type=float, required=True,
                       help='Z translation (camera to gripper)')
    parser.add_argument('--roll', type=float, default=90.0,
                       help='Roll rotation in degrees (default: 0)')
    parser.add_argument('--pitch', type=float, default=0.0,
                       help='Pitch rotation in degrees (default: 0)')
    parser.add_argument('--yaw', type=float, default=0.0,
                       help='Yaw rotation in degrees (default: 0)')
    parser.add_argument('--units', choices=['m', 'mm', 'cm'], default='m',
                       help='Units for translation (default: meters)')
    parser.add_argument('--camera-calib', default='../output/orbbec_calibration.npz',
                       help='Path to camera intrinsics calibration')
    parser.add_argument('--output', default='./calibrate_result/EyeInHand_manual.npz',
                       help='Output file path')
    parser.add_argument('--no-set-main', action='store_true',
                       help='Do not save as main EyeInHand.npz')

    args = parser.parse_args()

    # Convert units to meters
    if args.units == 'mm':
        x, y, z = args.x / 1000.0, args.y / 1000.0, args.z / 1000.0
        print(f"Converting from mm: ({args.x}, {args.y}, {args.z}) mm → ({x:.4f}, {y:.4f}, {z:.4f}) m")
    elif args.units == 'cm':
        x, y, z = args.x / 100.0, args.y / 100.0, args.z / 100.0
        print(f"Converting from cm: ({args.x}, {args.y}, {args.z}) cm → ({x:.4f}, {y:.4f}, {z:.4f}) m")
    else:
        x, y, z = args.x, args.y, args.z

    create_manual_extrinsics(
        x, y, z, args.roll, args.pitch, args.yaw,
        camera_calib_path=args.camera_calib,
        output_path=args.output,
        set_as_main=not args.no_set_main
    )


if __name__ == "__main__":
    main()
