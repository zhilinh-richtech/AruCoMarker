#!/usr/bin/env python3
"""
Visualize Gripper Poses from .npy Files

This script loads robot gripper poses from .npy files and visualizes their
XYZ positions in 3D space.

Usage:
    python VisualizeGripperPoses.py --poses-dir ./new_poses_folder
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse


def load_pose_file(npy_path):
    """Load pose data from .npy file"""
    try:
        data = np.load(npy_path, allow_pickle=True).item()

        # Check which format the data is in
        if "R_gripper2base" in data:
            R = data["R_gripper2base"]
            t = data["t_gripper2base"]
        elif "R" in data:
            R = data["R"]
            t = data["t"]
        else:
            print(f"⚠️  Unknown pose format in {npy_path}")
            return None, None

        return R, t

    except Exception as e:
        print(f"❌ Failed to load {npy_path}: {e}")
        return None, None


def visualize_gripper_poses(poses_dir):
    """Visualize gripper poses in 3D"""

    # Find all .npy files
    npy_files = sorted(glob.glob(os.path.join(poses_dir, "pose*.npy")))

    if len(npy_files) == 0:
        print(f"❌ No pose files found in {poses_dir}")
        return

    print(f"Found {len(npy_files)} pose files")

    # Load all poses
    positions = []
    rotations = []
    pose_names = []

    for npy_file in npy_files:
        R, t = load_pose_file(npy_file)
        if R is not None and t is not None:
            positions.append(t)
            rotations.append(R)
            pose_names.append(os.path.basename(npy_file).replace('.npy', ''))

    if len(positions) == 0:
        print("❌ No valid poses loaded")
        return

    positions = np.array(positions)

    # Convert to millimeters for visualization
    positions_mm = positions * 1000.0

    print(f"\n✓ Loaded {len(positions)} poses")
    print(f"\nPosition statistics (millimeters):")
    print(f"  X: min={positions_mm[:,0].min():.2f}, max={positions_mm[:,0].max():.2f}, mean={positions_mm[:,0].mean():.2f}")
    print(f"  Y: min={positions_mm[:,1].min():.2f}, max={positions_mm[:,1].max():.2f}, mean={positions_mm[:,1].mean():.2f}")
    print(f"  Z: min={positions_mm[:,2].min():.2f}, max={positions_mm[:,2].max():.2f}, mean={positions_mm[:,2].mean():.2f}")

    # Create 3D visualization
    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions_mm[:, 0], positions_mm[:, 1], positions_mm[:, 2],
                c=range(len(positions_mm)), cmap='viridis', s=50, alpha=0.6)
    ax1.plot(positions_mm[:, 0], positions_mm[:, 1], positions_mm[:, 2],
             'b-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Gripper Trajectory')
    ax1.grid(True)

    # Add coordinate frame axes
    for i, (pos, rot) in enumerate(zip(positions_mm[::5], rotations[::5])):  # Show every 5th frame
        axis_length = 50.0  # 50mm axes
        colors = ['r', 'g', 'b']
        for j in range(3):
            axis = rot[:, j] * axis_length
            ax1.plot([pos[0], pos[0] + axis[0]],
                    [pos[1], pos[1] + axis[1]],
                    [pos[2], pos[2] + axis[2]],
                    colors[j], alpha=0.5, linewidth=1)

    # XY plane view
    ax2 = fig.add_subplot(132)
    sc = ax2.scatter(positions_mm[:, 0], positions_mm[:, 1],
                     c=range(len(positions_mm)), cmap='viridis', s=50, alpha=0.6)
    ax2.plot(positions_mm[:, 0], positions_mm[:, 1], 'b-', alpha=0.3, linewidth=1)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Plane (Top View)')
    ax2.grid(True)
    ax2.axis('equal')

    # Add start and end markers
    ax2.scatter(positions_mm[0, 0], positions_mm[0, 1], c='green', s=200, marker='*',
                edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax2.scatter(positions_mm[-1, 0], positions_mm[-1, 1], c='red', s=200, marker='X',
                edgecolors='black', linewidths=2, label='End', zorder=5)
    ax2.legend()

    # XZ plane view
    ax3 = fig.add_subplot(133)
    ax3.scatter(positions_mm[:, 0], positions_mm[:, 2],
                c=range(len(positions_mm)), cmap='viridis', s=50, alpha=0.6)
    ax3.plot(positions_mm[:, 0], positions_mm[:, 2], 'b-', alpha=0.3, linewidth=1)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Plane (Side View)')
    ax3.grid(True)
    ax3.axis('equal')

    # Add start and end markers
    ax3.scatter(positions_mm[0, 0], positions_mm[0, 2], c='green', s=200, marker='*',
                edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax3.scatter(positions_mm[-1, 0], positions_mm[-1, 2], c='red', s=200, marker='X',
                edgecolors='black', linewidths=2, label='End', zorder=5)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=[ax2, ax3], orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Pose Sequence Number')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(poses_dir, 'gripper_poses_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")

    plt.show()

    # Create a detailed position table
    print("\n" + "="*70)
    print("Detailed Position Table (millimeters)")
    print("="*70)
    print(f"{'Pose':<12} {'X (mm)':>10} {'Y (mm)':>10} {'Z (mm)':>10} {'Distance to Prev (mm)':>22}")
    print("-"*70)

    for i, (name, pos) in enumerate(zip(pose_names, positions)):
        if i == 0:
            dist = 0.0
        else:
            dist = np.linalg.norm(pos - positions[i-1]) * 1000

        print(f"{name:<12} {pos[0]*1000:>10.2f} {pos[1]*1000:>10.2f} {pos[2]*1000:>10.2f} {dist:>22.2f}")

    print("="*70)

    # Calculate total path length
    total_length = 0.0
    for i in range(1, len(positions)):
        total_length += np.linalg.norm(positions[i] - positions[i-1])

    print(f"\n📏 Total path length: {total_length*1000:.2f} mm ({total_length:.4f} m)")

    # Calculate workspace bounds
    workspace_x = (positions[:, 0].max() - positions[:, 0].min()) * 1000
    workspace_y = (positions[:, 1].max() - positions[:, 1].min()) * 1000
    workspace_z = (positions[:, 2].max() - positions[:, 2].min()) * 1000

    print(f"📦 Workspace bounds:")
    print(f"   X range: {workspace_x:.2f} mm")
    print(f"   Y range: {workspace_y:.2f} mm")
    print(f"   Z range: {workspace_z:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description="Visualize gripper poses from .npy files")
    parser.add_argument("--poses-dir", default="./new_poses_folder",
                       help="Directory containing pose*.npy files")

    args = parser.parse_args()

    print("="*70)
    print("Gripper Pose Visualization")
    print("="*70)
    print(f"Poses directory: {args.poses_dir}\n")

    if not os.path.exists(args.poses_dir):
        print(f"❌ Directory not found: {args.poses_dir}")
        return

    visualize_gripper_poses(args.poses_dir)


if __name__ == "__main__":
    main()
