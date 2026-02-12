#!/usr/bin/env python3
"""
Create first_pose.txt from recorded ROS2 bag or manual input.

Since your existing data was collected with the old collector (no first_pose.txt),
this script helps you create it manually or from a ROS bag.

Usage:
    # Manual input (if you know the starting position):
    python script/create_first_pose_from_rosbag.py --sequence 01 --manual \
        --x -10.0 --y 5.0 --z 0.0 --yaw 0.0

    # From ROS2 bag (TODO: implement):
    # python script/create_first_pose_from_rosbag.py --sequence 01 --bag path/to/bag

Author: awang
Date: 2026-02-10
"""

import argparse
import numpy as np
from pathlib import Path


def create_pose_matrix(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    """
    Create 4x4 pose matrix from position and orientation.

    Args:
        x, y, z: Position in meters
        roll, pitch, yaw: Orientation in radians

    Returns:
        4x4 transformation matrix
    """
    # Rotation matrix from Euler angles (ZYX convention)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def main():
    parser = argparse.ArgumentParser(description="Create first_pose.txt for coordinate alignment")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence ID (e.g., 01)")
    parser.add_argument("--manual", action="store_true", help="Manual input mode")
    parser.add_argument("--x", type=float, default=0.0, help="X position (meters)")
    parser.add_argument("--y", type=float, default=0.0, help="Y position (meters)")
    parser.add_argument("--z", type=float, default=0.0, help="Z position (meters)")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll (radians)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch (radians)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw (radians)")
    parser.add_argument("--bag", type=str, help="Path to ROS2 bag (not yet implemented)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CREATE FIRST_POSE.TXT")
    print("="*60 + "\n")

    sequence_dir = Path(f"/home/awang/Documents/TRAILbot/OpenGraph/data/isaac_warehouse/{args.sequence}")

    if not sequence_dir.exists():
        print(f"ERROR: Sequence directory not found: {sequence_dir}")
        return

    if args.manual:
        print("Manual input mode")
        print(f"  Position: ({args.x:.2f}, {args.y:.2f}, {args.z:.2f})")
        print(f"  Orientation: roll={args.roll:.3f}, pitch={args.pitch:.3f}, yaw={args.yaw:.3f} rad")

        T = create_pose_matrix(args.x, args.y, args.z, args.roll, args.pitch, args.yaw)
    elif args.bag:
        print(f"ERROR: ROS bag parsing not yet implemented")
        print(f"Please use --manual mode for now")
        return
    else:
        print("ERROR: Must specify either --manual or --bag")
        return

    print(f"\nFirst pose (4x4):")
    print(T)

    # Save to first_pose.txt
    first_pose_path = sequence_dir / "first_pose.txt"
    pose_flat = T[:3, :].flatten()
    np.savetxt(first_pose_path, pose_flat.reshape(1, -1), fmt='%.12e')

    print(f"\n✓ Saved to: {first_pose_path}")
    print("\nNow you can run:")
    print(f"  python script/build_tiergraph.py --config-name=isaac_warehouse sequence={args.sequence}")
    print(f"  python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence={args.sequence}")


if __name__ == "__main__":
    main()
