#!/usr/bin/env python3
"""
Compute the Z offset needed to align OpenGraph point clouds with warehouse layout.

This determines the robot's base_link height above the Isaac Sim floor by
measuring the floor Z coordinate in the point cloud data.

Usage:
    python script/compute_floor_offset.py --sequence 02
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import argparse
import numpy as np
from pathlib import Path


def compute_floor_z_from_velodyne(sequence_dir: Path, num_frames: int = 10):
    """
    Estimate floor Z coordinate from raw velodyne scans.

    The floor should be the lowest Z values in the point clouds.

    Args:
        sequence_dir: Path to sequence directory
        num_frames: Number of frames to sample

    Returns:
        Estimated floor Z coordinate in robot frame
    """
    velodyne_dir = sequence_dir / "velodyne"
    bin_files = sorted(velodyne_dir.glob("*.bin"))

    if not bin_files:
        print(f"No velodyne files found in {velodyne_dir}")
        return None

    # Sample evenly spaced frames
    indices = np.linspace(0, len(bin_files)-1, min(num_frames, len(bin_files)), dtype=int)

    all_floor_z = []

    for idx in indices:
        bin_file = bin_files[idx]
        # Load point cloud (Nx4: x,y,z,intensity)
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

        # Get Z coordinates
        z_coords = points[:, 2]

        # Floor should be in bottom 5th percentile
        floor_candidates = z_coords[z_coords < np.percentile(z_coords, 5)]

        if len(floor_candidates) > 0:
            # Take median of floor candidates (robust to outliers)
            floor_z = np.median(floor_candidates)
            all_floor_z.append(floor_z)

    if not all_floor_z:
        return None

    # Average across frames
    floor_z_estimate = np.median(all_floor_z)

    return floor_z_estimate


def main():
    parser = argparse.ArgumentParser(description="Compute Z offset for coordinate alignment")
    parser.add_argument("--sequence", type=str, default="02", help="Sequence ID")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to sample")
    args = parser.parse_args()

    print("="*60)
    print("COMPUTING Z OFFSET FOR COORDINATE ALIGNMENT")
    print("="*60)

    sequence_dir = Path(f"/home/awang/Documents/TRAILbot/OpenGraph/data/isaac_warehouse/{args.sequence}")

    if not sequence_dir.exists():
        print(f"ERROR: Sequence directory not found: {sequence_dir}")
        return

    # Compute floor Z from velodyne scans
    print(f"\nAnalyzing {args.num_frames} frames from sequence {args.sequence}...")
    floor_z_robot = compute_floor_z_from_velodyne(sequence_dir, args.num_frames)

    if floor_z_robot is None:
        print("ERROR: Could not estimate floor Z coordinate")
        return

    print(f"\n✓ Floor Z coordinate in robot frame: {floor_z_robot:.3f}m")

    # Isaac Sim floor is at -0.14m in global coordinates
    isaac_sim_floor_z = -0.14

    # After alignment, we want floor at Z=0 in both frames
    # So offset = -(floor_z_robot)
    # This brings robot floor from floor_z_robot to 0.0
    # And we separately offset warehouse from -0.14 to 0.0

    # Actually, the total offset needed for objects is:
    # We want: object_z_robot + offset = object_z_warehouse
    # Where warehouse floor is at 0 after +0.14 offset
    # And robot floor is at floor_z_robot
    # So: floor_z_robot + offset = 0
    # Therefore: offset = -floor_z_robot

    z_offset_for_objects = -floor_z_robot
    z_offset_for_warehouse = 0.14  # Standard offset to bring floor from -0.14 to 0.0

    print(f"\n" + "="*60)
    print("RECOMMENDED OFFSETS")
    print("="*60)
    print(f"For warehouse layout (in apply_z_offset): +{z_offset_for_warehouse:.3f}m")
    print(f"For object centroids (in extract_object_centroid): +{z_offset_for_objects:.3f}m")
    print(f"\nUpdate build_tiergraph.py line 73:")
    print(f"  def extract_object_centroid(pcd, z_offset={z_offset_for_objects:.3f}):")
    print("="*60)

    # Also estimate robot base_link height above floor
    base_link_height = isaac_sim_floor_z - floor_z_robot
    print(f"\nInferred robot base_link height above Isaac Sim floor: {base_link_height:.3f}m")
    print(f"  (Isaac floor at {isaac_sim_floor_z:.2f}m, robot floor at {floor_z_robot:.3f}m)")


if __name__ == "__main__":
    main()
