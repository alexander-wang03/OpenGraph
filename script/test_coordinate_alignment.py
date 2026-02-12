#!/usr/bin/env python3
"""
Test coordinate alignment between robot and warehouse layout.

This script verifies that the warehouse layout can be correctly transformed
from Isaac Sim global coordinates to robot-relative coordinates.

Usage:
    python script/test_coordinate_alignment.py
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import numpy as np
from pathlib import Path
from utils.coordinate_alignment import (
    load_absolute_first_pose,
    save_absolute_first_pose,
    get_aligned_warehouse_layout
)


def main():
    print("\n" + "="*60)
    print("COORDINATE ALIGNMENT TEST")
    print("="*60 + "\n")

    # Test with sequence 01
    sequence_dir = Path("/home/awang/Documents/TRAILbot/OpenGraph/data/isaac_warehouse/01")
    warehouse_layout_path = "/home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json"

    # Check if first_pose.txt exists
    first_pose_path = sequence_dir / "first_pose.txt"

    if not first_pose_path.exists():
        print("WARNING: first_pose.txt not found!")
        print(f"  Expected at: {first_pose_path}")
        print("\nThis means the data was collected with the old collector.")
        print("You have two options:")
        print("  1. Re-collect data with the updated collector")
        print("  2. Manually create first_pose.txt with the robot's starting position")
        print("\nFor now, using identity transform (no alignment).")
        T_first = np.eye(4)
    else:
        print(f"Loading first pose from: {first_pose_path}")
        T_first = load_absolute_first_pose(sequence_dir)
        print(f"First pose (4x4):")
        print(T_first)
        print(f"\nFirst position: [{T_first[0, 3]:.2f}, {T_first[1, 3]:.2f}, {T_first[2, 3]:.2f}]")

    # Load and align warehouse layout
    print(f"\nLoading warehouse layout from: {warehouse_layout_path}")
    aligned_layout = get_aligned_warehouse_layout(warehouse_layout_path, sequence_dir)

    # Show some transformed bounds
    print("\n" + "="*60)
    print("SAMPLE TRANSFORMED BOUNDS")
    print("="*60)

    storage_zone = next(z for z in aligned_layout['functional_zones'] if z['id'] == 'zone_storage')
    print(f"\nStorage Zone bounds (robot-relative):")
    print(f"  Min: [{storage_zone['bounds']['min']['x']:.2f}, "
          f"{storage_zone['bounds']['min']['y']:.2f}]")
    print(f"  Max: [{storage_zone['bounds']['max']['x']:.2f}, "
          f"{storage_zone['bounds']['max']['y']:.2f}]")
    print(f"  Center: [{storage_zone['bounds']['center']['x']:.2f}, "
            f"{storage_zone['bounds']['center']['y']:.2f}]")

    if storage_zone.get('aisles'):
        aisle = list(storage_zone['aisles'].values())[0]
        print(f"\nFirst aisle bounds (robot-relative):")
        print(f"  Min: [{aisle['bounds']['min']['x']:.2f}, "
              f"{aisle['bounds']['min']['y']:.2f}]")
        print(f"  Max: [{aisle['bounds']['max']['x']:.2f}, "
              f"{aisle['bounds']['max']['y']:.2f}]")

    # Save aligned layout
    output_path = sequence_dir / "warehouse_layout_aligned_test.json"
    import json
    with open(output_path, 'w') as f:
        json.dump(aligned_layout, f, indent=2)
    print(f"\nSaved aligned layout to: {output_path}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

    if not first_pose_path.exists():
        print("IMPORTANT: To fix visualization, you need to either:")
        print("  1. Re-collect data with: python script/collect_isaac_data.py ...")
        print("     (This will create first_pose.txt automatically)")
        print("  2. OR manually create first_pose.txt with 12 numbers (3x4 pose matrix)")
        print("\nWithout this, the warehouse layout won't align with your point clouds!")


if __name__ == "__main__":
    main()
