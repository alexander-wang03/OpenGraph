#!/usr/bin/env python3
"""
Quick test of TierGraph implementation.

Tests the warehouse hierarchy building and object assignment logic
without requiring full OpenGraph results.

Usage:
    python script/test_tiergraph.py
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import numpy as np
from utils.warehouse_graph_builder import WarehouseGraphBuilder


def main():
    print("\n" + "="*60)
    print("TIERGRAPH TEST")
    print("="*60 + "\n")

    # Initialize builder with warehouse layout
    warehouse_layout_path = "/home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json"
    print(f"Loading warehouse layout from:\n  {warehouse_layout_path}\n")

    builder = WarehouseGraphBuilder(warehouse_layout_path)

    # Print initial infrastructure
    builder.print_summary()

    # Test object assignments with known positions
    print("\n" + "="*60)
    print("TESTING OBJECT ASSIGNMENT")
    print("="*60 + "\n")

    test_objects = [
        # (position [x, y, z], description)
        (np.array([-15.0, 15.0, 0.5]), "Box on bottom shelf"),  # Storage zone, shelf
        (np.array([-15.0, 15.0, 3.0]), "Box on top shelf"),     # Storage zone, shelf, high
        (np.array([-22.0, 16.0, 1.0]), "Box in aisle 0"),       # Storage zone, aisle 0
        (np.array([0.0, 0.0, 0.2]), "Box on floor"),            # Storage zone floor
        (np.array([-20.0, -18.0, 0.5]), "Box in receiving"),    # Receiving zone
        (np.array([-50.0, 0.0, 0.0]), "Box outside warehouse"), # Outside (should fail)
    ]

    for i, (position, description) in enumerate(test_objects):
        object_id = f"test_object_{i}"
        success, path = builder.assign_object_to_hierarchy(
            object_id=object_id,
            object_position=position,
            object_caption=description
        )

        status = "✓" if success else "✗"
        print(f"{status} [{i}] {description}")
        print(f"    Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        print(f"    Hierarchy: {path}")
        print()

    # Print final summary
    builder.print_summary()

    # Export test graph
    output_path = "/tmp/tiergraph_test.json"
    builder.export_graph(output_path)
    print(f"\nTest graph exported to: {output_path}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
