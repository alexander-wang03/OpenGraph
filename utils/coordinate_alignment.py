"""
Coordinate Alignment Utilities for TierGraph

Handles transformation between:
- Isaac Sim global coordinates (warehouse layout)
- Robot-relative coordinates (OpenGraph point clouds)

The issue: collect_isaac_data.py saves relative poses (first frame = identity),
but warehouse_layout.json is in Isaac Sim global coordinates. We need to align them.

Author: awang
Date: 2026-02-10
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_absolute_first_pose(sequence_dir: Path) -> np.ndarray:
    """
    Load the absolute first pose from the sequence directory.

    If first_pose.txt exists, load it. Otherwise, return identity.

    Args:
        sequence_dir: Path to sequence directory (e.g., data/isaac_warehouse/01/)

    Returns:
        4x4 transformation matrix
    """
    first_pose_path = sequence_dir / "first_pose.txt"

    if first_pose_path.exists():
        with open(first_pose_path, 'r') as f:
            pose_flat = np.array([float(x) for x in f.read().strip().split()])
            pose = pose_flat.reshape(3, 4)
            # Convert to 4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
            return pose_4x4
    else:
        # No first pose saved, assume identity (relative coordinates)
        return np.eye(4)


def save_absolute_first_pose(sequence_dir: Path, absolute_pose: np.ndarray):
    """
    Save the absolute first pose to the sequence directory.

    This should be called during data collection.

    Args:
        sequence_dir: Path to sequence directory
        absolute_pose: 4x4 transformation matrix
    """
    first_pose_path = sequence_dir / "first_pose.txt"
    pose_flat = absolute_pose[:3, :].flatten()
    np.savetxt(first_pose_path, pose_flat.reshape(1, -1), fmt='%.12f')


def transform_warehouse_layout(layout: Dict, transform: np.ndarray) -> Dict:
    """
    Transform warehouse layout from global coordinates to robot-relative.

    Applies the inverse of the first robot pose to all infrastructure bounding boxes.

    Args:
        layout: Warehouse layout dict (from warehouse_layout.json)
        transform: 4x4 transformation matrix (robot's first pose in global frame)

    Returns:
        Transformed layout dict
    """
    # Inverse transform: global -> robot-relative
    T_inv = np.linalg.inv(transform)

    # Transform all bounding box corners
    def transform_bounds(bounds):
        """Transform a bounding box."""
        min_pt = bounds['min']
        max_pt = bounds['max']

        # Get 8 corners of the box
        min_z = min_pt.get('z', 0.0)
        max_z = max_pt.get('z', 3.0)

        corners = np.array([
            [min_pt['x'], min_pt['y'], min_z],
            [max_pt['x'], min_pt['y'], min_z],
            [max_pt['x'], max_pt['y'], min_z],
            [min_pt['x'], max_pt['y'], min_z],
            [min_pt['x'], min_pt['y'], max_z],
            [max_pt['x'], min_pt['y'], max_z],
            [max_pt['x'], max_pt['y'], max_z],
            [min_pt['x'], max_pt['y'], max_z],
        ])

        # Transform corners
        corners_homo = np.hstack([corners, np.ones((8, 1))])
        corners_transformed = (T_inv @ corners_homo.T).T[:, :3]

        # Compute new min/max
        new_min = corners_transformed.min(axis=0)
        new_max = corners_transformed.max(axis=0)
        new_center = (new_min + new_max) / 2
        new_size = new_max - new_min

        return {
            'min': {'x': float(new_min[0]), 'y': float(new_min[1]), 'z': float(new_min[2])},
            'max': {'x': float(new_max[0]), 'y': float(new_max[1]), 'z': float(new_max[2])},
            'center': {'x': float(new_center[0]), 'y': float(new_center[1]), 'z': float(new_center[2])},
            'size': {'x': float(new_size[0]), 'y': float(new_size[1]), 'z': float(new_size[2])},
        }

    # Deep copy layout
    import copy
    transformed_layout = copy.deepcopy(layout)

    # Transform floor vertices
    if 'floor' in transformed_layout and 'vertices' in transformed_layout['floor']:
        floor_vertices = transformed_layout['floor']['vertices']
        floor_z = transformed_layout['floor']['bounds']['min'].get('z', 0.0)

        # Transform each vertex
        transformed_vertices = []
        for vertex in floor_vertices:
            vertex_3d = np.array([vertex[0], vertex[1], floor_z, 1.0])
            vertex_transformed = T_inv @ vertex_3d
            transformed_vertices.append([float(vertex_transformed[0]), float(vertex_transformed[1])])

        transformed_layout['floor']['vertices'] = transformed_vertices

        # Recompute floor bounds from transformed vertices
        if transformed_vertices:
            xs = [v[0] for v in transformed_vertices]
            ys = [v[1] for v in transformed_vertices]
            transformed_layout['floor']['bounds'] = {
                'min': {'x': min(xs), 'y': min(ys), 'z': floor_z},
                'max': {'x': max(xs), 'y': max(ys), 'z': floor_z},
                'center': {'x': (min(xs) + max(xs)) / 2, 'y': (min(ys) + max(ys)) / 2, 'z': floor_z},
                'size': {'x': max(xs) - min(xs), 'y': max(ys) - min(ys), 'z': 0.0}
            }

    # Transform functional zone vertices (for General zone polygon)
    for zone in transformed_layout['functional_zones']:
        if 'vertices' in zone and zone['vertices']:
            zone_z = zone['bounds']['min'].get('z', 0.0) if 'bounds' in zone else 0.0

            transformed_zone_verts = []
            for vertex in zone['vertices']:
                vertex_3d = np.array([vertex[0], vertex[1], zone_z, 1.0])
                vertex_transformed = T_inv @ vertex_3d
                transformed_zone_verts.append([float(vertex_transformed[0]), float(vertex_transformed[1])])

            zone['vertices'] = transformed_zone_verts

    # Transform functional zone bounds
    for zone in transformed_layout['functional_zones']:
        zone['bounds'] = transform_bounds(zone['bounds'])

        # Transform aisles
        if 'aisles' in zone and zone['aisles']:
            for aisle_id, aisle in zone['aisles'].items():
                aisle['bounds'] = transform_bounds(aisle['bounds'])

        # Transform shelves
        if 'shelves' in zone and zone['shelves']:
            for shelf_id, shelf in zone['shelves'].items():
                shelf['bounds'] = transform_bounds(shelf['bounds'])

                # Transform sections
                if 'sections' in shelf and shelf['sections']:
                    for section_idx, section_level_data in shelf['sections'].items():
                        if isinstance(section_level_data, dict):
                            for tier_idx, tier_data in section_level_data.items():
                                if isinstance(tier_data, dict) and 'bounds' in tier_data:
                                    tier_data['bounds'] = transform_bounds(tier_data['bounds'])

    return transformed_layout


def get_aligned_warehouse_layout(warehouse_layout_path: str, sequence_dir: Path,
                                 z_offset: float = 0.14) -> Dict:
    """
    Load warehouse layout and align it to robot-relative coordinates.

    This is the main function to use in TierGraph scripts.

    Args:
        warehouse_layout_path: Path to warehouse_layout.json (global coordinates)
        sequence_dir: Path to sequence directory (to load first_pose.txt)
        z_offset: Z offset to apply (default 0.14m to account for Isaac Sim floor height)

    Returns:
        Warehouse layout in robot-relative coordinates
    """
    # Load global layout
    with open(warehouse_layout_path, 'r') as f:
        layout = json.load(f)

    # Load first pose
    T_first = load_absolute_first_pose(sequence_dir)

    # Apply Z offset to account for Isaac Sim floor being at -0.14m
    # but robot base_link being at approximately Z=0 (height above floor)
    print(f"Applying Z offset: {z_offset:.3f}m (Isaac Sim floor correction)")
    layout = apply_z_offset(layout, z_offset)

    # Check if we have a first pose for XY alignment
    if np.allclose(T_first, np.eye(4)):
        print("Warning: No first_pose.txt found for XY alignment.")
        print("Using only Z offset. If XY alignment looks wrong, recollect data or create first_pose.txt")
        return layout

    # Transform layout to robot-relative
    print(f"Aligning warehouse layout to robot frame...")
    print(f"  First pose: [{T_first[0, 3]:.2f}, {T_first[1, 3]:.2f}, {T_first[2, 3]:.2f}]")
    transformed_layout = transform_warehouse_layout(layout, T_first)

    return transformed_layout


def apply_z_offset(layout: Dict, z_offset: float) -> Dict:
    """
    Apply a constant Z offset to all bounding boxes in the layout.

    This accounts for the Isaac Sim floor being at Z=-0.14m while the robot's
    base_link frame has Z=0 at approximately floor height.

    Args:
        layout: Warehouse layout dict
        z_offset: Z offset to add (positive moves layout up)

    Returns:
        Layout with Z-shifted bounds
    """
    import copy
    shifted_layout = copy.deepcopy(layout)

    def shift_bounds_z(bounds):
        """Add z_offset to all Z coordinates in bounds."""
        shifted = copy.deepcopy(bounds)

        # Shift min/max/center Z if present
        if 'min' in shifted and 'z' in shifted['min']:
            shifted['min']['z'] += z_offset
        if 'max' in shifted and 'z' in shifted['max']:
            shifted['max']['z'] += z_offset
        if 'center' in shifted and 'z' in shifted['center']:
            shifted['center']['z'] += z_offset

        return shifted

    # Shift functional zones
    for zone in shifted_layout['functional_zones']:
        if 'bounds' in zone:
            zone['bounds'] = shift_bounds_z(zone['bounds'])

        # Shift aisles
        if 'aisles' in zone and zone['aisles']:
            for aisle_id, aisle in zone['aisles'].items():
                if 'bounds' in aisle:
                    aisle['bounds'] = shift_bounds_z(aisle['bounds'])

        # Shift shelves
        if 'shelves' in zone and zone['shelves']:
            for shelf_id, shelf in zone['shelves'].items():
                if 'bounds' in shelf:
                    shelf['bounds'] = shift_bounds_z(shelf['bounds'])

                # Shift sections
                if 'sections' in shelf and shelf['sections']:
                    for section_idx, section_level_data in shelf['sections'].items():
                        if isinstance(section_level_data, dict):
                            for tier_idx, tier_data in section_level_data.items():
                                if isinstance(tier_data, dict) and 'bounds' in tier_data:
                                    tier_data['bounds'] = shift_bounds_z(tier_data['bounds'])

    return shifted_layout
