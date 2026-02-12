#!/usr/bin/env python3
"""
Visualize TierGraph - Hierarchical Warehouse Scene Graph

Visualizes TierGraph results with:
- Infrastructure bounding boxes (zones, aisles, shelves, sections)
- Object point clouds with hierarchical coloring (Z-offset corrected)
- Hierarchy connections (parent → child edges)

IMPORTANT: This script applies a Z offset (+0.781m) to OpenGraph point clouds
to align them with the warehouse layout coordinate frame. The warehouse layout
has floor at Z=0.0m after alignment, but OpenGraph point clouds are in robot
frame where floor is at Z≈-0.781m (robot base_link is 0.641m above floor).

Usage:
    python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=02

Controls:
    - [1] Show/hide zone bounding boxes
    - [2] Show/hide aisle bounding boxes
    - [3] Show/hide shelf bounding boxes
    - [4] Show/hide section bounding boxes
    - [5] Show/hide hierarchy edges
    - [6] Color by hierarchy level
    - [I] Color by instance
    - [R] Color by RGB (original colors)
    - [Q] Exit

Author: awang (TierGraph thesis)
Date: 2026-02-10 (Z-offset fix: 2026-02-10)
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import json
import gzip
import pickle
import hydra
import numpy as np
import open3d as o3d
from pathlib import Path
from omegaconf import DictConfig
from some_class.map_calss import MapObjectList
from utils.coordinate_alignment import get_aligned_warehouse_layout
import distinctipy


def compute_point_cloud_z_offset(objects, sample_size=10):
    """
    Compute Z offset needed to align point clouds with warehouse layout.

    Estimates the floor Z coordinate from point clouds and computes offset
    to bring floor to Z=0.0m (matching the warehouse layout).

    Args:
        objects: MapObjectList from OpenGraph
        sample_size: Number of objects to sample for floor estimation

    Returns:
        Z offset to apply to point clouds
    """
    floor_z_estimates = []

    for obj in objects[:sample_size]:
        points = np.asarray(obj['pcd'].points)
        if len(points) == 0:
            continue

        # Floor should be in bottom 5th percentile
        z_coords = points[:, 2]
        floor_candidates = z_coords[z_coords < np.percentile(z_coords, 5)]

        if len(floor_candidates) > 0:
            floor_z_estimates.append(np.median(floor_candidates))

    if not floor_z_estimates:
        print("Warning: Could not estimate floor Z, using default offset 0.781m")
        return 0.781

    floor_z = np.median(floor_z_estimates)
    offset = -floor_z  # Bring floor from floor_z to 0.0

    print(f"Computed Z offset from point clouds: {offset:.3f}m")
    print(f"  (Floor at Z={floor_z:.3f}m in robot frame → Z=0.0m after offset)")

    return offset


# Visualization utilities (from OpenGraph)
def create_bbox_lineset(bounds, color=[1, 0, 0]):
    """Create Open3D lineset for a bounding box."""
    min_pt = bounds['min']
    max_pt = bounds['max']

    # Get Z bounds (default to 0 and 3m if not specified)
    min_z = min_pt.get('z', 0.0)
    max_z = max_pt.get('z', 3.0)

    # 8 corners of the box
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

    # 12 edges connecting the corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return lineset


def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """Create a ball mesh at the given center."""
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def create_hierarchy_edge(point1, point2, color=[0, 1, 0], radius=0.02):
    """Create a cylinder connecting two points (parent-child relationship)."""
    # Create line between points
    direction = point2 - point1
    length = np.linalg.norm(direction)

    if length < 0.01:  # Too short to visualize
        return []

    direction = direction / length

    # Create cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

    # Align cylinder with the direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        cylinder.rotate(R, center=[0, 0, 0])

    # Translate to position
    cylinder.translate(point1 + direction * length / 2)
    cylinder.paint_uniform_color(color)

    return [cylinder]


def load_tiergraph(graph_path):
    """Load TierGraph JSON."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def load_opengraph_results(result_path):
    """Load OpenGraph results."""
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        bg_objects = MapObjectList()
        if results['bg_objects']:
            bg_objects.load_serializable(results["bg_objects"])
        else:
            bg_objects = None
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError(f"Unknown results type: {type(results)}")

    return objects, bg_objects


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "="*60)
    print("TIERGRAPH VISUALIZATION")
    print("="*60 + "\n")

    # Load data
    result_path = Path(cfg.result_path)
    graph_path = Path(cfg.scenegraph_path)

    print(f"Loading OpenGraph results from {result_path}...")
    objects, bg_objects = load_opengraph_results(result_path)
    print(f"  Loaded {len(objects)} objects")

    print(f"Loading TierGraph from {graph_path}...")
    tiergraph = load_tiergraph(graph_path)
    print(f"  Loaded {tiergraph['statistics']['total_nodes']} nodes")
    print(f"  Loaded {tiergraph['statistics']['total_edges']} edges")

    # Build node lookup
    nodes_by_id = {node['id']: node for node in tiergraph['nodes']}

    # Prepare colors
    hierarchy_colors = {
        'zone': [1.0, 0.0, 0.0],      # Red
        'aisle': [0.0, 1.0, 0.0],     # Green
        'shelf': [0.0, 0.0, 1.0],     # Blue
        'section': [1.0, 1.0, 0.0],   # Yellow
        'object': [1.0, 0.0, 1.0],    # Magenta
    }

    # Generate instance colors
    instance_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
    instance_colors_dict = {f"object_{i}": c for i, c in enumerate(instance_colors)}

    # Create geometries
    print("\nCreating visualizations...")

    # 1. Object point clouds (with Z offset to align with warehouse layout)
    # CRITICAL: OpenGraph point clouds are in robot frame (floor at Z≈-0.781m)
    # but warehouse bounding boxes are offset to have floor at Z=0.0m
    # Compute the Z offset dynamically from the point cloud data
    POINT_CLOUD_Z_OFFSET = compute_point_cloud_z_offset(objects, sample_size=20)

    pcds = []
    object_centers = {}
    for i, obj in enumerate(objects):
        # Get original point cloud points and colors
        original_pcd = obj['pcd']
        points = np.asarray(original_pcd.points).copy()

        # Apply Z offset to align with warehouse coordinate frame
        points[:, 2] += POINT_CLOUD_Z_OFFSET

        # Create new point cloud with transformed points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Copy colors if they exist
        if original_pcd.has_colors():
            pcd.colors = original_pcd.colors

        pcds.append(pcd)

        # Store center for hierarchy edges (after offset)
        center = np.mean(points, axis=0)
        object_centers[f"object_{i}"] = center

    # 2. Infrastructure bounding boxes
    zone_boxes = []
    aisle_boxes = []
    shelf_boxes = []
    section_boxes = []

    for node in tiergraph['nodes']:
        if node['type'] == 'zone':
            lineset = create_bbox_lineset(node['bounds'], hierarchy_colors['zone'])
            zone_boxes.append(lineset)
            # Store center for edges
            center = node['bounds']['center']
            object_centers[node['id']] = np.array([center['x'], center['y'], center.get('z', 1.5)])
        elif node['type'] == 'aisle':
            lineset = create_bbox_lineset(node['bounds'], hierarchy_colors['aisle'])
            aisle_boxes.append(lineset)
            center = node['bounds']['center']
            object_centers[node['id']] = np.array([center['x'], center['y'], center.get('z', 1.5)])
        elif node['type'] == 'shelf':
            lineset = create_bbox_lineset(node['bounds'], hierarchy_colors['shelf'])
            shelf_boxes.append(lineset)
            center = node['bounds']['center']
            object_centers[node['id']] = np.array([center['x'], center['y'], center.get('z', 1.5)])
        elif node['type'] == 'section':
            lineset = create_bbox_lineset(node['bounds'], hierarchy_colors['section'])
            section_boxes.append(lineset)
            center = node['bounds']['center']
            object_centers[node['id']] = np.array([center['x'], center['y'], center.get('z', 1.5)])

    print(f"  Created {len(zone_boxes)} zone boxes")
    print(f"  Created {len(aisle_boxes)} aisle boxes")
    print(f"  Created {len(shelf_boxes)} shelf boxes")
    print(f"  Created {len(section_boxes)} section boxes")

    # 3. Hierarchy edges (parent → child)
    hierarchy_edges = []
    for edge in tiergraph['edges']:
        source_id = edge['source']
        target_id = edge['target']

        if source_id in object_centers and target_id in object_centers:
            source_pt = object_centers[source_id]
            target_pt = object_centers[target_id]

            # Use green color for containment edges
            edge_geom = create_hierarchy_edge(source_pt, target_pt, color=[0.0, 0.8, 0.0], radius=0.02)
            hierarchy_edges.extend(edge_geom)

    print(f"  Created {len(hierarchy_edges)} hierarchy edges")

    # Initialize visualizer
    print("\nInitializing Open3D visualizer...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='TierGraph Visualization', width=1920, height=1080)

    # Add all geometries
    for pcd in pcds:
        vis.add_geometry(pcd)

    # Add infrastructure (initially hidden except zones)
    for box in zone_boxes:
        vis.add_geometry(box)
    for box in aisle_boxes:
        vis.add_geometry(box)
    for box in shelf_boxes:
        vis.add_geometry(box)
    # Sections are too many, keep hidden initially

    # Store visibility state
    class State:
        show_zones = True
        show_aisles = True
        show_shelves = True
        show_sections = False
        show_edges = False
        color_mode = 'instance'  # 'instance', 'hierarchy', 'rgb'

    state = State()

    # Callback functions
    def toggle_zones(vis):
        state.show_zones = not state.show_zones
        for box in zone_boxes:
            if state.show_zones:
                vis.add_geometry(box, reset_bounding_box=False)
            else:
                vis.remove_geometry(box, reset_bounding_box=False)
        print(f"Zones: {'ON' if state.show_zones else 'OFF'}")

    def toggle_aisles(vis):
        state.show_aisles = not state.show_aisles
        for box in aisle_boxes:
            if state.show_aisles:
                vis.add_geometry(box, reset_bounding_box=False)
            else:
                vis.remove_geometry(box, reset_bounding_box=False)
        print(f"Aisles: {'ON' if state.show_aisles else 'OFF'}")

    def toggle_shelves(vis):
        state.show_shelves = not state.show_shelves
        for box in shelf_boxes:
            if state.show_shelves:
                vis.add_geometry(box, reset_bounding_box=False)
            else:
                vis.remove_geometry(box, reset_bounding_box=False)
        print(f"Shelves: {'ON' if state.show_shelves else 'OFF'}")

    def toggle_sections(vis):
        state.show_sections = not state.show_sections
        if state.show_sections:
            for box in section_boxes:
                vis.add_geometry(box, reset_bounding_box=False)
        else:
            for box in section_boxes:
                vis.remove_geometry(box, reset_bounding_box=False)
        print(f"Sections: {'ON' if state.show_sections else 'OFF'}")

    def toggle_edges(vis):
        state.show_edges = not state.show_edges
        if state.show_edges:
            for edge in hierarchy_edges:
                vis.add_geometry(edge, reset_bounding_box=False)
        else:
            for edge in hierarchy_edges:
                vis.remove_geometry(edge, reset_bounding_box=False)
        print(f"Hierarchy edges: {'ON' if state.show_edges else 'OFF'}")

    def color_by_hierarchy(vis):
        """Color objects by their hierarchy level."""
        state.color_mode = 'hierarchy'
        for i, obj in enumerate(objects):
            node_id = f"object_{i}"
            node = nodes_by_id.get(node_id)
            if node:
                color = hierarchy_colors.get(node['type'], [0.5, 0.5, 0.5])
            else:
                color = [0.5, 0.5, 0.5]

            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
            vis.update_geometry(pcd)
        print("Colored by hierarchy level")

    def color_by_instance(vis):
        """Color objects by instance (random colors)."""
        state.color_mode = 'instance'
        for i, obj in enumerate(objects):
            color = instance_colors[i]
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
            vis.update_geometry(pcd)
        print("Colored by instance")

    def color_by_rgb(vis):
        """Restore original RGB colors."""
        state.color_mode = 'rgb'
        for i, obj in enumerate(objects):
            pcd = pcds[i]
            # Original colors should be stored in obj['pcd']
            original_colors = np.asarray(obj['pcd'].colors)
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            vis.update_geometry(pcd)
        print("Colored by RGB (original)")

    # Register callbacks
    vis.register_key_callback(ord("1"), toggle_zones)
    vis.register_key_callback(ord("2"), toggle_aisles)
    vis.register_key_callback(ord("3"), toggle_shelves)
    vis.register_key_callback(ord("4"), toggle_sections)
    vis.register_key_callback(ord("5"), toggle_edges)
    vis.register_key_callback(ord("6"), color_by_hierarchy)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("R"), color_by_rgb)

    # Initial coloring
    color_by_instance(vis)

    print("\n" + "="*60)
    print("KEYBOARD CONTROLS")
    print("="*60)
    print("  [1] Toggle zone bounding boxes")
    print("  [2] Toggle aisle bounding boxes")
    print("  [3] Toggle shelf bounding boxes")
    print("  [4] Toggle section bounding boxes")
    print("  [5] Toggle hierarchy edges")
    print("  [6] Color by hierarchy level")
    print("  [I] Color by instance (random colors)")
    print("  [R] Color by RGB (original colors)")
    print("  [Q] Exit")
    print("="*60 + "\n")

    # Run visualizer
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
