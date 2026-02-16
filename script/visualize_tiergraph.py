#!/usr/bin/env python3
"""
Visualize TierGraph in Isaac Sim GLOBAL coordinates (not robot-relative).

This version visualizes the warehouse layout in the original Isaac Sim global
frame, making it match the warehouse_layout.json visualization. Point clouds
are transformed from robot frame back to global frame.

Usage:
    python script/visualize_tiergraph_global.py --config-name=isaac_warehouse sequence=02
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
from utils.coordinate_alignment import load_absolute_first_pose


def transform_points_to_global(points, T_first):
    """
    Transform points from robot-relative frame to Isaac Sim global frame.

    Args:
        points: Nx3 array of points in robot frame
        T_first: 4x4 first pose matrix (robot's starting pose in global frame)

    Returns:
        Nx3 array of points in global frame
    """
    # Add homogeneous coordinate
    points_homo = np.hstack([points, np.ones((len(points), 1))])

    # Transform: global = T_first @ robot
    points_global = (T_first @ points_homo.T).T[:, :3]

    return points_global


def load_tiergraph(graph_path):
    """Load TierGraph JSON."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def build_object_hierarchy_map(tiergraph):
    """
    Build mapping from object IDs to their hierarchical assignments.

    Returns:
        dict: {object_id: {'zone': zone_id, 'section': section_id or None}}
    """
    obj_map = {}

    nodes = tiergraph.get('nodes', [])

    # Build parent lookups
    node_lookup = {node['id']: node for node in nodes}

    for node in nodes:
        if node['type'] == 'object':
            obj_id = node['id']
            obj_map[obj_id] = {'zone': None, 'section': None, 'shelf': None}

            # Traverse up the hierarchy to find zone and section
            current = node
            while current:
                parent_id = current.get('parent_id')
                if not parent_id:
                    break

                parent = node_lookup.get(parent_id)
                if not parent:
                    break

                if parent['type'] == 'zone':
                    obj_map[obj_id]['zone'] = parent['id']
                elif parent['type'] == 'section':
                    obj_map[obj_id]['section'] = parent['id']
                elif parent['type'] == 'shelf':
                    obj_map[obj_id]['shelf'] = parent['id']

                current = parent

    return obj_map


def get_hierarchical_colors(obj_map, num_objects):
    """
    Generate colors based on hierarchical assignments.

    Objects in same section get same color.
    Objects in same zone (but different sections) get similar hues.

    Returns:
        list: Colors for each object index
    """
    import colorsys

    # Zone to base hue mapping
    zone_hues = {
        'zone_storage': 0.55,      # Blue range
        'zone_receiving': 0.33,    # Green range
        'zone_packing': 0.08,      # Orange range
        'zone_pallet_truck': 0.83, # Purple range
        'zone_hub_robot': 0.66,    # Cyan range
        'zone_forklift': 0.16,     # Yellow range
        'zone_general': 0.0,       # Red range
    }

    # Group objects by section (or zone if no section)
    section_to_objects = {}
    zone_only_objects = {}
    unassigned_objects = []

    for obj_idx, (obj_id, assignment) in enumerate(obj_map.items()):
        if assignment['section']:
            # Has section assignment
            section_id = assignment['section']
            if section_id not in section_to_objects:
                section_to_objects[section_id] = []
            section_to_objects[section_id].append(obj_idx)
        elif assignment['zone']:
            # Has zone but no section
            zone_id = assignment['zone']
            if zone_id not in zone_only_objects:
                zone_only_objects[zone_id] = []
            zone_only_objects[zone_id].append(obj_idx)
        else:
            # Unassigned
            unassigned_objects.append(obj_idx)

    colors = [[0.5, 0.5, 0.5]] * num_objects  # Default gray

    # Color objects by section (same section = same color)
    section_ids = list(section_to_objects.keys())
    for i, section_id in enumerate(section_ids):
        # Extract zone from section_id (format: "shelf_X_section_Y_tier_Z")
        zone_id = 'zone_storage'  # Sections are only in storage zone
        base_hue = zone_hues.get(zone_id, 0.5)

        # Vary hue slightly for different sections in same zone
        hue = base_hue + (i * 0.05) % 0.15 - 0.075  # Slight variation
        saturation = 0.7
        value = 0.9

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        for obj_idx in section_to_objects[section_id]:
            colors[obj_idx] = list(rgb)

    # Color zone-only objects (different shade per zone)
    for zone_id, obj_indices in zone_only_objects.items():
        base_hue = zone_hues.get(zone_id, 0.5)

        # Spread objects in this zone across hue range
        for i, obj_idx in enumerate(obj_indices):
            hue = base_hue + (i * 0.08) % 0.2 - 0.1
            saturation = 0.5
            value = 0.8
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[obj_idx] = list(rgb)

    # Unassigned objects get gray
    for obj_idx in unassigned_objects:
        colors[obj_idx] = [0.4, 0.4, 0.4]

    return colors


def print_color_legend(obj_map, objects):
    """Print color legend showing hierarchical assignments."""
    # Count objects per zone and section
    zone_counts = {}
    section_counts = {}
    unassigned_count = 0

    for assignment in obj_map.values():
        if assignment['section']:
            section_id = assignment['section']
            section_counts[section_id] = section_counts.get(section_id, 0) + 1
        elif assignment['zone']:
            zone_id = assignment['zone']
            zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
        else:
            unassigned_count += 1

    print("\n" + "="*60)
    print("COLOR LEGEND")
    print("="*60)
    print("\nInfrastructure:")
    print("  Floor:      Dark gray mesh")
    print("  Zones:      Colored wireframe boxes")
    print("    - Storage zone:   Green")
    print("    - Receiving:      Orange")
    print("    - Packing:        Magenta")
    print("  Shelves:    Blue wireframe boxes")
    print("  Sections:   Yellow wireframe boxes")

    print(f"\nObjects (total: {len(objects)}):")
    print("  Boundaries: White wireframe boxes (NEW!)")
    print("  Point clouds colored by zone/section:")
    print("    - Same section → Same color (blue shades)")
    print("    - Same zone → Similar hue")
    print("    - Unassigned → Gray")

    if section_counts:
        print(f"\n  Objects in shelf sections: {sum(section_counts.values())}")
        top_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for section_id, count in top_sections:
            print(f"    {section_id}: {count} objects")

    if zone_counts:
        print(f"\n  Objects in zones (not in sections): {sum(zone_counts.values())}")
        for zone_id, count in sorted(zone_counts.items()):
            print(f"    {zone_id}: {count} objects")

    if unassigned_count > 0:
        print(f"\n  Unassigned objects: {unassigned_count}")

    print("="*60 + "\n")


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


def create_bbox_lineset(bounds, color=[1, 0, 0]):
    """Create Open3D lineset for a bounding box."""
    min_pt = bounds['min']
    max_pt = bounds['max']
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

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return lineset


def create_floor_mesh(floor_vertices, floor_z=-0.14, color=[0.5, 0.5, 0.5]):
    """Create Open3D mesh for floor polygon."""
    vertices_3d = []
    for v in floor_vertices:
        # Floor vertices are stored as [x, y] arrays
        if isinstance(v, (list, tuple)):
            vertices_3d.append([v[0], v[1], floor_z])
        else:
            vertices_3d.append([v['x'], v['y'], floor_z])

    vertices_3d = np.array(vertices_3d)

    # Create triangles for the floor polygon using fan triangulation
    # Assumes vertices are in order (convex or simple polygon)
    num_vertices = len(vertices_3d)
    triangles = []
    for i in range(1, num_vertices - 1):
        triangles.append([0, i, i + 1])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "="*60)
    print("TIERGRAPH VISUALIZATION (ISAAC SIM GLOBAL FRAME)")
    print("="*60 + "\n")

    # Paths
    result_path = Path(cfg.result_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    warehouse_layout_path = "/home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json"
    tiergraph_path = Path(cfg.scenegraph_path)

    # Load data
    print(f"Loading OpenGraph results from {result_path}...")
    objects, bg_objects = load_opengraph_results(result_path)
    print(f"  Loaded {len(objects)} objects")

    print(f"Loading TierGraph from {tiergraph_path}...")
    tiergraph = load_tiergraph(tiergraph_path)
    print(f"  Loaded {len(tiergraph.get('nodes', []))} nodes")

    print(f"Loading ORIGINAL warehouse layout (Isaac Sim global coordinates)...")
    with open(warehouse_layout_path, 'r') as f:
        warehouse_layout = json.load(f)

    # Load first pose for transforming point clouds
    T_first = load_absolute_first_pose(sequence_dir)
    has_first_pose = not np.allclose(T_first, np.eye(4))

    if has_first_pose:
        print(f"Found first_pose.txt - will transform point clouds to global frame")
        print(f"  Robot starting pose: [{T_first[0, 3]:.2f}, {T_first[1, 3]:.2f}, {T_first[2, 3]:.2f}]")
    else:
        print("WARNING: No first_pose.txt found - point clouds may not align with warehouse")

    # Compute Z offset for point clouds
    # In robot frame, floor is at Z≈-0.7m
    # In global frame, floor is at Z=-0.14m
    # We need to add the difference
    ROBOT_FLOOR_Z = -0.781  # From compute_floor_offset.py
    ISAAC_FLOOR_Z = -0.14
    Z_OFFSET = ISAAC_FLOOR_Z - ROBOT_FLOOR_Z  # ≈ +0.641m

    # Transform point clouds to global frame
    print("\nTransforming point clouds to global frame...")
    pcds = []
    for i, obj in enumerate(objects):
        # Get points in robot frame
        points_robot = np.asarray(obj['pcd'].points).copy()

        # Transform to global frame
        if has_first_pose:
            # Apply XY rotation/translation from first_pose
            points_global = transform_points_to_global(points_robot, T_first)
            # CRITICAL: Also apply Z offset correction
            # first_pose.txt has wrong Z (0.00m), but robot is actually 0.641m above floor
            # This brings floor from -0.781m to -0.14m in global frame
            points_global[:, 2] += Z_OFFSET
        else:
            # No first_pose - just apply Z offset
            points_global = points_robot.copy()
            points_global[:, 2] += Z_OFFSET

        # Create new point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_global)

        if obj['pcd'].has_colors():
            pcd.colors = obj['pcd'].colors

        pcds.append(pcd)

    # Build object hierarchy mapping
    print("\nBuilding hierarchical color scheme...")
    obj_hierarchy_map = build_object_hierarchy_map(tiergraph)
    instance_colors = get_hierarchical_colors(obj_hierarchy_map, len(objects))

    # Print color legend
    print_color_legend(obj_hierarchy_map, objects)

    # Create infrastructure bounding boxes from ORIGINAL warehouse layout
    print("Creating infrastructure visualization...")

    # 1. Create floor mesh
    floor_mesh = create_floor_mesh(
        warehouse_layout['floor']['vertices'],
        floor_z=-0.14,
        color=[0.3, 0.3, 0.3]
    )
    print(f"  Created floor mesh with {len(warehouse_layout['floor']['vertices'])} vertices")

    # 2. Create zone bounding boxes
    zone_boxes = []
    for zone in warehouse_layout['functional_zones']:
        # Compute Z bounds for this zone
        min_z = -0.14  # Floor level
        max_z = -0.14  # Start at floor, will update with shelf heights

        # Find highest shelf in this zone
        shelves = zone.get('shelves', {})
        if isinstance(shelves, dict):
            for shelf in shelves.values():
                shelf_max_z = shelf['bounds']['max'].get('z', min_z)
                max_z = max(max_z, shelf_max_z)

        # Create zone box with computed Z bounds
        zone_bounds = {
            'min': {
                'x': zone['bounds']['min']['x'],
                'y': zone['bounds']['min']['y'],
                'z': min_z
            },
            'max': {
                'x': zone['bounds']['max']['x'],
                'y': zone['bounds']['max']['y'],
                'z': max_z
            }
        }

        # Color zones distinctly
        zone_colors = {
            'zone_storage': [0, 1, 0],      # Green
            'zone_loading': [1, 0.5, 0],    # Orange
            'zone_unloading': [1, 0, 1],    # Magenta
            'zone_office': [0, 1, 1],       # Cyan
        }
        zone_color = zone_colors.get(zone['id'], [0.5, 0.5, 0.5])

        zone_lineset = create_bbox_lineset(zone_bounds, zone_color)
        zone_boxes.append(zone_lineset)

    print(f"  Created {len(zone_boxes)} zone boxes")

    # 3. Create shelf and section boxes (only for storage zone)
    shelf_boxes = []
    for zone in warehouse_layout['functional_zones']:
        if zone['id'] == 'zone_storage':  # Only show storage zone shelves
            shelves = zone.get('shelves', {})
            if isinstance(shelves, dict):
                for shelf_id, shelf in shelves.items():
                    lineset = create_bbox_lineset(shelf['bounds'], [0, 0, 1])
                    shelf_boxes.append(lineset)

                    # Create section boxes
                    sections = shelf.get('sections', {})
                    if isinstance(sections, dict):
                        for level_dict in sections.values():
                            if isinstance(level_dict, dict):
                                for section in level_dict.values():
                                    if isinstance(section, dict) and 'bounds' in section:
                                        sec_lineset = create_bbox_lineset(
                                            section['bounds'], [1, 1, 0]
                                        )
                                        shelf_boxes.append(sec_lineset)

    print(f"  Created {len(shelf_boxes)} shelf/section boxes")

    # 4. Create object bounding boxes
    object_boxes = []
    for i, pcd in enumerate(pcds):
        if len(pcd.points) > 0:
            # Compute axis-aligned bounding box from point cloud
            points = np.asarray(pcd.points)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            # Create bounds dict
            obj_bounds = {
                'min': {'x': min_pt[0], 'y': min_pt[1], 'z': min_pt[2]},
                'max': {'x': max_pt[0], 'y': max_pt[1], 'z': max_pt[2]}
            }

            # Create white wireframe for object boundary
            obj_lineset = create_bbox_lineset(obj_bounds, color=[1, 1, 1])
            object_boxes.append(obj_lineset)

    print(f"  Created {len(object_boxes)} object bounding boxes")

    # Initialize visualizer
    print("\nInitializing Open3D visualizer...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TierGraph (Isaac Sim Global Frame)", width=1920, height=1080)

    # Add geometries
    vis.add_geometry(floor_mesh)
    for box in zone_boxes:
        vis.add_geometry(box)
    for box in shelf_boxes:
        vis.add_geometry(box)
    for box in object_boxes:
        vis.add_geometry(box)
    for pcd in pcds:
        vis.add_geometry(pcd)

    # Color by instance
    for i, pcd in enumerate(pcds):
        pcd.paint_uniform_color(instance_colors[i])

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)

    print("Colored by instance")
    print("\n" + "="*60)
    print("VISUALIZATION READY")
    print("="*60)
    print("Close window when done (press Q or close button)")
    print("="*60 + "\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
