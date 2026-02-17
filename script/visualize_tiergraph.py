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
    # Zone to RGB color mapping (aligned with zone visualization colors)
    zone_base_colors = {
        'zone_storage': (0.0, 0.7, 0.0),      # Green
        'zone_receiving': (0.9, 0.5, 0.1),    # Orange
        'zone_packing': (0.9, 0.2, 0.9),      # Magenta
        'zone_pallet_truck': (0.9, 0.9, 0.2), # Yellow
        'zone_hub_robot': (0.2, 0.7, 0.9),    # Cyan
        'zone_forklift': (0.9, 0.6, 0.3),     # Brown
        'zone_general': (0.5, 0.5, 0.5),      # Gray
    }

    # Group objects by section (or zone if no section)
    section_to_objects = {}
    zone_only_objects = {}
    unassigned_objects = []

    for obj_id, assignment in obj_map.items():
        # Extract numeric index from object_id (e.g., "object_17" -> 17)
        # This matches the OpenGraph object index
        if obj_id.startswith('object_'):
            obj_idx = int(obj_id.split('_')[1])
        else:
            continue  # Skip non-object nodes

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

    # Color objects by section (sections are in storage zone, use green shades)
    section_ids = list(section_to_objects.keys())
    storage_base = zone_base_colors.get('zone_storage', (0.0, 0.7, 0.0))

    for i, section_id in enumerate(section_ids):
        # Vary brightness for different sections
        brightness = 0.5 + 0.5 * ((i % 10) / 10.0)  # Cycle through 10 brightness levels

        color = [c * brightness for c in storage_base]
        # Boost green channel slightly for visibility
        color[1] = min(1.0, color[1] * 1.2)
        # Clamp to [0, 1]
        color = [min(1.0, max(0.15, c)) for c in color]

        for obj_idx in section_to_objects[section_id]:
            colors[obj_idx] = color

    # Color zone-only objects using their zone's color with brightness variations
    for zone_id, obj_indices in zone_only_objects.items():
        base_color = zone_base_colors.get(zone_id, (0.5, 0.5, 0.5))

        for i, obj_idx in enumerate(obj_indices):
            # Vary brightness to distinguish objects within the same zone
            brightness = 0.6 + 0.4 * (i / max(len(obj_indices) - 1, 1))
            color = [c * brightness for c in base_color]
            # Clamp to [0, 1]
            color = [min(1.0, max(0.2, c)) for c in color]
            colors[obj_idx] = color

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
    print("  Floor:         Dark gray mesh")
    print("  Zone floors:   Semi-transparent colored polygons")
    print("  Zone boxes:    Colored wireframe boundaries")
    print("    - Storage zone:    Green (has shelf/aisle hierarchy)")
    print("    - Receiving:       Orange")
    print("    - Packing:         Magenta")
    print("    - Pallet Truck:    Yellow")
    print("    - Hub Robot:       Cyan")
    print("    - Forklift:        Brown")
    print("    - General:         Gray")
    print("  Aisles:        Yellow wireframe (storage zone only)")
    print("  Shelves:       Blue wireframe boxes (storage zone only)")
    print("  Sections:      Yellow/orange wireframe (storage zone only)")

    print(f"\nObjects (total: {len(objects)}):")
    print("  Centroid markers: Small white spheres")
    print("  Boundaries:       White wireframe boxes")
    print("  Point clouds colored by zone assignment:")
    print("    - Storage zone objects:  Green shades")
    print("    - Receiving zone:        Orange shades")
    print("    - Packing zone:          Magenta shades")
    print("    - Pallet Truck zone:     Yellow shades")
    print("    - Hub Robot zone:        Cyan shades")
    print("    - Forklift zone:         Brown shades")
    print("    - General zone:          Gray shades")
    print("    - Unassigned:            Dark gray")

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

    # 2. Create zone floor polygons and bounding boxes
    zone_boxes = []
    zone_floor_meshes = []
    zone_labels = []

    # Define colors for all zone types
    zone_color_map = {
        'zone_storage': ([0.0, 0.8, 0.0], [0.2, 1.0, 0.2]),      # Dark green / Light green
        'zone_receiving': ([0.8, 0.4, 0.0], [1.0, 0.6, 0.2]),    # Dark orange / Light orange
        'zone_packing': ([0.8, 0.0, 0.8], [1.0, 0.4, 1.0]),      # Dark magenta / Light magenta
        'zone_pallet_truck': ([0.8, 0.8, 0.0], [1.0, 1.0, 0.3]), # Dark yellow / Light yellow
        'zone_hub_robot': ([0.0, 0.6, 0.8], [0.3, 0.8, 1.0]),    # Dark cyan / Light cyan
        'zone_forklift': ([0.8, 0.5, 0.2], [1.0, 0.7, 0.4]),     # Dark brown / Light brown
        'zone_general': ([0.4, 0.4, 0.4], [0.6, 0.6, 0.6]),      # Dark gray / Light gray
    }

    # Use progressive Z offsets for each zone floor to prevent z-fighting
    zone_floor_z_offset = 0.005  # 5mm increments

    for zone_idx, zone in enumerate(warehouse_layout['functional_zones']):
        zone_id = zone['id']
        zone_name = zone['name']

        # Get zone colors (wireframe, floor fill)
        wireframe_color, fill_color = zone_color_map.get(zone_id, ([0.5, 0.5, 0.5], [0.7, 0.7, 0.7]))

        # Compute Z bounds for this zone
        min_z = -0.14  # Floor level
        max_z = -0.14  # Start at floor, will update with shelf heights

        # Find highest shelf in this zone (for storage zone visualization height)
        shelves = zone.get('shelves', {})
        if isinstance(shelves, dict):
            for shelf in shelves.values():
                shelf_max_z = shelf['bounds']['max'].get('z', min_z)
                max_z = max(max_z, shelf_max_z)

        # If no shelves, use default height for non-storage zones
        if max_z == min_z:
            max_z = min_z + 0.5  # 50cm tall wireframe for non-storage zones

        # Create zone floor polygon with progressive Z offset to avoid z-fighting
        zone_verts = zone.get('vertices', [])
        if zone_verts:
            # Each zone gets a unique Z offset (storage zone gets lowest to be most visible)
            if zone_id == 'zone_storage':
                z_offset = 0.001  # Storage zone closest to actual floor
            else:
                z_offset = 0.01 + (zone_idx * zone_floor_z_offset)

            zone_floor = create_floor_mesh(
                zone_verts,
                floor_z=min_z + z_offset,
                color=fill_color
            )
            zone_floor.paint_uniform_color(fill_color)
            zone_floor_meshes.append((zone_floor, zone_id))

        # Create zone bounding box wireframe (thick lines)
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

        # Special styling for storage zone (thicker, brighter)
        if zone_id == 'zone_storage':
            zone_lineset = create_bbox_lineset(zone_bounds, wireframe_color)
            # Make storage zone lines thicker (we'll need to modify this via Open3D)
        else:
            zone_lineset = create_bbox_lineset(zone_bounds, wireframe_color)

        zone_boxes.append((zone_lineset, zone_id, zone_name))

        # Create zone label (text)
        zone_center = zone['bounds']['center']
        zone_labels.append({
            'position': [zone_center['x'], zone_center['y'], max_z + 0.5],
            'text': zone_name,
            'color': wireframe_color
        })

    print(f"  Created {len(zone_boxes)} zone boxes + {len(zone_floor_meshes)} floor polygons")

    # 3. Create aisle, shelf, and section boxes (only for storage zone)
    aisle_boxes = []
    shelf_boxes = []

    for zone in warehouse_layout['functional_zones']:
        if zone['id'] == 'zone_storage':  # Only show storage zone infrastructure
            # Create aisle boxes (corridors between shelves)
            aisles = zone.get('aisles', {})
            if isinstance(aisles, dict):
                for aisle_id, aisle in aisles.items():
                    # Aisles shown as semi-transparent yellow boxes
                    aisle_lineset = create_bbox_lineset(aisle['bounds'], [0.9, 0.9, 0.0])
                    aisle_boxes.append(aisle_lineset)

            # Create shelf boxes (blue wireframes)
            shelves = zone.get('shelves', {})
            if isinstance(shelves, dict):
                for shelf_id, shelf in shelves.items():
                    lineset = create_bbox_lineset(shelf['bounds'], [0.2, 0.4, 1.0])
                    shelf_boxes.append(lineset)

                    # Create section boxes (yellow wireframes)
                    sections = shelf.get('sections', {})
                    if isinstance(sections, dict):
                        for level_dict in sections.values():
                            if isinstance(level_dict, dict):
                                for section in level_dict.values():
                                    if isinstance(section, dict) and 'bounds' in section:
                                        sec_lineset = create_bbox_lineset(
                                            section['bounds'], [1.0, 0.8, 0.0]
                                        )
                                        shelf_boxes.append(sec_lineset)

    print(f"  Created {len(aisle_boxes)} aisles + {len(shelf_boxes)} shelf/section boxes")

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

    # 5. Create object centroid markers (small spheres with object IDs)
    object_markers = []
    for i, obj in enumerate(objects):
        if len(obj['pcd'].points) > 0:
            # Compute centroid in global frame
            points_robot = np.asarray(obj['pcd'].points)
            if has_first_pose:
                points_global = transform_points_to_global(points_robot, T_first)
                points_global[:, 2] += Z_OFFSET
            else:
                points_global = points_robot.copy()
                points_global[:, 2] += Z_OFFSET

            centroid = points_global.mean(axis=0)

            # Create small sphere at centroid
            marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            marker_sphere.translate(centroid)
            marker_sphere.paint_uniform_color([1.0, 1.0, 1.0])  # White marker
            object_markers.append(marker_sphere)

    print(f"  Created {len(object_markers)} object centroid markers")

    # Print detailed object information for reference
    print("\n" + "="*60)
    print("OBJECT LABELS & HIERARCHY")
    print("="*60)
    for i, (obj_id, assignment) in enumerate(obj_hierarchy_map.items()):
        obj = objects[i]

        # Get centroid position
        points_robot = np.asarray(obj['pcd'].points)
        if has_first_pose:
            points_global = transform_points_to_global(points_robot, T_first)
            points_global[:, 2] += Z_OFFSET
        else:
            points_global = points_robot.copy()
            points_global[:, 2] += Z_OFFSET
        centroid = points_global.mean(axis=0)

        # Get hierarchy path
        path_parts = []
        if assignment['zone']:
            path_parts.append(assignment['zone'])
        if assignment.get('aisle'):
            path_parts.append(assignment['aisle'])
        if assignment.get('shelf'):
            path_parts.append(assignment['shelf'])
        if assignment['section']:
            path_parts.append(assignment['section'])
        path_parts.append(obj_id)

        hierarchy_path = " → ".join(path_parts) if path_parts else obj_id

        # Print with color indicator
        color = instance_colors[i]
        color_hex = f"RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"

        print(f"{obj_id}:")
        print(f"  Position: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        print(f"  Color: {color_hex}")
        print(f"  Hierarchy: {hierarchy_path}")
        print()

    print("="*60 + "\n")

    # Initialize visualizer
    print("\nInitializing Open3D visualizer...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TierGraph (Isaac Sim Global Frame)", width=1920, height=1080)

    # Add geometries in order (back to front for proper transparency)
    # 1. Floor
    vis.add_geometry(floor_mesh)

    # 2. Zone floor polygons (semi-transparent colored regions)
    for zone_floor, _ in zone_floor_meshes:
        vis.add_geometry(zone_floor)

    # 3. Zone bounding boxes (wireframes)
    for zone_lineset, _, _ in zone_boxes:
        vis.add_geometry(zone_lineset)

    # 4. Aisles (storage zone only)
    for aisle_box in aisle_boxes:
        vis.add_geometry(aisle_box)

    # 5. Shelves and sections (storage zone only)
    for shelf_box in shelf_boxes:
        vis.add_geometry(shelf_box)

    # 6. Object bounding boxes
    for obj_box in object_boxes:
        vis.add_geometry(obj_box)

    # 7. Object centroid markers (small white spheres)
    for marker in object_markers:
        vis.add_geometry(marker)

    # 8. Object point clouds (colored by hierarchy)
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
