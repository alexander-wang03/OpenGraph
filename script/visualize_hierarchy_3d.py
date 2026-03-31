#!/usr/bin/env python3
"""
3D Hierarchy Overlay Visualization — TierGraph

Renders the TierGraph hierarchy overlaid on the 3D point cloud map:
  - Small colored spheres at the center of each region (zone, shelf, section)
  - Edges drawn from parent sphere → child spheres → object point clouds
  - Object point clouds colored by zone assignment

Inspired by STAR paper figure. Supervisor explicitly requested this.

Outputs:
  - Interactive Open3D window
  - Static screenshot saved to results/warehouse_{seq}/pcd/hierarchy_3d_overlay.png

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/visualize_hierarchy_3d.py --config-name=isaac_warehouse sequence=03

Author: awang (TierGraph thesis)
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


# ---------------------------------------------------------------------------
# Zone colors (consistent with visualize_tiergraph.py)
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    'zone_storage':      [0.0, 0.8, 0.0],   # Green
    'zone_receiving':    [0.9, 0.5, 0.1],    # Orange
    'zone_staging':      [0.9, 0.2, 0.9],    # Magenta
    'zone_packing':      [0.9, 0.2, 0.9],    # Alias (legacy)
    'zone_pallet_truck': [0.9, 0.9, 0.2],    # Yellow
    'zone_hub_robot':    [0.2, 0.7, 0.9],    # Cyan
    'zone_forklift':     [0.85, 0.1, 0.1],   # Red
    'zone_general':      [0.5, 0.5, 0.5],    # Gray
}

# Hierarchy level sphere sizes and Z offsets
LEVEL_CONFIG = {
    'zone':    {'radius': 0.8,  'z_offset': 6.0, 'alpha': 0.9},
    'shelf':   {'radius': 0.5,  'z_offset': 4.5, 'alpha': 0.8},
    'section': {'radius': 0.3,  'z_offset': 3.5, 'alpha': 0.7},
    'aisle':   {'radius': 0.4,  'z_offset': 5.0, 'alpha': 0.8},
    'object':  {'radius': 0.15, 'z_offset': 0.0, 'alpha': 1.0},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tiergraph(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_objects(path: Path) -> MapObjectList:
    with gzip.open(path, 'rb') as f:
        results = pickle.load(f)
    objects = MapObjectList()
    if isinstance(results, dict):
        objects.load_serializable(results["objects"])
    elif isinstance(results, list):
        objects.load_serializable(results)
    return objects


def transform_to_global(points: np.ndarray, T_first: np.ndarray,
                         z_offset: float = 0.641) -> np.ndarray:
    """Transform points from robot frame to Isaac Sim global frame."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    pts_global = (T_first @ pts_h.T).T[:, :3]
    pts_global[:, 2] += z_offset
    return pts_global


# ---------------------------------------------------------------------------
# Build hierarchy graph for visualization
# ---------------------------------------------------------------------------

def build_hierarchy_viz(tiergraph: dict, objects: MapObjectList,
                        T_first: np.ndarray, use_global: bool = True):
    """
    Build visualization geometry for the hierarchy overlay.

    Returns:
        spheres: list of (mesh, node_id, node_type) for hierarchy nodes
        edges: list of (start_xyz, end_xyz, color) for parent→child edges
        object_pcds: list of (pcd, color, obj_id) for object point clouds
    """
    nodes = {n['id']: n for n in tiergraph['nodes']}
    z_off = 0.641 if use_global else 0.0

    spheres = []
    edges = []
    object_pcds = []

    # Compute object centroids in the target frame
    obj_centroids = {}
    for node in tiergraph['nodes']:
        if node['type'] != 'object':
            continue
        obj_id = node['id']
        parts = obj_id.split('_')
        if parts[0] != 'object' or len(parts) != 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx >= len(objects):
            continue

        pts = np.asarray(objects[idx]['pcd'].points)
        if len(pts) == 0:
            continue

        if use_global:
            pts_global = transform_to_global(pts, T_first, z_off)
            centroid = pts_global.mean(axis=0)
        else:
            centroid = pts.mean(axis=0)
            centroid[2] += 0.781  # robot-frame Z offset

        obj_centroids[obj_id] = centroid

    # For each non-object node, compute center from its children's centroids
    # (more accurate than bounds center, which may not reflect actual objects)
    node_centers = {}

    def get_node_center(node_id: str) -> np.ndarray:
        """Recursively compute node center from children."""
        if node_id in node_centers:
            return node_centers[node_id]

        node = nodes.get(node_id)
        if node is None:
            return None

        if node['type'] == 'object':
            center = obj_centroids.get(node_id)
            if center is not None:
                node_centers[node_id] = center
            return center

        # Compute from children
        children = node.get('children', [])
        child_centers = []
        for child_id in children:
            cc = get_node_center(child_id)
            if cc is not None:
                child_centers.append(cc)

        if child_centers:
            center = np.mean(child_centers, axis=0)
        else:
            # Fallback: use bounds center
            b = node.get('bounds', {}).get('center', {})
            if use_global:
                # Bounds are in robot-relative — need transform
                pt = np.array([[b.get('x', 0), b.get('y', 0), b.get('z', 0)]])
                center = transform_to_global(pt, T_first, z_off)[0]
            else:
                center = np.array([b.get('x', 0), b.get('y', 0),
                                   b.get('z', 0) + 0.781])

        node_centers[node_id] = center
        return center

    # Compute all centers
    for node_id in nodes:
        get_node_center(node_id)

    # Find the zone ancestor for each node (for coloring)
    def find_zone(node_id: str) -> str:
        node = nodes.get(node_id)
        if node is None:
            return None
        if node['type'] == 'zone':
            return node_id
        parent = node.get('parent_id')
        if parent:
            return find_zone(parent)
        return None

    # Build spheres for hierarchy nodes (zones, shelves, sections, aisles)
    for node in tiergraph['nodes']:
        nid = node['id']
        ntype = node['type']

        if ntype == 'object':
            continue  # objects shown as point clouds, not spheres

        center = node_centers.get(nid)
        if center is None:
            continue

        # Skip nodes with no object descendants
        children = node.get('children', [])
        if not children and ntype != 'zone':
            continue

        cfg = LEVEL_CONFIG.get(ntype, LEVEL_CONFIG['section'])
        zone_id = find_zone(nid) if ntype != 'zone' else nid
        color = ZONE_COLORS.get(zone_id, [0.5, 0.5, 0.5])

        # Place sphere above the region
        sphere_pos = center.copy()
        sphere_pos[2] += cfg['z_offset']

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cfg['radius'])
        sphere.translate(sphere_pos)
        # Lighten color for higher levels
        if ntype == 'zone':
            sphere_color = [min(1.0, c + 0.2) for c in color]
        else:
            sphere_color = color
        sphere.paint_uniform_color(sphere_color)
        sphere.compute_vertex_normals()
        spheres.append((sphere, nid, ntype))

    # Build edges: parent → child
    for node in tiergraph['nodes']:
        nid = node['id']
        ntype = node['type']
        children = node.get('children', [])

        parent_center = node_centers.get(nid)
        if parent_center is None:
            continue

        parent_cfg = LEVEL_CONFIG.get(ntype, LEVEL_CONFIG['section'])
        parent_pos = parent_center.copy()
        if ntype != 'object':
            parent_pos[2] += parent_cfg['z_offset']

        zone_id = find_zone(nid)
        color = ZONE_COLORS.get(zone_id, [0.5, 0.5, 0.5])

        for child_id in children:
            child_node = nodes.get(child_id)
            if child_node is None:
                continue

            child_center = node_centers.get(child_id)
            if child_center is None:
                continue

            child_cfg = LEVEL_CONFIG.get(child_node['type'],
                                          LEVEL_CONFIG['section'])
            child_pos = child_center.copy()
            if child_node['type'] != 'object':
                child_pos[2] += child_cfg['z_offset']

            edges.append((parent_pos.copy(), child_pos.copy(), color))

    # Build object point clouds
    for node in tiergraph['nodes']:
        if node['type'] != 'object':
            continue
        obj_id = node['id']
        parts = obj_id.split('_')
        if parts[0] != 'object' or len(parts) != 2:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx >= len(objects):
            continue

        pts = np.asarray(objects[idx]['pcd'].points)
        if len(pts) == 0:
            continue

        if use_global:
            pts = transform_to_global(pts, T_first, z_off)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        zone_id = find_zone(obj_id)
        color = ZONE_COLORS.get(zone_id, [0.5, 0.5, 0.5])
        pcd.paint_uniform_color(color)

        object_pcds.append((pcd, color, obj_id))

    return spheres, edges, object_pcds


def edges_to_lineset(edges: list) -> o3d.geometry.LineSet:
    """Convert list of (start, end, color) edges to a single LineSet."""
    if not edges:
        return None

    points = []
    lines = []
    colors = []
    for start, end, color in edges:
        idx = len(points)
        points.append(start)
        points.append(end)
        lines.append([idx, idx + 1])
        colors.append(color)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(colors))
    return ls


# ---------------------------------------------------------------------------
# Warehouse boundary helpers (shared logic with visualize_tiergraph.py)
# ---------------------------------------------------------------------------

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
        if isinstance(v, (list, tuple)):
            vertices_3d.append([v[0], v[1], floor_z])
        else:
            vertices_3d.append([v['x'], v['y'], floor_z])

    vertices_3d = np.array(vertices_3d)

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


ZONE_COLOR_MAP = {
    'zone_storage':      ([0.0, 0.8, 0.0], [0.2, 1.0, 0.2]),
    'zone_receiving':    ([0.8, 0.4, 0.0], [1.0, 0.6, 0.2]),
    'zone_staging':      ([0.8, 0.0, 0.8], [1.0, 0.4, 1.0]),
    'zone_packing':      ([0.8, 0.0, 0.8], [1.0, 0.4, 1.0]),
    'zone_pallet_truck': ([0.8, 0.8, 0.0], [1.0, 1.0, 0.3]),
    'zone_hub_robot':    ([0.0, 0.6, 0.8], [0.3, 0.8, 1.0]),
    'zone_forklift':     ([0.7, 0.0, 0.0], [1.0, 0.3, 0.3]),
    'zone_general':      ([0.4, 0.4, 0.4], [0.6, 0.6, 0.6]),
}


def build_warehouse_boundaries(warehouse_layout):
    """Build zone/aisle/shelf boundary geometries from warehouse_layout.json.

    Returns a list of Open3D geometries (linesets + floor meshes).
    """
    geometries = []
    zone_floor_z_offset = 0.005

    for zone_idx, zone in enumerate(warehouse_layout['functional_zones']):
        zone_id = zone['id']
        wireframe_color, fill_color = ZONE_COLOR_MAP.get(
            zone_id, ([0.5, 0.5, 0.5], [0.7, 0.7, 0.7]))

        min_z = -0.14
        max_z = -0.14

        shelves = zone.get('shelves', {})
        if isinstance(shelves, dict):
            for shelf in shelves.values():
                shelf_max_z = shelf['bounds']['max'].get('z', min_z)
                max_z = max(max_z, shelf_max_z)

        if max_z == min_z:
            max_z = min_z + 0.5

        # Zone bounding box wireframe
        zone_bounds = {
            'min': {'x': zone['bounds']['min']['x'],
                    'y': zone['bounds']['min']['y'], 'z': min_z},
            'max': {'x': zone['bounds']['max']['x'],
                    'y': zone['bounds']['max']['y'], 'z': max_z},
        }
        geometries.append(create_bbox_lineset(zone_bounds, wireframe_color))

        # Aisle + shelf + section boxes (storage zone only)
        if zone_id == 'zone_storage':
            aisles = zone.get('aisles', {})
            if isinstance(aisles, dict):
                for aisle in aisles.values():
                    geometries.append(
                        create_bbox_lineset(aisle['bounds'], [0.9, 0.9, 0.0]))

            if isinstance(shelves, dict):
                for shelf in shelves.values():
                    geometries.append(
                        create_bbox_lineset(shelf['bounds'], [0.2, 0.4, 1.0]))
                    sections = shelf.get('sections', {})
                    if isinstance(sections, dict):
                        for level_dict in sections.values():
                            if isinstance(level_dict, dict):
                                for section in level_dict.values():
                                    if isinstance(section, dict) and 'bounds' in section:
                                        geometries.append(
                                            create_bbox_lineset(
                                                section['bounds'],
                                                [1.0, 0.8, 0.0]))

    return geometries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("3D HIERARCHY OVERLAY VISUALIZATION")
    print("=" * 60 + "\n")

    # Paths
    result_path = Path(cfg.result_path)
    tiergraph_path = Path(cfg.scenegraph_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    output_png = tiergraph_path.parent / "hierarchy_3d_overlay.png"

    # Load data
    print(f"Loading TierGraph from {tiergraph_path}...")
    tiergraph = load_tiergraph(tiergraph_path)
    stats = tiergraph.get('statistics', {})
    print(f"  {stats.get('total_nodes', '?')} nodes, "
          f"{stats.get('nodes_by_type', {}).get('object', '?')} objects")

    print(f"Loading objects from {result_path}...")
    objects = load_objects(result_path)
    print(f"  {len(objects)} objects in full_pcd")

    warehouse_layout_path = (
        Path(__file__).parent.parent / "data" / "warehouse_layout" / "warehouse_layout.json"
    )
    print(f"Loading warehouse layout from {warehouse_layout_path}...")
    with open(warehouse_layout_path, 'r') as f:
        warehouse_layout = json.load(f)

    T_first = load_absolute_first_pose(sequence_dir)
    has_first_pose = not np.allclose(T_first, np.eye(4))
    print(f"  First pose: {'found' if has_first_pose else 'identity (no transform)'}")

    # Build hierarchy visualization
    print("\nBuilding 3D hierarchy overlay...")
    spheres, edges, object_pcds = build_hierarchy_viz(
        tiergraph, objects, T_first, use_global=has_first_pose)

    print(f"  {len(spheres)} hierarchy spheres")
    print(f"  {len(edges)} hierarchy edges")
    print(f"  {len(object_pcds)} object point clouds")

    # Count by type
    type_counts = {}
    for _, _, ntype in spheres:
        type_counts[ntype] = type_counts.get(ntype, 0) + 1
    for ntype, count in sorted(type_counts.items()):
        print(f"    {ntype}: {count} spheres")

    # Build edge LineSet
    edge_lineset = edges_to_lineset(edges)

    # Assemble geometries
    geometries = []

    # Add object point clouds
    for pcd, _, obj_id in object_pcds:
        geometries.append(pcd)

    # Add hierarchy spheres
    for sphere, nid, ntype in spheres:
        geometries.append(sphere)

    # Add edges
    if edge_lineset is not None:
        geometries.append(edge_lineset)

    # Add warehouse/zone boundaries
    print("Creating warehouse boundary visualization...")
    boundary_geoms = build_warehouse_boundaries(warehouse_layout)
    geometries.extend(boundary_geoms)
    print(f"  Added {len(boundary_geoms)} boundary geometries")

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # Print legend
    print("\n" + "=" * 60)
    print("COLOR LEGEND")
    print("=" * 60)
    print("Hierarchy spheres (elevated above scene):")
    print("  Large spheres  = zones")
    print("  Medium spheres = shelves / aisles")
    print("  Small spheres  = sections")
    print("Edges connect parent → child nodes")
    print("Object point clouds colored by zone:")
    for zone_id, color in sorted(ZONE_COLORS.items()):
        if zone_id == 'zone_packing':
            continue  # skip legacy alias
        print(f"  {zone_id}: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
    print("=" * 60 + "\n")

    # Visualize
    print("Launching Open3D visualizer...")
    print("  - Rotate: left-click drag")
    print("  - Pan: middle-click drag")
    print("  - Zoom: scroll wheel")
    print("  - Close window to exit")

    vis = o3d.visualization.Visualizer()
    vis.create_window("TierGraph 3D Hierarchy Overlay", width=1920, height=1080)

    for geom in geometries:
        vis.add_geometry(geom)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.line_width = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.1])  # dark background

    # Set camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)

    # Run
    vis.run()

    # Save screenshot before closing
    vis.capture_screen_image(str(output_png), do_render=True)
    print(f"\nScreenshot saved to: {output_png}")

    vis.destroy_window()
    print("Done.")


if __name__ == "__main__":
    main()
