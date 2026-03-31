#!/usr/bin/env python3
"""
3D Visualization of IB-Compressed TierGraph

Renders the compressed TierGraph with cluster-colored point clouds:
  - Each cluster's member objects are merged into one point cloud with a
    distinct color
  - Singleton (unmerged) objects get their own color
  - Warehouse zone/aisle/shelf/section boundaries shown as wireframes
  - Click a centroid sphere to highlight a cluster and print its members

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/visualize_compressed.py --config-name=isaac_warehouse sequence=03

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
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path
from omegaconf import DictConfig
from some_class.map_calss import MapObjectList
from utils.coordinate_alignment import load_absolute_first_pose


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_FLOOR_Z = -0.781
ISAAC_FLOOR_Z = -0.14
Z_OFFSET = ISAAC_FLOOR_Z - ROBOT_FLOOR_Z  # +0.641m

# Zone base colors for cluster/object coloring
ZONE_BASE_COLORS = {
    'zone_storage':      np.array([0.0, 0.8, 0.0]),   # Green
    'zone_receiving':    np.array([0.9, 0.5, 0.1]),    # Orange
    'zone_staging':      np.array([0.9, 0.2, 0.9]),    # Magenta
    'zone_packing':      np.array([0.9, 0.2, 0.9]),    # Magenta (alias)
    'zone_pallet_truck': np.array([0.9, 0.9, 0.2]),    # Yellow
    'zone_hub_robot':    np.array([0.2, 0.7, 0.9]),    # Cyan
    'zone_forklift':     np.array([0.85, 0.1, 0.1]),   # Red
    'zone_general':      np.array([0.5, 0.5, 0.5]),    # Gray
}

# Wireframe colors for boundaries (darker variant, same hue)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_compressed_tiergraph(path: Path) -> dict:
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


def transform_points_to_global(points: np.ndarray, T_first: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    pts_global = (T_first @ pts_h.T).T[:, :3]
    pts_global[:, 2] += Z_OFFSET
    return pts_global


def parse_obj_index(obj_id: str):
    """Parse 'object_17' → 17, return None if invalid."""
    parts = obj_id.split('_')
    if parts[0] != 'object' or len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Warehouse boundary helpers
# ---------------------------------------------------------------------------

def create_bbox_lineset(bounds, color=[1, 0, 0]):
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
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def build_warehouse_boundaries(warehouse_layout):
    geometries = []
    for zone_idx, zone in enumerate(warehouse_layout['functional_zones']):
        zone_id = zone['id']
        wireframe_color, _ = ZONE_COLOR_MAP.get(
            zone_id, ([0.5, 0.5, 0.5], [0.7, 0.7, 0.7]))

        min_z = -0.14
        max_z = -0.14
        shelves = zone.get('shelves', {})
        if isinstance(shelves, dict):
            for shelf in shelves.values():
                max_z = max(max_z, shelf['bounds']['max'].get('z', min_z))
        if max_z == min_z:
            max_z = min_z + 0.5

        zone_bounds = {
            'min': {'x': zone['bounds']['min']['x'],
                    'y': zone['bounds']['min']['y'], 'z': min_z},
            'max': {'x': zone['bounds']['max']['x'],
                    'y': zone['bounds']['max']['y'], 'z': max_z},
        }
        geometries.append(create_bbox_lineset(zone_bounds, wireframe_color))

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
# Build cluster visualization data
# ---------------------------------------------------------------------------

def build_cluster_viz(graph: dict, objects: MapObjectList, T_first: np.ndarray,
                      has_first_pose: bool):
    """
    Build merged point clouds for clusters and singletons.

    Returns:
        entries: list of dicts with keys:
            id, type ('cluster' or 'object'), name, color, pcd, centroid,
            member_ids, parent_id
    """
    nodes = {n['id']: n for n in graph['nodes']}
    entries = []

    # Collect all cluster and singleton object nodes
    cluster_nodes = [n for n in graph['nodes'] if n['type'] == 'cluster']
    object_nodes = [n for n in graph['nodes'] if n['type'] == 'object']

    def find_zone(node_id: str) -> str:
        """Walk up parent_id chain to find the zone ancestor."""
        visited = set()
        nid = node_id
        while nid and nid not in visited:
            visited.add(nid)
            node = nodes.get(nid)
            if node is None:
                return None
            if node['type'] == 'zone':
                return nid
            nid = node.get('parent_id')
        return None

    def zone_color_variant(zone_id: str, index: int, total: int) -> list:
        """Return a brightness-varied color based on zone, differentiated
        by index within that zone."""
        base = ZONE_BASE_COLORS.get(zone_id, np.array([0.5, 0.5, 0.5]))
        # Vary brightness between 0.5 and 1.0
        if total <= 1:
            brightness = 0.75
        else:
            brightness = 0.5 + 0.5 * (index / (total - 1))
        color = np.clip(base * brightness, 0.15, 1.0)
        return color.tolist()

    def get_merged_pcd(obj_ids, color):
        """Merge point clouds for a list of object IDs, color uniformly."""
        all_pts = []
        for oid in obj_ids:
            idx = parse_obj_index(oid)
            if idx is None or idx >= len(objects):
                continue
            pts = np.asarray(objects[idx]['pcd'].points)
            if len(pts) == 0:
                continue
            if has_first_pose:
                pts = transform_points_to_global(pts, T_first)
            else:
                pts = pts.copy()
                pts[:, 2] += Z_OFFSET
            all_pts.append(pts)

        if not all_pts:
            return o3d.geometry.PointCloud(), np.zeros(3)

        merged = np.vstack(all_pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd.paint_uniform_color(color)
        centroid = merged.mean(axis=0)
        return pcd, centroid

    # Group all renderable nodes (clusters + singletons) by zone
    all_nodes_to_render = []
    for cnode in cluster_nodes:
        all_nodes_to_render.append(('cluster', cnode))
    for onode in object_nodes:
        idx = parse_obj_index(onode['id'])
        if idx is not None and idx < len(objects):
            all_nodes_to_render.append(('object', onode))

    # Count entries per zone for brightness distribution
    zone_counts = {}
    zone_indices = {}
    for ntype, node in all_nodes_to_render:
        zone_id = find_zone(node['id']) or 'unknown'
        zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
        zone_indices[node['id']] = zone_counts[zone_id] - 1

    # Build entries with zone-based colors
    for ntype, node in all_nodes_to_render:
        zone_id = find_zone(node['id']) or 'unknown'
        idx_in_zone = zone_indices[node['id']]
        total_in_zone = zone_counts[zone_id]
        color = zone_color_variant(zone_id, idx_in_zone, total_in_zone)

        if ntype == 'cluster':
            member_ids = node.get('metadata', {}).get('member_ids', [])
        else:
            member_ids = [node['id']]

        pcd, centroid = get_merged_pcd(member_ids, color)

        entries.append({
            'id': node['id'],
            'type': ntype,
            'name': node.get('name', node['id']),
            'color': color,
            'pcd': pcd,
            'centroid': centroid,
            'member_ids': member_ids,
            'member_count': len(member_ids),
            'parent_id': node.get('parent_id', ''),
            'zone_id': zone_id,
        })

    return entries


# ---------------------------------------------------------------------------
# Interactive viewer (GUI-based, with click-to-highlight)
# ---------------------------------------------------------------------------

class CompressedViewer:
    SPHERE_RADIUS = 0.12
    HIT_RADIUS = 0.30

    def __init__(self, entries, infra_geometries):
        self._entries = entries
        self._infra = infra_geometries
        self._selected = -1
        self._click_start = None

        # Pre-build normal/highlighted PCD variants
        self._pcd_normal = []
        self._pcd_highlight = []
        for e in entries:
            self._pcd_normal.append(e['pcd'])
            # Highlighted = brighter version
            bright_pcd = o3d.geometry.PointCloud(e['pcd'])
            base = np.asarray(e['color'])
            lighter = base + (1.0 - base) * 0.5
            lighter = np.clip(lighter, 0.0, 1.0)
            bright_pcd.paint_uniform_color(lighter.tolist())
            self._pcd_highlight.append(bright_pcd)

    def run(self):
        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Compressed TierGraph", 1920, 1080)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._window.renderer)
        self._scene.scene.set_background([0.05, 0.05, 0.1, 1.0])

        # Info label
        self._label = gui.Label("")
        self._label.text_color = gui.Color(1, 1, 1)

        # Layout
        self._window.add_child(self._scene)
        self._window.add_child(self._label)
        self._window.set_on_layout(self._on_layout)

        # Add infrastructure
        for name, geom, shader in self._infra:
            try:
                mat = rendering.MaterialRecord()
                mat.shader = shader
                if shader == "unlitLine":
                    mat.line_width = 1.5
                self._scene.scene.add_geometry(name, geom, mat)
            except RuntimeError as e:
                print(f"  [WARNING] Could not add {name}: {e}")

        # Add cluster/object PCDs and markers
        for i, e in enumerate(self._entries):
            # Point cloud
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 3.0
            pcd = self._pcd_normal[i]
            if len(pcd.points) > 0:
                self._scene.scene.add_geometry(f"pcd_{i}", pcd, mat)

            # Centroid sphere
            c = e['centroid']
            if np.any(c != 0):
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.SPHERE_RADIUS)
                sphere.translate(c)
                sphere.paint_uniform_color(e['color'])
                sphere.compute_vertex_normals()
                smat = rendering.MaterialRecord()
                smat.shader = "defaultLit"
                self._scene.scene.add_geometry(f"marker_{i}", sphere, smat)

        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0, origin=[0, 0, 0])
        cmat = rendering.MaterialRecord()
        cmat.shader = "defaultLit"
        self._scene.scene.add_geometry("coord_frame", coord, cmat)

        # Fit camera
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

        # Events
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)

        self._label.text = ("Click a sphere to inspect cluster  |  "
                            "Press Esc to deselect  |  Close window to exit")

        print("\n" + "=" * 60)
        print("INTERACTIVE VIEWER")
        print("=" * 60)
        print("Click centroid sphere → highlight cluster + show members")
        print("Esc → deselect  |  Close window → exit")
        print("=" * 60 + "\n")

        app.run()

    def _on_layout(self, ctx):
        r = self._window.content_rect
        label_h = 30
        self._scene.frame = gui.Rect(r.x, r.y, r.width, r.height - label_h)
        self._label.frame = gui.Rect(r.x, r.y + r.height - label_h,
                                     r.width, label_h)

    # --- Event handlers ---

    def _on_key(self, event):
        if (event.type == gui.KeyEvent.DOWN and
                event.key == gui.KeyName.ESCAPE):
            self._deselect()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse(self, event):
        has_ctrl = (event.is_modifier_down(gui.KeyModifier.CTRL)
                    or event.is_modifier_down(gui.KeyModifier.META))
        if has_ctrl:
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.buttons & int(gui.MouseButton.LEFT):
                self._click_start = (event.x, event.y)
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self._click_start is not None:
                sx, sy = self._click_start
                dx = event.x - sx
                dy = event.y - sy
                self._click_start = None
                if dx * dx + dy * dy < 25:
                    hit = self._pick(sx, sy)
                    print(f"[DBG] click at ({sx},{sy})  pick={hit}")
                    if hit >= 0:
                        self._select(hit)
                    else:
                        self._deselect()

        return gui.Widget.EventCallbackResult.IGNORED

    def _pick(self, wx, wy):
        """Return index of entry hit by click at window pos (wx, wy), else -1."""
        r = self._scene.frame
        sx = wx - r.x
        sy_gui = wy - r.y
        W, H = r.width, r.height

        cam = self._scene.scene.camera

        # Camera eye in world space (from view matrix: eye = -R^T @ t)
        V = np.array(cam.get_view_matrix())
        eye = -(V[:3, :3].T @ V[:3, 3])

        # Screen pixel → NDC (OpenGL: y=-1 bottom, +1 top)
        ndc_x = 2.0 * sx / W - 1.0
        ndc_y = 1.0 - 2.0 * sy_gui / H

        # Unproject through inverse P and V to get world-space ray point
        P = np.array(cam.get_projection_matrix())
        P_inv = np.linalg.inv(P)
        V_inv = np.linalg.inv(V)

        p_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        p_view_hom = P_inv @ p_clip
        p_view = p_view_hom[:3] / p_view_hom[3]

        p_world_hom = V_inv @ np.array([p_view[0], p_view[1], p_view[2], 1.0])
        p_world = p_world_hom[:3] / p_world_hom[3]

        ray_dir = p_world - eye
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-8:
            return -1
        ray_dir /= norm

        R2 = self.HIT_RADIUS ** 2
        best_idx, best_t = -1, float('inf')

        for i, e in enumerate(self._entries):
            c = e['centroid']
            if np.all(c == 0):
                continue
            oc = eye - c
            b = 2.0 * np.dot(ray_dir, oc)
            cc = np.dot(oc, oc) - R2
            disc = b * b - 4.0 * cc
            if disc < 0:
                continue
            t = (-b - np.sqrt(disc)) / 2.0
            if 0 < t < best_t:
                best_t = t
                best_idx = i
        return best_idx

    def _select(self, idx):
        self._deselect()
        self._selected = idx
        e = self._entries[idx]

        # Highlight selected PCD
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0
        self._scene.scene.remove_geometry(f"pcd_{idx}")
        bright = self._pcd_highlight[idx]
        if len(bright.points) > 0:
            self._scene.scene.add_geometry(f"pcd_{idx}", bright, mat)

        # Enlarge marker
        c = e['centroid']
        if np.any(c != 0):
            self._scene.scene.remove_geometry(f"marker_{idx}")
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.SPHERE_RADIUS * 1.5)
            sphere.translate(c)
            sphere.paint_uniform_color([1.0, 0.85, 0.0])
            sphere.compute_vertex_normals()
            smat = rendering.MaterialRecord()
            smat.shader = "defaultLit"
            self._scene.scene.add_geometry(f"marker_{idx}", sphere, smat)

        # Dim all others
        dim_mat = rendering.MaterialRecord()
        dim_mat.shader = "defaultUnlit"
        dim_mat.point_size = 2.0
        for j, ej in enumerate(self._entries):
            if j == idx:
                continue
            self._scene.scene.remove_geometry(f"pcd_{j}")
            pcd = self._pcd_normal[j]
            if len(pcd.points) > 0:
                dimmed = o3d.geometry.PointCloud(pcd)
                dimmed.paint_uniform_color([0.15, 0.15, 0.15])
                self._scene.scene.add_geometry(f"pcd_{j}", dimmed, dim_mat)

        # Update label
        members_str = ", ".join(e['member_ids'][:8])
        if e['member_count'] > 8:
            members_str += f", ... ({e['member_count']} total)"
        label = (f"[{e['id']}] \"{e['name']}\"  |  "
                 f"parent: {e['parent_id']}  |  "
                 f"members: {members_str}  |  "
                 f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
        self._label.text = label

        # Print details to console
        print(f"\n--- Selected: {e['id']} ---")
        print(f"  Name:    {e['name']}")
        print(f"  Type:    {e['type']}")
        print(f"  Parent:  {e['parent_id']}")
        print(f"  Members ({e['member_count']}): {', '.join(e['member_ids'])}")
        print(f"  Color:   {e['color']}")
        print(f"  Centroid: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

    def _deselect(self):
        if self._selected < 0:
            return

        # Restore all PCDs to normal
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        for i, e in enumerate(self._entries):
            self._scene.scene.remove_geometry(f"pcd_{i}")
            pcd = self._pcd_normal[i]
            if len(pcd.points) > 0:
                self._scene.scene.add_geometry(f"pcd_{i}", pcd, mat)

            # Restore marker
            c = e['centroid']
            if np.any(c != 0):
                self._scene.scene.remove_geometry(f"marker_{i}")
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.SPHERE_RADIUS)
                sphere.translate(c)
                sphere.paint_uniform_color(e['color'])
                sphere.compute_vertex_normals()
                smat = rendering.MaterialRecord()
                smat.shader = "defaultLit"
                self._scene.scene.add_geometry(f"marker_{i}", sphere, smat)

        self._selected = -1
        self._label.text = ("Click a sphere to inspect cluster  |  "
                            "Press Esc to deselect  |  Close window to exit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config",
            config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("IB-COMPRESSED TIERGRAPH 3D VISUALIZATION")
    print("=" * 60 + "\n")

    # Paths
    result_path = Path(cfg.result_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    tiergraph_path = Path(cfg.scenegraph_path)
    compressed_path = tiergraph_path.parent / "object_relations_compressed.json"
    warehouse_layout_path = (
        Path(__file__).parent.parent / "data" / "warehouse_layout"
        / "warehouse_layout.json"
    )

    if not compressed_path.exists():
        print(f"ERROR: Compressed TierGraph not found at {compressed_path}")
        print("Run compress_tiergraph.py first.")
        return

    # Load data
    print(f"Loading compressed TierGraph from {compressed_path}...")
    graph = load_compressed_tiergraph(compressed_path)
    stats = graph.get('statistics', {})
    by_type = stats.get('nodes_by_type', {})
    compression = graph.get('metadata', {}).get('compression', {})
    print(f"  {by_type.get('cluster', 0)} clusters, "
          f"{by_type.get('object', 0)} singletons  "
          f"(from {compression.get('original_object_count', '?')} objects)")

    print(f"Loading objects from {result_path}...")
    objects = load_objects(result_path)
    print(f"  {len(objects)} objects in full_pcd")

    T_first = load_absolute_first_pose(sequence_dir)
    has_first_pose = not np.allclose(T_first, np.eye(4))
    print(f"  First pose: {'found' if has_first_pose else 'identity'}")

    print(f"Loading warehouse layout...")
    with open(warehouse_layout_path, 'r') as f:
        warehouse_layout = json.load(f)

    # Build cluster visualization
    print("\nBuilding cluster visualizations...")
    entries = build_cluster_viz(graph, objects, T_first, has_first_pose)

    n_clusters = sum(1 for e in entries if e['type'] == 'cluster')
    n_singletons = sum(1 for e in entries if e['type'] == 'object')
    print(f"  {n_clusters} cluster PCDs (merged)")
    print(f"  {n_singletons} singleton PCDs")

    # Build boundaries
    print("Building warehouse boundaries...")
    boundary_geoms = build_warehouse_boundaries(warehouse_layout)
    infra = []
    for bi, bg in enumerate(boundary_geoms):
        infra.append((f"boundary_{bi}", bg, "unlitLine"))
    print(f"  {len(infra)} boundary geometries")

    # Print cluster legend grouped by zone
    print("\n" + "=" * 60)
    print("CLUSTER COLOR LEGEND (grouped by zone)")
    print("=" * 60)
    # Group entries by zone
    from collections import defaultdict
    by_zone = defaultdict(list)
    for e in entries:
        by_zone[e.get('zone_id', 'unknown')].append(e)
    for zone_id in sorted(by_zone.keys()):
        zone_entries = by_zone[zone_id]
        print(f"\n  {zone_id}:")
        for e in zone_entries:
            c = e['color']
            tag = "cluster" if e['type'] == 'cluster' else "singleton"
            members = ", ".join(e['member_ids'][:5])
            if e['member_count'] > 5:
                members += f", ... ({e['member_count']} total)"
            print(f"    {e['id']:>12s}  ({tag:>9s})  "
                  f"\"{e['name'][:35]}\"  ← {members}")
    print("=" * 60 + "\n")

    # Launch viewer
    print("Launching interactive viewer...")
    viewer = CompressedViewer(entries, infra)
    viewer.run()
    print("Done.")


if __name__ == "__main__":
    main()
