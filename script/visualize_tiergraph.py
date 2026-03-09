#!/usr/bin/env python3
"""
Visualize TierGraph in Isaac Sim GLOBAL coordinates (not robot-relative).

This version visualizes the warehouse layout in the original Isaac Sim global
frame, making it match the warehouse_layout.json visualization. Point clouds
are transformed from robot frame back to global frame.

Click on a orange centroid sphere to highlight the associated point cloud.
Press Esc to deselect.

Usage:
    python script/visualize_tiergraph_global.py --config-name=isaac_warehouse sequence=02
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import copy
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
        'zone_staging': (0.9, 0.2, 0.9),       # Magenta
        'zone_packing': (0.9, 0.2, 0.9),       # Alias for zone_staging (legacy tiergraph results)
        'zone_pallet_truck': (0.9, 0.9, 0.2), # Yellow
        'zone_hub_robot': (0.2, 0.7, 0.9),    # Cyan
        'zone_forklift': (0.85, 0.1, 0.1),    # Red
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
    print("  Zone floors:   Transparent (not rendered)")
    print("  Zone boxes:    Colored wireframe boundaries")
    print("    - Storage zone:    Green (has shelf/aisle hierarchy)")
    print("    - Receiving:       Orange")
    print("    - Staging:         Magenta")
    print("    - Pallet Truck:    Yellow")
    print("    - Hub Robot:       Cyan")
    print("    - Forklift:        Brown")
    print("    - General:         Gray")
    print("  Aisles:        Yellow wireframe (storage zone only)")
    print("  Shelves:       Blue wireframe boxes (storage zone only)")
    print("  Sections:      Yellow/orange wireframe (storage zone only)")

    print(f"\nObjects (total: {len(objects)}):")
    print("  Centroid markers: Small orange spheres (click to highlight)")
    print("  Boundaries:       White wireframe boxes")
    print("  Point clouds colored by zone assignment:")
    print("    - Storage zone objects:  Green shades")
    print("    - Receiving zone:        Orange shades")
    print("    - Staging zone:          Magenta shades")
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


class TierGraphViewer:
    """
    Interactive TierGraph viewer with click-to-highlight support.

    Click a orange centroid sphere to highlight the selected object's point
    cloud (shows original RGB) and dim all others.  Press Esc to deselect.
    """

    SPHERE_RADIUS = 0.10   # centroid marker radius (m)
    HIT_RADIUS    = 0.25   # slightly larger hit sphere for easier clicking

    def __init__(self, pcds, instance_colors, centroids,
                 obj_hierarchy_map, objects, infra_geometries):
        """
        Args:
            pcds: list of o3d.PointCloud (global frame, colors may be set)
            instance_colors: list of [r,g,b] per object (hierarchy colors)
            centroids: list of np.ndarray shape (3,), one per object
            obj_hierarchy_map: {obj_id: {zone, shelf, section, aisle}}
            objects: MapObjectList (for captions)
            infra_geometries: list of (name: str, geom, shader: str)
        """
        self.pcds = pcds
        self.instance_colors = instance_colors
        self.centroids = centroids
        self.obj_hierarchy_map = obj_hierarchy_map
        self.objects = objects
        self.n = len(pcds)
        self.selected = -1
        self._click_start = None

        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window(
            "TierGraph Visualization (Isaac Sim Global Frame)", 1920, 1080)
        em = self.win.theme.font_size

        # ---------- info label overlaid at the top ----------
        self.info_label = gui.Label(
            "Click a orange centroid sphere to inspect an object    |    Esc = deselect")

        panel = gui.Horiz(0, gui.Margins(int(0.5 * em), int(0.3 * em),
                                         int(0.5 * em), int(0.3 * em)))
        panel.background_color = gui.Color(0, 0, 0, 0.75)
        panel.add_child(self.info_label)
        self._panel = panel

        # ---------- 3-D scene widget ----------
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.win.renderer)

        self.win.add_child(self._scene)
        self.win.add_child(panel)

        def on_layout(ctx):
            r = self.win.content_rect
            self._scene.frame = r
            pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
            panel.frame = gui.Rect(r.x, r.y, r.width, pref.height)

        self.win.set_on_layout(on_layout)
        self.win.set_on_key(self._on_key)
        self._scene.set_on_mouse(self._on_mouse)

        # ---------- materials (kept as instance vars for use in select/deselect) ----------
        self._pcd_mat = rendering.MaterialRecord()
        self._pcd_mat.shader = "defaultUnlit"
        self._pcd_mat.point_size = 4.0

        self._mesh_mat = rendering.MaterialRecord()
        self._mesh_mat.shader = "defaultUnlit"

        # ---------- infrastructure geometries (static) ----------
        for name, geom, shader in infra_geometries:
            mat = rendering.MaterialRecord()
            mat.shader = shader
            if shader == "unlitLine":
                mat.line_width = 1.5
            elif shader == "defaultLit":
                mat.shader = "defaultLit"
            self._scene.scene.add_geometry(name, geom, mat)

        # ---------- per-object geometry ----------
        # Store all variants as instance variables; only the 'normal' variant
        # is in the scene initially.  _select/_deselect swap with remove+add,
        # which is more reliable than show_geometry toggling.
        self._pcd_variants    = {}   # i -> {'normal', 'highlighted', 'dimmed'}
        self._marker_variants = {}   # i -> {'normal', 'selected'}
        self._obj_state       = {}   # i -> 'normal' | 'highlighted' | 'dimmed'

        for i in range(self.n):
            pcd = pcds[i]

            if len(pcd.points) == 0:
                self._obj_state[i] = 'empty'
                continue

            # PCDs
            pcd_normal = copy.deepcopy(pcd)
            pcd_normal.paint_uniform_color(instance_colors[i])

            # Highlighted: blend instance color 60% toward white
            base = np.array(instance_colors[i])
            lighter = base + (1.0 - base) * 0.6
            lighter = np.clip(lighter, 0.0, 1.0)
            pcd_hl = copy.deepcopy(pcd)
            pcd_hl.paint_uniform_color(lighter.tolist())

            self._pcd_variants[i] = {
                'normal':      pcd_normal,
                'highlighted': pcd_hl,
            }

            # Markers
            c = centroids[i]

            m_normal = o3d.geometry.TriangleMesh.create_sphere(radius=self.SPHERE_RADIUS)
            m_normal.translate(c)
            m_normal.paint_uniform_color([1.0, 0.5, 0.0])   # bright orange
            m_normal.compute_vertex_normals()

            m_sel = o3d.geometry.TriangleMesh.create_sphere(radius=self.SPHERE_RADIUS * 1.5)
            m_sel.translate(c)
            m_sel.paint_uniform_color([1.0, 0.85, 0.0])
            m_sel.compute_vertex_normals()

            self._marker_variants[i] = {'normal': m_normal, 'selected': m_sel}

            # Add only the normal variant to the scene initially
            self._scene.scene.add_geometry(f"pcd_{i}",    pcd_normal, self._pcd_mat)
            self._scene.scene.add_geometry(f"marker_{i}", m_normal,   self._mesh_mat)
            self._obj_state[i] = 'normal'

        # ---------- initial camera ----------
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60.0, bounds, bounds.get_center())

        print("\n" + "="*60)
        print("VISUALIZATION READY")
        print("="*60)
        print("Click a orange centroid sphere to inspect an object")
        print("Press Esc to deselect  |  Close window to exit")
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_key(self, event):
        if (event.type == gui.KeyEvent.DOWN and
                event.key == gui.KeyName.ESCAPE):
            self._deselect()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.buttons & int(gui.MouseButton.LEFT):
                self._click_start = (event.x, event.y)
            return gui.Widget.EventCallbackResult.IGNORED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self._click_start is not None:
                start_x, start_y = self._click_start
                dx = event.x - start_x
                dy = event.y - start_y
                self._click_start = None
                if dx * dx + dy * dy < 25:   # <5 px = click, not drag
                    hit = self._pick(start_x, start_y)
                    print(f"[DBG] click at ({start_x},{start_y})  pick={hit}")
                    if hit >= 0:
                        self._select(hit)
                    else:
                        self._deselect()

        return gui.Widget.EventCallbackResult.IGNORED

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _pick(self, wx, wy):
        """Return index of centroid hit by click at window pos (wx, wy), else -1."""
        r = self._scene.frame
        sx     = wx - r.x
        sy_gui = wy - r.y          # GUI convention: y=0 at top
        W, H   = r.width, r.height

        cam = self._scene.scene.camera

        # Camera eye in world space (from view matrix: eye = -R^T @ t)
        V   = np.array(cam.get_view_matrix())
        eye = -(V[:3, :3].T @ V[:3, 3])

        # Convert screen pixel to NDC.
        # GUI: y=0 at top. NDC (OpenGL): y=-1 at bottom, +1 at top.
        ndc_x =  2.0 * sx / W - 1.0
        ndc_y = 1.0 - 2.0 * sy_gui / H

        # Unproject through inverse projection + inverse view matrices.
        # This gives a correct per-pixel world-space ray — unlike cam.unproject(z=0)
        # which returns a near-plane point only ~0.1 m from eye (essentially no
        # per-pixel variation).
        P     = np.array(cam.get_projection_matrix())
        P_inv = np.linalg.inv(P)
        V_inv = np.linalg.inv(V)

        # NDC near-plane → view space (perspective divide)
        p_clip     = np.array([ndc_x, ndc_y, -1.0, 1.0])
        p_view_hom = P_inv @ p_clip
        p_view     = p_view_hom[:3] / p_view_hom[3]

        # View space → world space
        p_world_hom = V_inv @ np.array([p_view[0], p_view[1], p_view[2], 1.0])
        p_world     = p_world_hom[:3] / p_world_hom[3]

        ray_dir = p_world - eye
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-8:
            return -1
        ray_dir /= norm

        # --- debug ---
        print(f"[DBG-ray] eye={np.round(eye,2)}  p_world={np.round(p_world,2)}  dir={np.round(ray_dir,3)}")
        valid_c = [(i, c) for i, c in enumerate(self.centroids)
                   if c is not None and len(self.pcds[i].points) > 0]
        if valid_c:
            rows = []
            for idx, c in valid_c[:5]:
                oc   = c - eye
                t_c  = float(np.dot(oc, ray_dir))
                perp = float(np.linalg.norm(oc - t_c * ray_dir))
                rows.append((perp, f"  centroid[{idx}] {np.round(c,2)}  t={t_c:.1f}  perp={perp:.2f}m"))
            rows.sort(key=lambda x: x[0])
            print("[DBG-ray] 5 closest candidates (perp dist to ray):")
            for _, row in rows[:5]:
                print(row)
        # --- end debug ---

        R2 = self.HIT_RADIUS ** 2
        best_idx, best_t = -1, float('inf')

        for i, c in enumerate(self.centroids):
            if c is None or len(self.pcds[i].points) == 0:
                continue
            oc   = eye - c
            b    = float(np.dot(oc, ray_dir))
            disc = b * b - (float(np.dot(oc, oc)) - R2)
            if disc >= 0:
                t = -b - np.sqrt(max(disc, 0.0))
                if 0 < t < best_t:
                    best_t   = t
                    best_idx = i

        return best_idx

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def _swap_obj(self, i, pcd_state, marker_state):
        """Replace object i's scene geometry with the requested variant."""
        if self._obj_state.get(i) == 'empty':
            return
        self._scene.scene.remove_geometry(f"pcd_{i}")
        self._scene.scene.remove_geometry(f"marker_{i}")
        self._scene.scene.add_geometry(
            f"pcd_{i}",    self._pcd_variants[i][pcd_state],    self._pcd_mat)
        self._scene.scene.add_geometry(
            f"marker_{i}", self._marker_variants[i][marker_state], self._mesh_mat)
        self._obj_state[i] = pcd_state

    def _select(self, idx):
        """Select object idx: highlight it, keep all others at normal color."""
        self.selected = idx
        for i in range(self.n):
            if i == idx:
                self._swap_obj(i, 'highlighted', 'selected')
            else:
                self._swap_obj(i, 'normal', 'normal')
        self._update_info(idx)
        self._scene.force_redraw()

    def _deselect(self):
        """Restore all objects to normal (unselected) state."""
        self.selected = -1
        for i in range(self.n):
            self._swap_obj(i, 'normal', 'normal')
        self.info_label.text = (
            "Click a orange centroid sphere to inspect an object    |    Esc = deselect")
        self._scene.force_redraw()

    def _update_info(self, idx):
        """Update the info label for the selected object."""
        obj_id = f"object_{idx}"
        assignment = self.obj_hierarchy_map.get(obj_id, {})

        obj = self.objects[idx]
        captions = obj.get('captions', [])
        caption = captions[0] if captions else obj.get('caption', f"Object {idx}")

        parts = []
        if assignment.get('zone'):
            parts.append(assignment['zone'])
        if assignment.get('aisle'):
            parts.append(assignment['aisle'])
        if assignment.get('shelf'):
            parts.append(assignment['shelf'])
        if assignment.get('section'):
            parts.append(assignment['section'])
        parts.append(obj_id)
        path = " -> ".join(parts)

        c = self.centroids[idx]
        self.info_label.text = (
            f"[{idx}] {caption[:60]}  |  {path}  |  "
            f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        )

    def run(self):
        gui.Application.instance.run()


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "="*60)
    print("TIERGRAPH VISUALIZATION (ISAAC SIM GLOBAL FRAME)")
    print("="*60 + "\n")

    # Paths
    result_path = Path(cfg.result_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    warehouse_layout_path = str(
        Path(__file__).parent.parent / "data" / "warehouse_layout" / "warehouse_layout.json"
    )
    tiergraph_path = Path(cfg.scenegraph_path)

    # Load data
    print(f"Loading OpenGraph results from {result_path}...")
    objects, _ = load_opengraph_results(result_path)
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

    # 1. Create zone floor polygons and bounding boxes
    zone_boxes = []
    zone_floor_meshes = []
    zone_labels = []

    # Define colors for all zone types
    zone_color_map = {
        'zone_storage': ([0.0, 0.8, 0.0], [0.2, 1.0, 0.2]),      # Dark green / Light green
        'zone_receiving': ([0.8, 0.4, 0.0], [1.0, 0.6, 0.2]),    # Dark orange / Light orange
        'zone_staging': ([0.8, 0.0, 0.8], [1.0, 0.4, 1.0]),        # Dark magenta / Light magenta
        'zone_packing': ([0.8, 0.0, 0.8], [1.0, 0.4, 1.0]),        # Alias for zone_staging (legacy)
        'zone_pallet_truck': ([0.8, 0.8, 0.0], [1.0, 1.0, 0.3]), # Dark yellow / Light yellow
        'zone_hub_robot': ([0.0, 0.6, 0.8], [0.3, 0.8, 1.0]),    # Dark cyan / Light cyan
        'zone_forklift': ([0.7, 0.0, 0.0], [1.0, 0.3, 0.3]),     # Dark red / Light red
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
        if zone_verts and len(zone_verts) >= 3 and zone_id != 'zone_general':
            # Each zone gets a unique Z offset (storage zone gets lowest to be most visible)
            if zone_id == 'zone_storage':
                z_offset = 0.001  # Storage zone closest to actual floor
            else:
                z_offset = 0.01 + (zone_idx * zone_floor_z_offset)

            # Blend dark wireframe color with light fill color (70/30) for a more solid floor
            solid_color = [0.7 * wireframe_color[j] + 0.3 * fill_color[j] for j in range(3)]
            zone_floor = create_floor_mesh(
                zone_verts,
                floor_z=min_z + z_offset,
                color=solid_color
            )
            zone_floor.paint_uniform_color(solid_color)
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
                for _, aisle in aisles.items():
                    # Aisles shown as semi-transparent yellow boxes
                    aisle_lineset = create_bbox_lineset(aisle['bounds'], [0.9, 0.9, 0.0])
                    aisle_boxes.append(aisle_lineset)

            # Create shelf boxes (blue wireframes)
            shelves = zone.get('shelves', {})
            if isinstance(shelves, dict):
                for _, shelf in shelves.items():
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

    # 5. Compute centroids (for click-to-highlight picking)
    centroids = []
    for i, obj in enumerate(objects):
        if len(obj['pcd'].points) > 0:
            points_robot = np.asarray(obj['pcd'].points)
            if has_first_pose:
                points_global = transform_points_to_global(points_robot, T_first)
                points_global[:, 2] += Z_OFFSET
            else:
                points_global = points_robot.copy()
                points_global[:, 2] += Z_OFFSET
            centroids.append(points_global.mean(axis=0))
        else:
            centroids.append(None)

    print(f"  Computed {sum(c is not None for c in centroids)} object centroids")

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

        hierarchy_path = " -> ".join(path_parts) if path_parts else obj_id

        # Print with color indicator
        color = instance_colors[i]
        color_hex = f"RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"

        print(f"{obj_id}:")
        print(f"  Position: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        print(f"  Color: {color_hex}")
        print(f"  Hierarchy: {hierarchy_path}")
        print()

    print("="*60 + "\n")

    # 6. Build infra_geometries list for TierGraphViewer
    infra = []
    # Gray floor mesh intentionally not added (hidden)

    for zone_lineset, zid, _ in zone_boxes:
        infra.append((f"zone_box_{zid}", zone_lineset, "unlitLine"))

    for ai, aisle_box in enumerate(aisle_boxes):
        infra.append((f"aisle_{ai}", aisle_box, "unlitLine"))

    for si, shelf_box in enumerate(shelf_boxes):
        infra.append((f"shelf_{si}", shelf_box, "unlitLine"))

    for bi, obj_box in enumerate(object_boxes):
        infra.append((f"objbox_{bi}", obj_box, "unlitLine"))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    infra.append(("coord_frame", coord_frame, "defaultLit"))

    # 7. Launch interactive viewer
    print("\nInitializing Open3D GUI visualizer with click-to-highlight...")
    viewer = TierGraphViewer(
        pcds=pcds,
        instance_colors=instance_colors,
        centroids=centroids,
        obj_hierarchy_map=obj_hierarchy_map,
        objects=objects,
        infra_geometries=infra,
    )
    viewer.run()


if __name__ == "__main__":
    main()
