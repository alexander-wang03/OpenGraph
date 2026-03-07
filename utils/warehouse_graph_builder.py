"""
Warehouse Graph Builder for TierGraph

Implements hierarchical scene graph construction for warehouse environments.
Assigns objects to the warehouse hierarchy using geometric containment checks.

Unlike OpenGraph's flat MST-based scene graph, TierGraph uses fixed warehouse
geometry to determine each object's precise location.

Hierarchy varies by zone type:

  Non-storage zones (receiving, staging, forklift, etc.):
    Zone → Object   (direct — these are open floor areas with no aisles/shelves)

  Storage zone only (aisles and shelves are sibling children of the zone):
    Zone → Shelf → Section → Object  (object centroid inside a shelf bounding box)
    Zone → Shelf → Object            (in shelf XY area, no matching section)
    Zone → Aisle → Object            (in aisle walkway, not in any shelf)
    Zone → Object                    (fallback — in storage zone but unmatched)

Author: awang (TierGraph thesis)
Date: 2026-02-10
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HierarchyNode:
    """Represents a node in the warehouse hierarchy."""
    id: str
    type: str  # 'zone', 'aisle', 'shelf', 'section', or 'object'
    name: str
    bounds: Dict  # {min: {x, y, z}, max: {x, y, z}, center: {x, y, z}}
    parent_id: Optional[str] = None
    children: List[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class WarehouseGraphBuilder:
    """
    Builds hierarchical scene graph for warehouse environments.

    Aisles and shelves exist ONLY in the storage zone.
    Non-storage zones (receiving, staging, forklift, hub_robot, general)
    are open floor areas — objects there attach directly to the zone node.

    Storage zone hierarchy (aisles and shelves are siblings under the zone):
        zone_storage
         ├─ Aisle   (corridor between shelf rows; objects in the walkway attach here)
         └─ Shelf   (shelving unit; takes priority over aisle in containment check)
             └─ Section (vertical tier on a shelf)
                 └─ Object

    Non-storage zone hierarchy:
        zone_receiving / zone_staging / zone_forklift / ...
         └─ Object  (direct — no aisle or shelf intermediaries)
    """

    def __init__(self, warehouse_layout_path: str):
        """
        Initialize with warehouse layout file.

        Args:
            warehouse_layout_path: Path to warehouse_layout.json
        """
        self.layout_path = Path(warehouse_layout_path)
        self.layout_data = self._load_layout()
        self.nodes = {}  # id -> HierarchyNode
        self.edges = []  # List of (parent_id, child_id, edge_type)
        self._build_infrastructure()

    def _load_layout(self) -> Dict:
        """Load warehouse layout from JSON."""
        with open(self.layout_path, 'r') as f:
            return json.load(f)

    def _build_infrastructure(self):
        """Build the infrastructure hierarchy (zones, aisles, shelves, sections)."""
        # Add functional zones
        for zone_data in self.layout_data['functional_zones']:
            zone_node = HierarchyNode(
                id=zone_data['id'],
                type='zone',
                name=zone_data['name'],
                bounds=zone_data['bounds']
            )
            self.nodes[zone_node.id] = zone_node

            # Add aisles within this zone.
            # Aisles are corridors between shelf rows — they exist ONLY in the
            # storage zone.  Non-storage zones (receiving, staging, forklift,
            # etc.) are open floor areas and never contain aisles.
            if zone_data['id'] == 'zone_storage' and 'aisles' in zone_data and zone_data['aisles']:
                for aisle_id, aisle_data in zone_data['aisles'].items():
                    aisle_node = HierarchyNode(
                        id=aisle_data['id'],
                        type='aisle',
                        name=f"Aisle {aisle_id}",
                        bounds=aisle_data['bounds'],
                        parent_id=zone_node.id
                    )
                    self.nodes[aisle_node.id] = aisle_node
                    zone_node.children.append(aisle_node.id)
                    self.edges.append((zone_node.id, aisle_node.id, 'contains'))

            # Add shelves within this zone.
            # Shelves are always direct children of the zone — they are siblings
            # of aisles, not children.  Objects are assigned to whichever element
            # (shelf or aisle) contains their centroid.
            if 'shelves' in zone_data and zone_data['shelves']:
                for shelf_id, shelf_data in zone_data['shelves'].items():
                    shelf_node = HierarchyNode(
                        id=shelf_data['id'],
                        type='shelf',
                        name=f"Shelf {shelf_id}",
                        bounds=shelf_data['bounds'],
                        parent_id=zone_node.id
                    )
                    self.nodes[shelf_node.id] = shelf_node
                    zone_node.children.append(shelf_node.id)
                    self.edges.append((zone_node.id, shelf_node.id, 'contains'))

                    # Add sections within this shelf
                    if 'sections' in shelf_data and shelf_data['sections']:
                        for section_idx, section_level_data in shelf_data['sections'].items():
                            # Sections have nested structure: {0: {0: {...}, 1: {...}}, 1: {...}}
                            if isinstance(section_level_data, dict):
                                for tier_idx, tier_data in section_level_data.items():
                                    if isinstance(tier_data, dict) and 'id' in tier_data:
                                        section_node = HierarchyNode(
                                            id=tier_data['id'],
                                            type='section',
                                            name=f"{shelf_id} Section {section_idx}-{tier_idx}",
                                            bounds=tier_data['bounds'],
                                            parent_id=shelf_node.id
                                        )
                                        self.nodes[section_node.id] = section_node
                                        shelf_node.children.append(section_node.id)
                                        self.edges.append((shelf_node.id, section_node.id, 'contains'))

        print(f"Infrastructure built: {len(self.nodes)} nodes, {len(self.edges)} edges")
        print(f"  Zones: {sum(1 for n in self.nodes.values() if n.type == 'zone')}")
        print(f"  Aisles: {sum(1 for n in self.nodes.values() if n.type == 'aisle')}")
        print(f"  Shelves: {sum(1 for n in self.nodes.values() if n.type == 'shelf')}")
        print(f"  Sections: {sum(1 for n in self.nodes.values() if n.type == 'section')}")

    def point_in_bounds(self, point: np.ndarray, bounds: Dict) -> bool:
        """
        Check if a 3D point is inside a bounding box.

        Args:
            point: 3D point as [x, y, z]
            bounds: Dict with 'min' and 'max' keys

        Returns:
            True if point is inside bounds

        Note: Some bounds (like zones) only have X/Y, not Z. In that case, only check X/Y.
        """
        min_pt = bounds['min']
        max_pt = bounds['max']

        # Check X and Y (always present)
        if not (min_pt['x'] <= point[0] <= max_pt['x'] and
                min_pt['y'] <= point[1] <= max_pt['y']):
            return False

        # Check Z if present (not all bounds have Z)
        if 'z' in min_pt and 'z' in max_pt:
            if not (min_pt['z'] <= point[2] <= max_pt['z']):
                return False

        return True

    def find_containing_zone(self, point: np.ndarray) -> Optional[str]:
        """
        Find which zone contains the given point.

        Priority order: Check all non-General zones first, then General as fallback.
        This prevents objects from being assigned to General when they're actually
        in a more specific zone (since General's bounds may overlap other zones).
        """
        # First pass: check all non-General zones
        for node in self.nodes.values():
            if node.type == 'zone' and node.id != 'zone_general' and self.point_in_bounds(point, node.bounds):
                return node.id

        # Second pass: check General zone as fallback
        for node in self.nodes.values():
            if node.type == 'zone' and node.id == 'zone_general' and self.point_in_bounds(point, node.bounds):
                return node.id

        return None

    def find_containing_aisle(self, point: np.ndarray, zone_id: str) -> Optional[str]:
        """Find which aisle in the given zone contains the point."""
        zone_node = self.nodes.get(zone_id)
        if not zone_node:
            return None

        for child_id in zone_node.children:
            child = self.nodes.get(child_id)
            if child and child.type == 'aisle' and self.point_in_bounds(point, child.bounds):
                return child_id
        return None

    def find_containing_shelf(self, point: np.ndarray, zone_id: str) -> Optional[str]:
        """Find which shelf (direct zone child) contains the point."""
        zone_node = self.nodes.get(zone_id)
        if not zone_node:
            return None

        for child_id in zone_node.children:
            child = self.nodes.get(child_id)
            if child and child.type == 'shelf' and self.point_in_bounds(point, child.bounds):
                return child_id
        return None

    def find_nearest_structure(self, point: np.ndarray, zone_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Return the (node_id, node_type) of the nearest aisle or shelf in the zone.

        Used as a fallback when a point is inside the storage zone but falls in a
        thin gap not covered by any aisle or shelf AABB.  This typically happens at
        the zone's left/right edges due to AABB inflation after coordinate rotation.
        """
        zone_node = self.nodes.get(zone_id)
        if not zone_node:
            return None, None

        best_id = None
        best_type = None
        best_dist = float('inf')

        for child_id in zone_node.children:
            child = self.nodes.get(child_id)
            if child is None or child.type not in ('aisle', 'shelf'):
                continue
            cx = child.bounds['center']['x']
            cy = child.bounds['center']['y']
            dist = (point[0] - cx) ** 2 + (point[1] - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = child_id
                best_type = child.type

        return best_id, best_type

    def find_containing_section(self, point: np.ndarray, shelf_id: str) -> Optional[str]:
        """Find which section on the shelf contains the point."""
        shelf_node = self.nodes.get(shelf_id)
        if not shelf_node:
            return None

        for child_id in shelf_node.children:
            child = self.nodes.get(child_id)
            if child and child.type == 'section' and self.point_in_bounds(point, child.bounds):
                return child_id
        return None

    def assign_object_to_hierarchy(self, object_id: str, object_position: np.ndarray,
                                    object_caption: str = "", object_data: Dict = None) -> Tuple[bool, str]:
        """
        Assign an OpenGraph object to the warehouse hierarchy.

        Hierarchy logic:
        - Non-storage zones: Zone → Object (direct assignment)
        - Storage zone:
          - If in shelf XY area: Zone → Shelf → Section → Object
          - If in aisle XY area (but not shelf): Zone → Aisle → Object
          - Otherwise: Zone → Object

        Args:
            object_id: Unique ID for the object
            object_position: 3D centroid position [x, y, z]
            object_caption: Object description from OpenGraph
            object_data: Additional object data (pcd, features, etc.)

        Returns:
            (success, hierarchy_path): Success flag and string describing the assignment
        """
        # Step 1: Find containing zone (uses XY coordinates only)
        zone_id = self.find_containing_zone(object_position)
        if not zone_id:
            return False, f"Object {object_id} not in any zone"

        # Step 2: Check if this is the storage zone
        # Only storage zone has shelf/aisle hierarchy
        if zone_id != 'zone_storage':
            # Non-storage zone: assign directly to zone
            object_node = HierarchyNode(
                id=object_id,
                type='object',
                name=object_caption if object_caption else f"Object {object_id}",
                bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                parent_id=zone_id
            )
            self.nodes[object_id] = object_node
            self.nodes[zone_id].children.append(object_id)
            self.edges.append((zone_id, object_id, 'contains'))

            path = f"{zone_id} → {object_id}"
            return True, path

        # Step 3: Storage zone - check if object is in a shelf area (XY containment)
        shelf_id = self.find_containing_shelf(object_position, zone_id)

        if shelf_id:
            # Object is in shelf XY area - try to find specific section (uses XYZ)
            section_id = self.find_containing_section(object_position, shelf_id)

            if section_id:
                # Full hierarchy: Zone → Shelf → Section → Object
                object_node = HierarchyNode(
                    id=object_id,
                    type='object',
                    name=object_caption if object_caption else f"Object {object_id}",
                    bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                    parent_id=section_id
                )
                self.nodes[object_id] = object_node
                self.nodes[section_id].children.append(object_id)
                self.edges.append((section_id, object_id, 'contains'))

                path = f"{zone_id} → {shelf_id} → {section_id} → {object_id}"
                return True, path
            else:
                # In shelf area but no section match: Zone → Shelf → Object
                object_node = HierarchyNode(
                    id=object_id,
                    type='object',
                    name=object_caption if object_caption else f"Object {object_id}",
                    bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                    parent_id=shelf_id
                )
                self.nodes[object_id] = object_node
                self.nodes[shelf_id].children.append(object_id)
                self.edges.append((shelf_id, object_id, 'contains'))

                path = f"{zone_id} → {shelf_id} → {object_id}"
                return True, path

        # Step 4: Not in shelf - check if in aisle (XY containment)
        aisle_id = self.find_containing_aisle(object_position, zone_id)

        if aisle_id:
            # In aisle: Zone → Aisle → Object
            object_node = HierarchyNode(
                id=object_id,
                type='object',
                name=object_caption if object_caption else f"Object {object_id}",
                bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                parent_id=aisle_id
            )
            self.nodes[object_id] = object_node
            self.nodes[aisle_id].children.append(object_id)
            self.edges.append((aisle_id, object_id, 'contains'))

            path = f"{zone_id} → {aisle_id} → {object_id}"
            return True, path

        # Step 5: In storage zone but outside every aisle and shelf AABB.
        # This can occur at zone edges due to AABB inflation after coordinate
        # rotation (the zone's outer boundary extends slightly beyond the
        # outermost shelf/aisle AABBs).  Fall back to the nearest structure.
        nearest_id, nearest_type = self.find_nearest_structure(object_position, zone_id)

        if nearest_id and nearest_type == 'shelf':
            section_id = self.find_containing_section(object_position, nearest_id)
            parent_id = section_id if section_id else nearest_id
            object_node = HierarchyNode(
                id=object_id,
                type='object',
                name=object_caption if object_caption else f"Object {object_id}",
                bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                parent_id=parent_id
            )
            self.nodes[object_id] = object_node
            self.nodes[parent_id].children.append(object_id)
            self.edges.append((parent_id, object_id, 'contains'))
            print(f"  [edge-fallback] {object_id} → nearest shelf {nearest_id}")
            path = f"{zone_id} → {nearest_id} → {section_id} → {object_id}" if section_id else f"{zone_id} → {nearest_id} → {object_id}"
            return True, path

        if nearest_id and nearest_type == 'aisle':
            object_node = HierarchyNode(
                id=object_id,
                type='object',
                name=object_caption if object_caption else f"Object {object_id}",
                bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
                parent_id=nearest_id
            )
            self.nodes[object_id] = object_node
            self.nodes[nearest_id].children.append(object_id)
            self.edges.append((nearest_id, object_id, 'contains'))
            print(f"  [edge-fallback] {object_id} → nearest aisle {nearest_id}")
            path = f"{zone_id} → {nearest_id} → {object_id}"
            return True, path

        # True fallback: layout has no aisles or shelves in this zone (shouldn't happen)
        object_node = HierarchyNode(
            id=object_id,
            type='object',
            name=object_caption if object_caption else f"Object {object_id}",
            bounds={'center': {'x': object_position[0], 'y': object_position[1], 'z': object_position[2]}},
            parent_id=zone_id
        )
        self.nodes[object_id] = object_node
        self.nodes[zone_id].children.append(object_id)
        self.edges.append((zone_id, object_id, 'contains'))
        path = f"{zone_id} → {object_id}"
        return True, path

    def export_graph(self, output_path: str):
        """
        Export the hierarchical scene graph to JSON.

        Args:
            output_path: Path to save JSON file
        """
        # Convert nodes to serializable format
        nodes_data = []
        for node in self.nodes.values():
            node_dict = {
                'id': node.id,
                'type': node.type,
                'name': node.name,
                'bounds': node.bounds,
                'parent_id': node.parent_id,
                'children': node.children
            }
            nodes_data.append(node_dict)

        # Build edges list
        edges_data = [
            {'source': src, 'target': dst, 'type': edge_type}
            for src, dst, edge_type in self.edges
        ]

        # Compute statistics
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'nodes_by_type': {
                'zone': sum(1 for n in self.nodes.values() if n.type == 'zone'),
                'aisle': sum(1 for n in self.nodes.values() if n.type == 'aisle'),
                'shelf': sum(1 for n in self.nodes.values() if n.type == 'shelf'),
                'section': sum(1 for n in self.nodes.values() if n.type == 'section'),
                'object': sum(1 for n in self.nodes.values() if n.type == 'object'),
            },
            'max_depth': self._compute_max_depth()
        }

        graph_data = {
            'metadata': {
                'type': 'TierGraph',
                'description': 'Hierarchical warehouse scene graph',
                'hierarchy_levels': 5,
                'warehouse_layout': str(self.layout_path)
            },
            'statistics': stats,
            'nodes': nodes_data,
            'edges': edges_data
        }

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)

        print(f"\nTierGraph exported to {output_path}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total edges: {stats['total_edges']}")
        print(f"  Objects assigned: {stats['nodes_by_type']['object']}")
        print(f"  Max hierarchy depth: {stats['max_depth']}")

    def _compute_max_depth(self) -> int:
        """Compute the maximum depth of the hierarchy tree."""
        def get_depth(node_id, visited=None):
            if visited is None:
                visited = set()
            if node_id in visited:
                return 0
            visited.add(node_id)

            node = self.nodes.get(node_id)
            if not node or not node.children:
                return 1
            return 1 + max(get_depth(child_id, visited.copy()) for child_id in node.children)

        # Find root nodes (zones)
        roots = [n.id for n in self.nodes.values() if n.type == 'zone']
        if not roots:
            return 0
        return max(get_depth(root_id) for root_id in roots)

    def print_summary(self):
        """Print a summary of the hierarchical scene graph."""
        print("\n" + "="*60)
        print("TIERGRAPH SUMMARY")
        print("="*60)

        stats = {
            'zone': sum(1 for n in self.nodes.values() if n.type == 'zone'),
            'aisle': sum(1 for n in self.nodes.values() if n.type == 'aisle'),
            'shelf': sum(1 for n in self.nodes.values() if n.type == 'shelf'),
            'section': sum(1 for n in self.nodes.values() if n.type == 'section'),
            'object': sum(1 for n in self.nodes.values() if n.type == 'object'),
        }

        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total edges: {len(self.edges)}")
        print(f"\nNodes by level:")
        print(f"  Level 1 (Zones):    {stats['zone']}")
        print(f"  Level 2 (Aisles):   {stats['aisle']}")
        print(f"  Level 3 (Shelves):  {stats['shelf']}")
        print(f"  Level 4 (Sections): {stats['section']}")
        print(f"  Level 5 (Objects):  {stats['object']}")
        print(f"\nMax depth: {self._compute_max_depth()}")
        print("="*60 + "\n")
