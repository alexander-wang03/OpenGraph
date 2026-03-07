#!/usr/bin/env python3
"""
Create a hierarchical JSON warehouse layout from object_lists.csv.
Includes floor map and intelligent shelf detection based on RackShield pairs.
"""

import csv
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def parse_floor_tiles(csv_path: Path) -> List[Dict]:
    """Extract floor tiles matching /World/full_warehouse/SM_floor## pattern."""
    floor_tiles = []
    floor_pattern = re.compile(r'^/World/full_warehouse/SM_floor\d+(_\d+)?$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth == 3 and floor_pattern.match(prim_path):
                tile = {
                    'bbox_min': (float(row['bbox_min_x']), float(row['bbox_min_y'])),
                    'bbox_max': (float(row['bbox_max_x']), float(row['bbox_max_y'])),
                    'z': float(row['trans_z']),
                    'bbox_max_z': float(row['bbox_max_z'])
                }
                floor_tiles.append(tile)
    return floor_tiles


def parse_rackshields(csv_path: Path) -> List[Dict]:
    """Extract RackShield positions and orientations."""
    rackshields = []
    rackshield_pattern = re.compile(r'^/World/full_warehouse/SM_Rackshield_\d+$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth == 3 and rackshield_pattern.match(prim_path):
                rackshield = {
                    'prim_path': prim_path,
                    'name': row['name'],
                    'position': {
                        'x': float(row['trans_x']),
                        'y': float(row['trans_y']),
                        'z': float(row['trans_z'])
                    },
                    'rotation': {
                        'roll': float(row['rot_x']),
                        'pitch': float(row['rot_y']),
                        'yaw': float(row['rot_z'])  # rotation about Z axis
                    },
                    'bounding_box': {
                        'min': {
                            'x': float(row['bbox_min_x']),
                            'y': float(row['bbox_min_y']),
                            'z': float(row['bbox_min_z'])
                        },
                        'max': {
                            'x': float(row['bbox_max_x']),
                            'y': float(row['bbox_max_y']),
                            'z': float(row['bbox_max_z'])
                        },
                        'center': {
                            'x': float(row['bbox_center_x']),
                            'y': float(row['bbox_center_y']),
                            'z': float(row['bbox_center_z'])
                        }
                    }
                }
                rackshields.append(rackshield)
    return rackshields


def parse_aisle_signs(csv_path: Path) -> List[Dict]:
    """Extract AisleSign positions."""
    aisle_signs = []
    aislesign_pattern = re.compile(r'^/World/full_warehouse/S_AisleSign')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            # Only match depth 3 (direct children of warehouse)
            if depth == 3 and aislesign_pattern.match(prim_path):
                aisle_sign = {
                    'prim_path': prim_path,
                    'name': row['name'],
                    'position': {
                        'x': float(row['trans_x']),
                        'y': float(row['trans_y']),
                        'z': float(row['trans_z'])
                    },
                    'bounding_box': {
                        'center': {
                            'x': float(row['bbox_center_x']),
                            'y': float(row['bbox_center_y']),
                            'z': float(row['bbox_center_z'])
                        }
                    }
                }
                aisle_signs.append(aisle_sign)
    return aisle_signs


def parse_rack_shelves(csv_path: Path) -> List[Dict]:
    """Extract RackShelf positions and bounding boxes."""
    rack_shelves = []
    rackshelf_pattern = re.compile(r'^/World/full_warehouse/SM_RackShelf_\d+$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            # Only match depth 3 (direct children of warehouse, not sub-objects)
            if depth == 3 and rackshelf_pattern.match(prim_path):
                rack_shelf = {
                    'prim_path': prim_path,
                    'name': row['name'],
                    'position': {
                        'x': float(row['trans_x']),
                        'y': float(row['trans_y']),
                        'z': float(row['trans_z'])
                    },
                    'bounding_box': {
                        'min': {
                            'x': float(row['bbox_min_x']),
                            'y': float(row['bbox_min_y']),
                            'z': float(row['bbox_min_z'])
                        },
                        'max': {
                            'x': float(row['bbox_max_x']),
                            'y': float(row['bbox_max_y']),
                            'z': float(row['bbox_max_z'])
                        },
                        'center': {
                            'x': float(row['bbox_center_x']),
                            'y': float(row['bbox_center_y']),
                            'z': float(row['bbox_center_z'])
                        }
                    }
                }
                rack_shelves.append(rack_shelf)
    return rack_shelves


def parse_rack_frames(csv_path: Path) -> List[Dict]:
    """Extract RackFrame positions and bounding boxes."""
    rack_frames = []
    rackframe_pattern = re.compile(r'^/World/full_warehouse/SM_RackFrame_\d+$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            # Only match depth 3 (direct children of warehouse, not sub-objects)
            if depth == 3 and rackframe_pattern.match(prim_path):
                rack_frame = {
                    'prim_path': prim_path,
                    'name': row['name'],
                    'position': {
                        'x': float(row['trans_x']),
                        'y': float(row['trans_y']),
                        'z': float(row['trans_z'])
                    },
                    'bounding_box': {
                        'min': {
                            'x': float(row['bbox_min_x']),
                            'y': float(row['bbox_min_y']),
                            'z': float(row['bbox_min_z'])
                        },
                        'max': {
                            'x': float(row['bbox_max_x']),
                            'y': float(row['bbox_max_y']),
                            'z': float(row['bbox_max_z'])
                        },
                        'center': {
                            'x': float(row['bbox_center_x']),
                            'y': float(row['bbox_center_y']),
                            'z': float(row['bbox_center_z'])
                        }
                    }
                }
                rack_frames.append(rack_frame)
    return rack_frames


def parse_asset_by_name(csv_path: Path, name: str) -> Optional[Dict]:
    """Extract the first asset matching the given name (any depth)."""
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['name'] == name:
                return {
                    'prim_path': row['prim_path'],
                    'name': row['name'],
                    'bounding_box': {
                        'min': {
                            'x': float(row['bbox_min_x']),
                            'y': float(row['bbox_min_y']),
                            'z': float(row['bbox_min_z'])
                        },
                        'max': {
                            'x': float(row['bbox_max_x']),
                            'y': float(row['bbox_max_y']),
                            'z': float(row['bbox_max_z'])
                        }
                    }
                }
    return None


def parse_staging_tapes(csv_path: Path) -> List[Dict]:
    """Extract staging zone boundary tapes (C04 long side + C02 short side)."""
    tapes = []
    long_pattern = re.compile(
        r'^/World/full_warehouse/VinylMessageTape8M75MM_C04_PR_NVD_\d+$')
    short_pattern = re.compile(
        r'^/World/full_warehouse/VinylMessageTape4M75MM_C02_PR_NVD_\d+$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth == 3 and (long_pattern.match(prim_path) or
                               short_pattern.match(prim_path)):
                tapes.append({
                    'prim_path': prim_path,
                    'name': row['name'],
                    'bounding_box': {
                        'min': {
                            'x': float(row['bbox_min_x']),
                            'y': float(row['bbox_min_y']),
                            'z': float(row['bbox_min_z'])
                        },
                        'max': {
                            'x': float(row['bbox_max_x']),
                            'y': float(row['bbox_max_y']),
                            'z': float(row['bbox_max_z'])
                        }
                    }
                })
    return tapes


def compute_staging_zone_bounds(
    tapes: List[Dict]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute rectangular (px_min, py_min, px_max, py_max) from the outer
    envelope of all staging boundary tapes.  Gaps in the tape boundary
    (entry/exit points) do not reduce the rectangle.

    Returns None if no tapes are found.
    """
    if not tapes:
        return None

    px_min = min(t['bounding_box']['min']['x'] for t in tapes)
    py_min = min(t['bounding_box']['min']['y'] for t in tapes)
    px_max = max(t['bounding_box']['max']['x'] for t in tapes)
    py_max = max(t['bounding_box']['max']['y'] for t in tapes)
    return (px_min, py_min, px_max, py_max)


def parse_boundary_walls(csv_path: Path) -> List[Dict]:
    """
    Extract depth-3 SM_WallA boundary wall assets.

    Skips entries with sentinel bbox values (3.4e+38).
    Returns list of dicts with prim_path, name, and 2D bounding_box.
    """
    walls = []
    wall_pattern = re.compile(r'^/World/full_warehouse/SM_WallA_')
    SENTINEL_THRESHOLD = 1e+30

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth == 3 and wall_pattern.match(prim_path):
                bbox_vals = [
                    float(row['bbox_min_x']), float(row['bbox_min_y']),
                    float(row['bbox_max_x']), float(row['bbox_max_y'])
                ]
                if any(abs(v) > SENTINEL_THRESHOLD for v in bbox_vals):
                    continue
                walls.append({
                    'prim_path': prim_path,
                    'name': row['name'],
                    'bounding_box': {
                        'min': {'x': bbox_vals[0], 'y': bbox_vals[1]},
                        'max': {'x': bbox_vals[2], 'y': bbox_vals[3]}
                    }
                })
    return walls


def compute_wall_boundary(
    walls: List[Dict]
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute outer bounding rectangle of all wall bboxes.

    Returns (wx_min, wy_min, wx_max, wy_max) or None if no walls.
    """
    if not walls:
        return None

    wx_min = min(w['bounding_box']['min']['x'] for w in walls)
    wy_min = min(w['bounding_box']['min']['y'] for w in walls)
    wx_max = max(w['bounding_box']['max']['x'] for w in walls)
    wy_max = max(w['bounding_box']['max']['y'] for w in walls)
    return (wx_min, wy_min, wx_max, wy_max)


def clip_floor_to_boundary(
    floor_polygon: Dict,
    boundary: Tuple[float, float, float, float]
) -> Dict:
    """
    Clip a floor polygon dict to a bounding rectangle.

    Uses Shapely intersection. Falls back to simple bound clamping
    if Shapely is unavailable.

    Args:
        floor_polygon: dict with 'vertices', 'bounds', 'area', etc.
        boundary: (wx_min, wy_min, wx_max, wy_max) clip rectangle

    Returns:
        New floor polygon dict with clipped vertices and updated metadata.
    """
    wx_min, wy_min, wx_max, wy_max = boundary

    try:
        from shapely.geometry import box as shapely_box, Polygon

        floor_verts = floor_polygon['vertices']
        floor_shape = Polygon(floor_verts)
        clip_box = shapely_box(wx_min, wy_min, wx_max, wy_max)
        clipped = floor_shape.intersection(clip_box)

        if clipped.is_empty:
            return floor_polygon  # no change if intersection is empty

        if clipped.geom_type == 'Polygon':
            coords = list(clipped.exterior.coords)
            new_verts = [[float(x), float(y)] for x, y in coords[:-1]]
            area = float(clipped.area)
        elif clipped.geom_type == 'MultiPolygon':
            largest = max(clipped.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
            new_verts = [[float(x), float(y)] for x, y in coords[:-1]]
            area = sum(float(p.area) for p in clipped.geoms)
        else:
            return floor_polygon

    except ImportError:
        # Fallback: clamp rectangular bounds
        old_b = floor_polygon['bounds']
        new_min_x = max(old_b['min']['x'], wx_min)
        new_min_y = max(old_b['min']['y'], wy_min)
        new_max_x = min(old_b['max']['x'], wx_max)
        new_max_y = min(old_b['max']['y'], wy_max)
        new_verts = [
            [new_min_x, new_min_y], [new_max_x, new_min_y],
            [new_max_x, new_max_y], [new_min_x, new_max_y]
        ]
        area = (new_max_x - new_min_x) * (new_max_y - new_min_y)

    all_x = [v[0] for v in new_verts]
    all_y = [v[1] for v in new_verts]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    avg_z = floor_polygon['bounds']['min']['z']

    return {
        'type': 'polygon',
        'vertices': new_verts,
        'vertex_count': len(new_verts),
        'bounds': {
            'min': {'x': min_x, 'y': min_y, 'z': avg_z},
            'max': {'x': max_x, 'y': max_y, 'z': avg_z},
            'center': {
                'x': (min_x + max_x) / 2,
                'y': (min_y + max_y) / 2,
                'z': avg_z
            },
            'size': {
                'x': max_x - min_x,
                'y': max_y - min_y,
                'z': 0.0
            }
        },
        'area': area
    }


def parse_reflective_tapes(csv_path: Path) -> List[Dict]:
    """
    Extract ReflectiveTape objects at depth 3.

    Skips entries with invalid/huge bbox values (sentinel ~3.4e+38).
    Returns list of dicts with prim_path, name, bounding_box.
    """
    tapes = []
    tape_pattern = re.compile(
        r'^/World/full_warehouse/ReflectiveTape\w+_J\d+_PR_NVD_\d+$')
    SENTINEL_THRESHOLD = 1e+30

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth == 3 and tape_pattern.match(prim_path):
                bbox_vals = [
                    float(row['bbox_min_x']), float(row['bbox_min_y']),
                    float(row['bbox_min_z']), float(row['bbox_max_x']),
                    float(row['bbox_max_y']), float(row['bbox_max_z'])
                ]
                # Skip tapes with sentinel/invalid bbox values
                if any(abs(v) > SENTINEL_THRESHOLD for v in bbox_vals):
                    continue
                tapes.append({
                    'prim_path': prim_path,
                    'name': row['name'],
                    'bounding_box': {
                        'min': {
                            'x': bbox_vals[0], 'y': bbox_vals[1],
                            'z': bbox_vals[2]
                        },
                        'max': {
                            'x': bbox_vals[3], 'y': bbox_vals[4],
                            'z': bbox_vals[5]
                        }
                    }
                })
    return tapes


def detect_tape_zones(
    tapes: List[Dict]
) -> List[Tuple[float, float, float, float]]:
    """
    Detect 3-sided rectangular zones from connected reflective tapes.

    Algorithm:
    1. Classify each tape as horizontal/vertical by bbox aspect ratio
    2. Compute 2 endpoints per tape (center of short dimension at each end)
    3. Build adjacency graph (tapes connected if endpoints within tolerance)
    4. BFS to find connected components
    5. Keep only components of exactly 3 tapes
    6. Return bounding rectangle (x_min, y_min, x_max, y_max) for each

    Returns list of (x_min, y_min, x_max, y_max) tuples.
    """
    if not tapes:
        return []

    ENDPOINT_TOLERANCE = 0.5  # meters

    # Classify tapes and compute endpoints
    tape_endpoints = []  # list of (endpoint1, endpoint2) per tape
    for t in tapes:
        bb = t['bounding_box']
        x_min, y_min = bb['min']['x'], bb['min']['y']
        x_max, y_max = bb['max']['x'], bb['max']['y']
        dx = x_max - x_min
        dy = y_max - y_min

        if dx >= dy:
            # Horizontal tape: endpoints at left/right, centered in Y
            cy = (y_min + y_max) / 2
            tape_endpoints.append(((x_min, cy), (x_max, cy)))
        else:
            # Vertical tape: endpoints at top/bottom, centered in X
            cx = (x_min + x_max) / 2
            tape_endpoints.append(((cx, y_min), (cx, y_max)))

    n = len(tapes)

    # Build adjacency graph
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            connected = False
            for ep_i in tape_endpoints[i]:
                for ep_j in tape_endpoints[j]:
                    dist = math.sqrt(
                        (ep_i[0] - ep_j[0]) ** 2 +
                        (ep_i[1] - ep_j[1]) ** 2
                    )
                    if dist < ENDPOINT_TOLERANCE:
                        connected = True
                        break
                if connected:
                    break
            if connected:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    # Keep only components of exactly 3 tapes
    zones = []
    for comp in components:
        if len(comp) != 3:
            continue
        # Compute bounding rectangle from the 3 tapes
        comp_tapes = [tapes[i] for i in comp]
        zx_min = min(t['bounding_box']['min']['x'] for t in comp_tapes)
        zy_min = min(t['bounding_box']['min']['y'] for t in comp_tapes)
        zx_max = max(t['bounding_box']['max']['x'] for t in comp_tapes)
        zy_max = max(t['bounding_box']['max']['y'] for t in comp_tapes)
        zones.append((zx_min, zy_min, zx_max, zy_max))

    return zones


def parse_zone_indicator_assets(
    csv_path: Path
) -> Dict[str, List[Dict]]:
    """
    Parse depth-3 assets that indicate what type each tape zone is.

    Matches:
    - PalletTruckScale|HeavyDutyPalletTruck|LowProfilePalletTruck -> 'pallet_truck'
    - iw_hub_robots -> 'hub_robot'
    - forklift_b_sensor(_\\d+)? -> 'forklift'

    Returns dict mapping type -> list of {name, center_x, center_y}.
    """
    indicators: Dict[str, List[Dict]] = {
        'pallet_truck': [],
        'hub_robot': [],
        'forklift': []
    }

    pallet_pattern = re.compile(
        r'^/World/full_warehouse/'
        r'(PalletTruckScale|HeavyDutyPalletTruck|LowProfilePalletTruck)')
    hub_pattern = re.compile(
        r'^/World/full_warehouse/iw_hub_robots')
    forklift_pattern = re.compile(
        r'^/World/full_warehouse/forklift_b_sensor(_\d+)?$')

    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            prim_path = row['prim_path']
            depth = int(row['depth'])
            if depth != 3:
                continue

            center_x = float(row['bbox_center_x'])
            center_y = float(row['bbox_center_y'])
            entry = {'name': row['name'],
                     'center_x': center_x, 'center_y': center_y}

            if pallet_pattern.match(prim_path):
                indicators['pallet_truck'].append(entry)
            elif hub_pattern.match(prim_path):
                indicators['hub_robot'].append(entry)
            elif forklift_pattern.match(prim_path):
                indicators['forklift'].append(entry)

    return indicators


def identify_tape_zones(
    tape_zone_rects: List[Tuple[float, float, float, float]],
    indicators: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Identify each tape zone by counting indicator assets inside its bounds.

    The asset type with the highest count wins. Returns a list of dicts with
    zone_id, name, type, and bounds for each identified zone.
    """
    ZONE_INFO = {
        'pallet_truck': ('zone_pallet_truck', 'Pallet Truck Area'),
        'hub_robot': ('zone_hub_robot', 'Hub Robot Area'),
        'forklift': ('zone_forklift', 'Forklift Area'),
    }

    identified = []
    used_types: set = set()

    for rect in tape_zone_rects:
        zx_min, zy_min, zx_max, zy_max = rect

        # Count indicator assets inside this zone
        counts: Dict[str, int] = {}
        for asset_type, assets in indicators.items():
            cnt = sum(
                1 for a in assets
                if zx_min <= a['center_x'] <= zx_max
                and zy_min <= a['center_y'] <= zy_max
            )
            if cnt > 0:
                counts[asset_type] = cnt

        if not counts:
            continue

        # Pick type with highest count (break ties alphabetically)
        best_type = max(counts, key=lambda t: (counts[t], t))
        if best_type in used_types:
            continue
        used_types.add(best_type)

        zone_id, zone_name = ZONE_INFO[best_type]
        identified.append({
            'zone_id': zone_id,
            'name': zone_name,
            'type': best_type,
            'bounds': rect,
        })

    return identified


def _point_in_rect(px: float, py: float,
                   rx_min: float, ry_min: float,
                   rx_max: float, ry_max: float) -> bool:
    """Check if point (px, py) is inside axis-aligned rectangle."""
    return rx_min <= px <= rx_max and ry_min <= py <= ry_max


def _make_zone_dict(zone_id: str, name: str, zone_type: str,
                    vertices: List[List[float]], area: float,
                    defining_asset: Optional[str],
                    aisle_ids: List[str],
                    shelf_ids: List[str]) -> Dict:
    """Build a zone dictionary from vertices and metadata."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return {
        'id': zone_id,
        'name': name,
        'type': zone_type,
        'vertices': vertices,
        'bounds': {
            'min': {'x': min(xs), 'y': min(ys)},
            'max': {'x': max(xs), 'y': max(ys)},
            'center': {
                'x': (min(xs) + max(xs)) / 2,
                'y': (min(ys) + max(ys)) / 2
            },
            'size': {
                'x': max(xs) - min(xs),
                'y': max(ys) - min(ys)
            }
        },
        'area': area,
        'defining_asset': defining_asset,
        'aisles': aisle_ids,
        'shelves': shelf_ids
    }


def detect_functional_zones(floor_polygon: Dict,
                            receiving_wall: Optional[Dict],
                            staging_bounds: Optional[Tuple[float, float, float, float]],
                            identified_tape_zones: List[Dict],
                            aisles: List[Dict],
                            shelves: List[Dict]) -> List[Dict]:
    """
    Create functional zones within the warehouse floor area.

    Zones (in priority order for claiming aisles/shelves):
    1. Receiving Area:  XY area from the sm_wall_a01_01 bounding box, clipped to floor
    2. Staging Area:    XY rectangle from boundary tape outer bounding box
    3. Tape Zones:      ReflectiveTape-bounded zones (Pallet Truck, Hub Robot, Forklift)
    4. Storage Area:    merged XY bounding box of all shelves and aisles
    5. General:         floor area minus all other zones

    Each zone lists which aisles and shelves fall within it (by center-point
    containment, claimed by the first matching zone in priority order).
    """
    EDGE_SNAP = 0.05  # snap receiving edges within 5cm of floor edges

    zones = []
    floor_bounds = floor_polygon.get('bounds', {})
    fx_min = floor_bounds['min']['x']
    fx_max = floor_bounds['max']['x']
    fy_min = floor_bounds['min']['y']
    fy_max = floor_bounds['max']['y']

    # Aisles are corridors between shelf rows and belong exclusively to the
    # storage zone.  Non-storage zones (receiving, staging, forklift, etc.)
    # are open floor areas — they never contain aisles.
    # Only shelves (and their sections) can appear in non-storage zones.
    claimed_shelf_ids: set = set()

    # --- 1. Receiving Area ---
    rx_min = rx_max = ry_min = ry_max = 0.0
    has_receiving = False

    if receiving_wall:
        wb = receiving_wall['bounding_box']
        rx_min = max(wb['min']['x'], fx_min)
        rx_max = min(wb['max']['x'], fx_max)
        ry_min = max(wb['min']['y'], fy_min)
        ry_max = min(wb['max']['y'], fy_max)

        # Snap edges that are close to floor edges
        if abs(rx_min - fx_min) < EDGE_SNAP:
            rx_min = fx_min
        if abs(rx_max - fx_max) < EDGE_SNAP:
            rx_max = fx_max
        if abs(ry_min - fy_min) < EDGE_SNAP:
            ry_min = fy_min
        if abs(ry_max - fy_max) < EDGE_SNAP:
            ry_max = fy_max

        if rx_max > rx_min and ry_max > ry_min:
            has_receiving = True

            # No aisles in receiving zone — it is open floor only.
            recv_shelves = []
            for shelf in shelves:
                sc = shelf['bounds']['center']
                if _point_in_rect(sc['x'], sc['y'], rx_min, ry_min, rx_max, ry_max):
                    recv_shelves.append(shelf['id'])
                    claimed_shelf_ids.add(shelf['id'])

            recv_vertices = [
                [rx_min, ry_min], [rx_max, ry_min],
                [rx_max, ry_max], [rx_min, ry_max]
            ]
            recv_area = (rx_max - rx_min) * (ry_max - ry_min)

            zones.append(_make_zone_dict(
                'zone_receiving', 'Receiving Area', 'receiving',
                recv_vertices, recv_area,
                receiving_wall['name'], [], recv_shelves
            ))

    # --- 2. Staging Area (bounded by floor tape) ---
    has_staging = False
    px_min = px_max = py_min = py_max = 0.0

    if staging_bounds is not None:
        px_min, py_min, px_max, py_max = staging_bounds

        # Clip to floor bounds
        px_min = max(px_min, fx_min)
        px_max = min(px_max, fx_max)
        py_min = max(py_min, fy_min)
        py_max = min(py_max, fy_max)

        if px_max > px_min and py_max > py_min:
            has_staging = True

            # No aisles in staging zone — it is open floor only.
            pack_shelves = []
            for shelf in shelves:
                if shelf['id'] in claimed_shelf_ids:
                    continue
                sc = shelf['bounds']['center']
                if _point_in_rect(sc['x'], sc['y'], px_min, py_min, px_max, py_max):
                    pack_shelves.append(shelf['id'])
                    claimed_shelf_ids.add(shelf['id'])

            pack_vertices = [
                [px_min, py_min], [px_max, py_min],
                [px_max, py_max], [px_min, py_max]
            ]
            pack_area = (px_max - px_min) * (py_max - py_min)

            zones.append(_make_zone_dict(
                'zone_staging', 'Staging Area', 'staging',
                pack_vertices, pack_area,
                None, [], pack_shelves
            ))

    # --- 3. Tape Zones (ReflectiveTape-bounded areas) ---
    tape_zone_rects = []  # collect for General subtraction
    for tz in identified_tape_zones:
        tz_x_min, tz_y_min, tz_x_max, tz_y_max = tz['bounds']

        # Clip to floor bounds
        tz_x_min = max(tz_x_min, fx_min)
        tz_x_max = min(tz_x_max, fx_max)
        tz_y_min = max(tz_y_min, fy_min)
        tz_y_max = min(tz_y_max, fy_max)

        if tz_x_max <= tz_x_min or tz_y_max <= tz_y_min:
            continue

        tape_zone_rects.append((tz_x_min, tz_y_min, tz_x_max, tz_y_max))

        # No aisles in tape zones (pallet truck, hub robot, forklift areas).
        tz_shelves = []
        for shelf in shelves:
            if shelf['id'] in claimed_shelf_ids:
                continue
            sc = shelf['bounds']['center']
            if _point_in_rect(sc['x'], sc['y'],
                              tz_x_min, tz_y_min, tz_x_max, tz_y_max):
                tz_shelves.append(shelf['id'])
                claimed_shelf_ids.add(shelf['id'])

        tz_vertices = [
            [tz_x_min, tz_y_min], [tz_x_max, tz_y_min],
            [tz_x_max, tz_y_max], [tz_x_min, tz_y_max]
        ]
        tz_area = (tz_x_max - tz_x_min) * (tz_y_max - tz_y_min)

        zones.append(_make_zone_dict(
            tz['zone_id'], tz['name'], tz['type'],
            tz_vertices, tz_area,
            None, [], tz_shelves
        ))

    # --- 4. Storage Area (bounding box of all shelves + aisles) ---
    # Storage zone is defined as the EXACT axis-aligned bounding box of all
    # shelves and aisles, with no expansion to include corridors.
    # ALL aisles belong to the storage zone — aisles are corridors between
    # shelf rows and are only meaningful in the storage zone context.
    has_storage = False
    sx_min = sx_max = sy_min = sy_max = 0.0
    all_bounds = []
    for s in shelves:
        all_bounds.append(s['bounds'])
    for a in aisles:
        all_bounds.append(a['bounds'])

    if all_bounds:
        # Storage zone: EXACT bounding box of all shelves + aisles (no expansion)
        sx_min = min(b['min']['x'] for b in all_bounds)
        sx_max = max(b['max']['x'] for b in all_bounds)
        sy_min = min(b['min']['y'] for b in all_bounds)
        sy_max = max(b['max']['y'] for b in all_bounds)

        has_storage = True

        # All aisles go to storage zone — no other zone claims aisles.
        stor_aisles = [a['id'] for a in aisles]

        stor_shelves = []
        for shelf in shelves:
            if shelf['id'] in claimed_shelf_ids:
                continue
            sc = shelf['bounds']['center']
            if _point_in_rect(sc['x'], sc['y'], sx_min, sy_min, sx_max, sy_max):
                stor_shelves.append(shelf['id'])
                claimed_shelf_ids.add(shelf['id'])

        stor_vertices = [
            [sx_min, sy_min], [sx_max, sy_min],
            [sx_max, sy_max], [sx_min, sy_max]
        ]
        stor_area = (sx_max - sx_min) * (sy_max - sy_min)

        zones.append(_make_zone_dict(
            'zone_storage', 'Storage Area', 'storage',
            stor_vertices, stor_area,
            None, stor_aisles, stor_shelves
        ))

    # --- 5. General Area (floor minus all other zones) ---
    # General zone is residual open floor — no aisles, only unclaimed shelves.
    gen_shelves = [s['id'] for s in shelves if s['id'] not in claimed_shelf_ids]

    try:
        from shapely.geometry import box as shapely_box

        floor_shape = shapely_box(fx_min, fy_min, fx_max, fy_max)
        if has_receiving:
            floor_shape = floor_shape.difference(
                shapely_box(rx_min, ry_min, rx_max, ry_max))
        if has_staging:
            floor_shape = floor_shape.difference(
                shapely_box(px_min, py_min, px_max, py_max))
        for tz_rect in tape_zone_rects:
            floor_shape = floor_shape.difference(
                shapely_box(tz_rect[0], tz_rect[1], tz_rect[2], tz_rect[3]))
        if has_storage:
            floor_shape = floor_shape.difference(
                shapely_box(sx_min, sy_min, sx_max, sy_max))

        gen_area = float(floor_shape.area)
        if floor_shape.is_empty:
            gen_vertices = [
                [fx_min, fy_min], [fx_max, fy_min],
                [fx_max, fy_max], [fx_min, fy_max]
            ]
            gen_area = 0.0
        elif floor_shape.geom_type == 'Polygon':
            coords = list(floor_shape.exterior.coords)
            gen_vertices = [[float(x), float(y)] for x, y in coords[:-1]]
        elif floor_shape.geom_type == 'MultiPolygon':
            gen_area = sum(float(p.area) for p in floor_shape.geoms)
            largest = max(floor_shape.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
            gen_vertices = [[float(x), float(y)] for x, y in coords[:-1]]
        else:
            gen_vertices = [
                [fx_min, fy_min], [fx_max, fy_min],
                [fx_max, fy_max], [fx_min, fy_max]
            ]
    except ImportError:
        # Fallback without Shapely: use floor rectangle, approximate area
        gen_vertices = [
            [fx_min, fy_min], [fx_max, fy_min],
            [fx_max, fy_max], [fx_min, fy_max]
        ]
        floor_area = (fx_max - fx_min) * (fy_max - fy_min)
        recv_area_val = (rx_max - rx_min) * (ry_max - ry_min) if has_receiving else 0.0
        pack_area_val = (px_max - px_min) * (py_max - py_min) if has_staging else 0.0
        tape_area_val = sum((r[2] - r[0]) * (r[3] - r[1]) for r in tape_zone_rects)
        stor_area_val = (sx_max - sx_min) * (sy_max - sy_min) if has_storage else 0.0
        gen_area = floor_area - recv_area_val - pack_area_val - tape_area_val - stor_area_val

    zones.append(_make_zone_dict(
        'zone_general', 'General', 'general',
        gen_vertices, gen_area,
        None, [], gen_shelves
    ))

    return zones


def get_facing_direction(yaw_rad: float) -> str:
    """
    Determine cardinal direction the rackshield is facing.
    yaw ≈ 0: +X (forward)
    yaw ≈ π/2 (90°): +Y (left)
    yaw ≈ π (180°): -X (backward)
    yaw ≈ -π/2 (-90°): -Y (right)
    """
    # Normalize to [-π, π]
    yaw = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))

    # Convert to degrees for easier understanding
    yaw_deg = math.degrees(yaw)

    # Determine direction with tolerance
    if -45 <= yaw_deg < 45:
        return '+X'
    elif 45 <= yaw_deg < 135:
        return '+Y'
    elif yaw_deg >= 135 or yaw_deg < -135:
        return '-X'
    else:  # -135 <= yaw_deg < -45
        return '-Y'


def are_shields_facing(shield1: Dict, shield2: Dict, tolerance: float = 0.5) -> bool:
    """
    Check if two rackshields are facing each other.
    They should face opposite directions along their separation axis.
    """
    dir1 = get_facing_direction(shield1['rotation']['yaw'])
    dir2 = get_facing_direction(shield2['rotation']['yaw'])

    # Get positions
    x1, y1 = shield1['position']['x'], shield1['position']['y']
    x2, y2 = shield2['position']['x'], shield2['position']['y']

    dx = x2 - x1
    dy = y2 - y1

    # Determine primary separation axis
    if abs(dx) > abs(dy):
        # Separated along X axis
        if dx > 0:
            # shield2 is to the right (+X) of shield1
            # shield1 should face +X, shield2 should face -X
            return (dir1 == '+X' and dir2 == '-X')
        else:
            # shield2 is to the left (-X) of shield1
            return (dir1 == '-X' and dir2 == '+X')
    else:
        # Separated along Y axis
        if dy > 0:
            # shield2 is above (+Y) shield1
            # shield1 should face +Y, shield2 should face -Y
            return (dir1 == '+Y' and dir2 == '-Y')
        else:
            # shield2 is below (-Y) shield1
            return (dir1 == '-Y' and dir2 == '+Y')


def detect_aisles(aisle_signs: List[Dict],
                  shelves: List[Dict],
                  proximity_threshold: float = 5.0) -> List[Dict]:
    """
    Detect aisles based on AisleSign locations.

    Aisles are the spaces BETWEEN shelf rows, running parallel to shelves.
    They span from one shelf row to the adjacent shelf row.

    Args:
        aisle_signs: List of AisleSign data
        shelves: List of detected shelf areas
        proximity_threshold: How close signs must be to shelf ends (meters)

    Returns:
        List of aisle dictionaries
    """
    if not aisle_signs or not shelves:
        return []

    aisles = []
    used_signs = set()

    # Group aisle signs by proximity (find pairs at opposite ends)
    for i, sign1 in enumerate(aisle_signs):
        if i in used_signs:
            continue

        x1, y1 = sign1['position']['x'], sign1['position']['y']

        for j, sign2 in enumerate(aisle_signs):
            if j <= i or j in used_signs:
                continue

            x2, y2 = sign2['position']['x'], sign2['position']['y']
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx*dx + dy*dy)

            # Signs should be separated (opposite ends of aisle)
            if distance < 5.0:  # Too close, not opposite ends
                continue

            # Determine orientation
            alignment_tolerance = 3.0

            if abs(dx) > abs(dy):
                # Aisle runs along X axis
                if abs(dy) > alignment_tolerance:
                    continue
                orientation = 'horizontal'
            else:
                # Aisle runs along Y axis
                if abs(dx) > alignment_tolerance:
                    continue
                orientation = 'vertical'

            # Find shelves on BOTH SIDES of the aisle
            # Aisle sits BETWEEN two shelf rows
            adjacent_shelves = []
            shelf_bounds_on_sides = []

            for shelf in shelves:
                # Check if this shelf runs parallel to the aisle
                if shelf['orientation'] != orientation:
                    continue

                shelf_center = shelf['bounds']['center']
                sx, sy = shelf_center['x'], shelf_center['y']

                # Check if shelf is adjacent to this aisle line
                if orientation == 'horizontal':
                    # Aisle runs along X, check Y proximity
                    avg_y = (y1 + y2) / 2
                    y_dist = abs(sy - avg_y)

                    if y_dist < proximity_threshold:
                        # Check X overlap
                        shelf_x_min = shelf['bounds']['min']['x']
                        shelf_x_max = shelf['bounds']['max']['x']
                        aisle_x_min = min(x1, x2)
                        aisle_x_max = max(x1, x2)
                        if not (shelf_x_max < aisle_x_min or shelf_x_min > aisle_x_max):
                            adjacent_shelves.append(shelf['id'])
                            # Store shelf Y bounds for later
                            shelf_bounds_on_sides.append({
                                'min': shelf['bounds']['min']['y'],
                                'max': shelf['bounds']['max']['y'],
                                'side': 'above' if sy > avg_y else 'below'
                            })

                else:  # vertical
                    # Aisle runs along Y, check X proximity
                    avg_x = (x1 + x2) / 2
                    x_dist = abs(sx - avg_x)

                    if x_dist < proximity_threshold:
                        # Check Y overlap
                        shelf_y_min = shelf['bounds']['min']['y']
                        shelf_y_max = shelf['bounds']['max']['y']
                        aisle_y_min = min(y1, y2)
                        aisle_y_max = max(y1, y2)
                        if not (shelf_y_max < aisle_y_min or shelf_y_min > aisle_y_max):
                            adjacent_shelves.append(shelf['id'])
                            # Store shelf X bounds for later
                            shelf_bounds_on_sides.append({
                                'min': shelf['bounds']['min']['x'],
                                'max': shelf['bounds']['max']['x'],
                                'side': 'right' if sx > avg_x else 'left'
                            })

            # Must have at least one adjacent shelf to be valid
            if not adjacent_shelves or not shelf_bounds_on_sides:
                continue

            # Compute aisle bounds BETWEEN shelves using shelf edges
            # The aisle length is determined by the span of the sign positions
            # The aisle width is the gap between shelf rows

            if orientation == 'horizontal':
                # Aisle runs along X axis
                # Length: span from one sign to the other along X
                min_x = min(x1, x2)
                max_x = max(x1, x2)

                # Width: gap between shelves in Y direction
                shelves_above = [s for s in shelf_bounds_on_sides if s['side'] == 'above']
                shelves_below = [s for s in shelf_bounds_on_sides if s['side'] == 'below']

                if shelves_above and shelves_below:
                    # Aisle is between two shelf rows
                    # min_y = max of shelf maxes below, max_y = min of shelf mins above
                    min_y = max(s['max'] for s in shelves_below)
                    max_y = min(s['min'] for s in shelves_above)
                elif shelves_above:
                    # Only shelf above, aisle extends down
                    max_y = min(s['min'] for s in shelves_above)
                    min_y = max_y - 2.5  # default width
                else:  # only below
                    # Only shelf below, aisle extends up
                    min_y = max(s['max'] for s in shelves_below)
                    max_y = min_y + 2.5  # default width

            else:  # vertical
                # Aisle runs along Y axis
                # Length: span from one sign to the other along Y
                min_y = min(y1, y2)
                max_y = max(y1, y2)

                # Width: gap between shelves in X direction
                shelves_right = [s for s in shelf_bounds_on_sides if s['side'] == 'right']
                shelves_left = [s for s in shelf_bounds_on_sides if s['side'] == 'left']

                if shelves_right and shelves_left:
                    # Aisle is between two shelf rows
                    min_x = max(s['max'] for s in shelves_left)
                    max_x = min(s['min'] for s in shelves_right)
                elif shelves_right:
                    # Only shelf to right, aisle extends left
                    max_x = min(s['min'] for s in shelves_right)
                    min_x = max_x - 2.5  # default width
                else:  # only left
                    # Only shelf to left, aisle extends right
                    min_x = max(s['max'] for s in shelves_left)
                    max_x = min_x + 2.5  # default width

            # Compute aisle dimensions
            aisle_width = (max_y - min_y) if orientation == 'horizontal' else (max_x - min_x)
            aisle_length = (max_x - min_x) if orientation == 'horizontal' else (max_y - min_y)

            aisle = {
                'id': f"aisle_{len(aisles)}",
                'aisle_sign_pair': [sign1['name'], sign2['name']],
                'orientation': orientation,
                'bounds': {
                    'min': {'x': min_x, 'y': min_y, 'z': 0.0},
                    'max': {'x': max_x, 'y': max_y, 'z': 2.5},
                    'center': {
                        'x': (min_x + max_x) / 2,
                        'y': (min_y + max_y) / 2,
                        'z': 1.25
                    },
                    'size': {
                        'x': max_x - min_x,
                        'y': max_y - min_y,
                        'z': 2.5
                    }
                },
                'length': aisle_length,
                'width': aisle_width,
                'adjacent_shelves': adjacent_shelves,
                'sign_positions': [
                    sign1['position'],
                    sign2['position']
                ]
            }

            aisles.append(aisle)
            used_signs.add(i)
            used_signs.add(j)
            break

    # Merge overlapping aisles
    aisles = merge_overlapping_aisles(aisles)

    return aisles


def merge_overlapping_aisles(aisles: List[Dict]) -> List[Dict]:
    """
    Merge aisle areas that overlap or are adjacent.
    This happens when multiple aisle sign pairs define the same physical aisle.
    """
    if len(aisles) <= 1:
        return aisles

    merged = []
    used = set()

    for i, aisle1 in enumerate(aisles):
        if i in used:
            continue

        # Start with this aisle
        merged_aisle = {
            'id': f"aisle_{len(merged)}",
            'aisle_sign_pair': aisle1['aisle_sign_pair'].copy(),
            'orientation': aisle1['orientation'],
            'bounds': aisle1['bounds'].copy(),
            'adjacent_shelves': aisle1['adjacent_shelves'].copy(),
            'sign_positions': aisle1['sign_positions'].copy()
        }

        # Try to merge with other aisles
        for j, aisle2 in enumerate(aisles):
            if j <= i or j in used:
                continue

            # Must have same orientation
            if aisle1['orientation'] != aisle2['orientation']:
                continue

            # Check if they overlap or are adjacent
            b1 = merged_aisle['bounds']
            b2 = aisle2['bounds']

            # Check for overlap/adjacency with small tolerance
            tolerance = 0.5
            x_overlap = not (b1['max']['x'] < b2['min']['x'] - tolerance or
                           b1['min']['x'] > b2['max']['x'] + tolerance)
            y_overlap = not (b1['max']['y'] < b2['min']['y'] - tolerance or
                           b1['min']['y'] > b2['max']['y'] + tolerance)

            if x_overlap and y_overlap:
                # Merge bounds
                merged_aisle['bounds']['min']['x'] = min(b1['min']['x'], b2['min']['x'])
                merged_aisle['bounds']['min']['y'] = min(b1['min']['y'], b2['min']['y'])
                merged_aisle['bounds']['max']['x'] = max(b1['max']['x'], b2['max']['x'])
                merged_aisle['bounds']['max']['y'] = max(b1['max']['y'], b2['max']['y'])

                # Update center and size
                merged_aisle['bounds']['center']['x'] = (merged_aisle['bounds']['min']['x'] +
                                                         merged_aisle['bounds']['max']['x']) / 2
                merged_aisle['bounds']['center']['y'] = (merged_aisle['bounds']['min']['y'] +
                                                         merged_aisle['bounds']['max']['y']) / 2
                merged_aisle['bounds']['size']['x'] = (merged_aisle['bounds']['max']['x'] -
                                                       merged_aisle['bounds']['min']['x'])
                merged_aisle['bounds']['size']['y'] = (merged_aisle['bounds']['max']['y'] -
                                                       merged_aisle['bounds']['min']['y'])

                # Merge sign pairs and shelves
                merged_aisle['aisle_sign_pair'].extend(aisle2['aisle_sign_pair'])
                merged_aisle['sign_positions'].extend(aisle2['sign_positions'])
                for shelf_id in aisle2['adjacent_shelves']:
                    if shelf_id not in merged_aisle['adjacent_shelves']:
                        merged_aisle['adjacent_shelves'].append(shelf_id)

                used.add(j)

        # Compute final length and width
        if merged_aisle['orientation'] == 'horizontal':
            merged_aisle['length'] = merged_aisle['bounds']['size']['x']
            merged_aisle['width'] = merged_aisle['bounds']['size']['y']
        else:
            merged_aisle['length'] = merged_aisle['bounds']['size']['y']
            merged_aisle['width'] = merged_aisle['bounds']['size']['x']

        merged.append(merged_aisle)
        used.add(i)

    return merged


def detect_shelves_and_sections(shelf_areas: List[Dict],
                                rack_shelves: List[Dict],
                                rack_frames: List[Dict],
                                floor_top_z: float) -> List[Dict]:
    """
    Create Shelf objects with nested Shelf Section bounding boxes.

    Shelf: 3D bounding box defined by rackshield pair XY area, from floor
    to tallest RackFrame within that shelf XY area.

    Shelf Section: 3D subdivision within a shelf:
    - Bottom: floor_top to lowest rackshelf top, XY from that lowest rackshelf
    - Intermediate: base rackshelf top to next rackshelf top above, XY from base
    - Top: highest rackshelf top to rackframe top, XY from that highest rackshelf

    Args:
        shelf_areas: List of detected shelf areas (XY bounds from rackshield pairs)
        rack_shelves: List of all RackShelf objects
        rack_frames: List of all RackFrame objects
        floor_top_z: Z coordinate of the floor top surface (max bbox_max_z of floor tiles)

    Returns:
        List of Shelf dictionaries, each containing a 'sections' list
    """
    shelves = []

    for shelf_area in shelf_areas:
        area_bounds = shelf_area['bounds']
        area_min_x = area_bounds['min']['x']
        area_max_x = area_bounds['max']['x']
        area_min_y = area_bounds['min']['y']
        area_max_y = area_bounds['max']['y']

        # Find all RackShelves within this shelf area's XY bounds
        shelves_in_area = []
        for rs in rack_shelves:
            cx = rs['bounding_box']['center']['x']
            cy = rs['bounding_box']['center']['y']
            if area_min_x <= cx <= area_max_x and area_min_y <= cy <= area_max_y:
                shelves_in_area.append(rs)

        # Find all RackFrames within this shelf area's XY bounds
        frames_in_area = []
        for rf in rack_frames:
            cx = rf['bounding_box']['center']['x']
            cy = rf['bounding_box']['center']['y']
            if area_min_x <= cx <= area_max_x and area_min_y <= cy <= area_max_y:
                frames_in_area.append(rf)

        # Determine shelf Z max from tallest rackframe
        if frames_in_area:
            shelf_z_max = max(f['bounding_box']['max']['z'] for f in frames_in_area)
        elif shelves_in_area:
            shelf_z_max = max(rs['bounding_box']['max']['z'] for rs in shelves_in_area) + 3.0
        else:
            continue

        # Build shelf bounds: XY from rackshield pair, Z from floor to rackframe top
        shelf_bounds = {
            'min': {'x': area_min_x, 'y': area_min_y, 'z': floor_top_z},
            'max': {'x': area_max_x, 'y': area_max_y, 'z': shelf_z_max},
            'center': {
                'x': (area_min_x + area_max_x) / 2,
                'y': (area_min_y + area_max_y) / 2,
                'z': (floor_top_z + shelf_z_max) / 2
            },
            'size': {
                'x': area_max_x - area_min_x,
                'y': area_max_y - area_min_y,
                'z': shelf_z_max - floor_top_z
            }
        }

        sections = []

        if shelves_in_area:
            # For each rackshelf, find "directly above" and "directly below"
            # "Directly above R" means: A's XY center is within R's XY bbox,
            #   A's bbox_max_z > R's bbox_max_z, and A has the lowest such z
            # "Directly below R" means: R's XY center is within B's XY bbox,
            #   B's bbox_max_z < R's bbox_max_z, and B has the highest such z
            above_map = {}  # index -> index of shelf directly above (or None)
            below_map = {}  # index -> index of shelf directly below (or None)

            for i, rs in enumerate(shelves_in_area):
                rs_bbox = rs['bounding_box']
                rs_z_top = rs_bbox['max']['z']

                # Find directly above
                best_above = None
                best_above_z = float('inf')
                for j, other in enumerate(shelves_in_area):
                    if j == i:
                        continue
                    other_z_top = other['bounding_box']['max']['z']
                    if other_z_top <= rs_z_top + 0.01:
                        continue
                    other_cx = other['bounding_box']['center']['x']
                    other_cy = other['bounding_box']['center']['y']
                    if (rs_bbox['min']['x'] <= other_cx <= rs_bbox['max']['x'] and
                        rs_bbox['min']['y'] <= other_cy <= rs_bbox['max']['y']):
                        if other_z_top < best_above_z:
                            best_above_z = other_z_top
                            best_above = j
                above_map[i] = best_above

                # Find directly below
                best_below = None
                best_below_z = float('-inf')
                rs_cx = rs_bbox['center']['x']
                rs_cy = rs_bbox['center']['y']
                for j, other in enumerate(shelves_in_area):
                    if j == i:
                        continue
                    other_z_top = other['bounding_box']['max']['z']
                    if other_z_top >= rs_z_top - 0.01:
                        continue
                    other_bbox = other['bounding_box']
                    if (other_bbox['min']['x'] <= rs_cx <= other_bbox['max']['x'] and
                        other_bbox['min']['y'] <= rs_cy <= other_bbox['max']['y']):
                        if other_z_top > best_below_z:
                            best_below_z = other_z_top
                            best_below = j
                below_map[i] = best_below

            # Assign rackshelf level indices based on unique Z heights
            z_tops = sorted(set(
                round(rs['bounding_box']['max']['z'], 4) for rs in shelves_in_area
            ))
            z_to_level = {z: idx for idx, z in enumerate(z_tops)}

            # Create sections
            section_idx = 0

            for i, rs in enumerate(shelves_in_area):
                rs_bbox = rs['bounding_box']
                is_lowest = below_map[i] is None
                is_highest = above_map[i] is None
                rs_level = z_to_level[round(rs_bbox['max']['z'], 4)]

                # Bottom section (only for lowest rackshelves in their column)
                if is_lowest:
                    section = {
                        'id': f"{shelf_area['id']}_section_{section_idx}",
                        'level_index': 0,
                        'level_type': 'bottom',
                        'bounds': {
                            'min': {
                                'x': rs_bbox['min']['x'],
                                'y': rs_bbox['min']['y'],
                                'z': floor_top_z
                            },
                            'max': {
                                'x': rs_bbox['max']['x'],
                                'y': rs_bbox['max']['y'],
                                'z': rs_bbox['max']['z']
                            },
                            'center': {
                                'x': rs_bbox['center']['x'],
                                'y': rs_bbox['center']['y'],
                                'z': (floor_top_z + rs_bbox['max']['z']) / 2
                            },
                            'size': {
                                'x': rs_bbox['max']['x'] - rs_bbox['min']['x'],
                                'y': rs_bbox['max']['y'] - rs_bbox['min']['y'],
                                'z': rs_bbox['max']['z'] - floor_top_z
                            }
                        },
                        'base_rackshelf': None,
                        'xy_defining_rackshelf': rs['name'],
                        'upper_rackshelf': rs['name']
                    }
                    sections.append(section)
                    section_idx += 1

                # Section above this rackshelf (intermediate or top)
                if not is_highest:
                    # Intermediate: from this shelf top to next shelf top above
                    above_idx = above_map[i]
                    above_rs = shelves_in_area[above_idx]
                    above_bbox = above_rs['bounding_box']

                    section = {
                        'id': f"{shelf_area['id']}_section_{section_idx}",
                        'level_index': rs_level + 1,
                        'level_type': 'intermediate',
                        'bounds': {
                            'min': {
                                'x': rs_bbox['min']['x'],
                                'y': rs_bbox['min']['y'],
                                'z': rs_bbox['max']['z']
                            },
                            'max': {
                                'x': rs_bbox['max']['x'],
                                'y': rs_bbox['max']['y'],
                                'z': above_bbox['max']['z']
                            },
                            'center': {
                                'x': rs_bbox['center']['x'],
                                'y': rs_bbox['center']['y'],
                                'z': (rs_bbox['max']['z'] + above_bbox['max']['z']) / 2
                            },
                            'size': {
                                'x': rs_bbox['max']['x'] - rs_bbox['min']['x'],
                                'y': rs_bbox['max']['y'] - rs_bbox['min']['y'],
                                'z': above_bbox['max']['z'] - rs_bbox['max']['z']
                            }
                        },
                        'base_rackshelf': rs['name'],
                        'xy_defining_rackshelf': rs['name'],
                        'upper_rackshelf': above_rs['name']
                    }
                    sections.append(section)
                    section_idx += 1
                else:
                    # Top: from this shelf top to rackframe top
                    section = {
                        'id': f"{shelf_area['id']}_section_{section_idx}",
                        'level_index': rs_level + 1,
                        'level_type': 'top',
                        'bounds': {
                            'min': {
                                'x': rs_bbox['min']['x'],
                                'y': rs_bbox['min']['y'],
                                'z': rs_bbox['max']['z']
                            },
                            'max': {
                                'x': rs_bbox['max']['x'],
                                'y': rs_bbox['max']['y'],
                                'z': shelf_z_max
                            },
                            'center': {
                                'x': rs_bbox['center']['x'],
                                'y': rs_bbox['center']['y'],
                                'z': (rs_bbox['max']['z'] + shelf_z_max) / 2
                            },
                            'size': {
                                'x': rs_bbox['max']['x'] - rs_bbox['min']['x'],
                                'y': rs_bbox['max']['y'] - rs_bbox['min']['y'],
                                'z': shelf_z_max - rs_bbox['max']['z']
                            }
                        },
                        'base_rackshelf': rs['name'],
                        'xy_defining_rackshelf': rs['name'],
                        'upper_rackshelf': None
                    }
                    sections.append(section)
                    section_idx += 1

            # Sort sections for consistent output
            sections.sort(key=lambda s: (
                s['level_index'],
                s['bounds']['min']['y'],
                s['bounds']['min']['x']
            ))
            # Re-number section IDs after sorting
            for idx, section in enumerate(sections):
                section['id'] = f"{shelf_area['id']}_section_{idx}"

        shelf = {
            'id': shelf_area['id'],
            'rackshield_pair': shelf_area['rackshield_pair'],
            'orientation': shelf_area['orientation'],
            'bounds': shelf_bounds,
            'rackframe_objects': [f['name'] for f in frames_in_area],
            'section_count': len(sections),
            'sections': sections
        }
        shelves.append(shelf)

    return shelves


def detect_shelf_areas(rackshields: List[Dict],
                       max_distance: float = 20.0,
                       alignment_tolerance: float = 2.0) -> List[Dict]:
    """
    Detect shelf areas by finding pairs of rackshields facing each other.

    Args:
        rackshields: List of rackshield data
        max_distance: Maximum distance between paired rackshields
        alignment_tolerance: How closely aligned shields must be (meters)

    Returns:
        List of shelf area dictionaries
    """
    shelf_areas = []
    used_shields = set()

    for i, shield1 in enumerate(rackshields):
        if i in used_shields:
            continue

        for j, shield2 in enumerate(rackshields):
            if j <= i or j in used_shields:
                continue

            # Check if they're facing each other
            if not are_shields_facing(shield1, shield2):
                continue

            x1, y1 = shield1['position']['x'], shield1['position']['y']
            x2, y2 = shield2['position']['x'], shield2['position']['y']

            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx*dx + dy*dy)

            # Too far apart
            if distance > max_distance:
                continue

            # Check alignment based on facing direction
            dir1 = get_facing_direction(shield1['rotation']['yaw'])

            if dir1 in ['+X', '-X']:
                # Facing along X, should be aligned in Y
                if abs(dy) > alignment_tolerance:
                    continue
            else:  # '+Y' or '-Y'
                # Facing along Y, should be aligned in X
                if abs(dx) > alignment_tolerance:
                    continue

            # Found a valid shelf area!
            # Create bounding box for the shelf area
            min_x = min(shield1['bounding_box']['min']['x'],
                       shield2['bounding_box']['min']['x'])
            max_x = max(shield1['bounding_box']['max']['x'],
                       shield2['bounding_box']['max']['x'])
            min_y = min(shield1['bounding_box']['min']['y'],
                       shield2['bounding_box']['min']['y'])
            max_y = max(shield1['bounding_box']['max']['y'],
                       shield2['bounding_box']['max']['y'])
            min_z = min(shield1['bounding_box']['min']['z'],
                       shield2['bounding_box']['min']['z'])
            max_z = max(shield1['bounding_box']['max']['z'],
                       shield2['bounding_box']['max']['z'])

            shelf_area = {
                'id': f"shelf_{len(shelf_areas)}",
                'rackshield_pair': [shield1['name'], shield2['name']],
                'orientation': 'horizontal' if dir1 in ['+X', '-X'] else 'vertical',
                'bounds': {
                    'min': {'x': min_x, 'y': min_y, 'z': min_z},
                    'max': {'x': max_x, 'y': max_y, 'z': max_z},
                    'center': {
                        'x': (min_x + max_x) / 2,
                        'y': (min_y + max_y) / 2,
                        'z': (min_z + max_z) / 2
                    },
                    'size': {
                        'x': max_x - min_x,
                        'y': max_y - min_y,
                        'z': max_z - min_z
                    }
                },
                'distance_between_shields': distance,
                'shield_positions': [
                    shield1['position'],
                    shield2['position']
                ]
            }

            shelf_areas.append(shelf_area)
            used_shields.add(i)
            used_shields.add(j)
            break  # Move to next shield1

    return shelf_areas


def compute_floor_polygon(tiles: List[Dict]) -> Dict:
    """Combine all floor tiles into a unified polygon shape."""
    if not tiles:
        return {}

    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        return compute_simple_bounds(tiles)

    rectangles = []
    for tile in tiles:
        min_x, min_y = tile['bbox_min']
        max_x, max_y = tile['bbox_max']
        rectangles.append(box(min_x, min_y, max_x, max_y))

    unified_polygon = unary_union(rectangles)

    # Close micro-gaps between tiles caused by floating-point precision,
    # then simplify to remove redundant vertices at tile boundaries.
    # buffer(eps).buffer(-eps) merges parts separated by < 2*eps.
    unified_polygon = unified_polygon.buffer(0.01).buffer(-0.01).simplify(0.01)

    if unified_polygon.is_empty:
        return compute_simple_bounds(tiles)
    elif unified_polygon.geom_type == 'Polygon':
        coords = list(unified_polygon.exterior.coords)
    elif unified_polygon.geom_type == 'MultiPolygon':
        largest = max(unified_polygon.geoms, key=lambda p: p.area)
        coords = list(largest.exterior.coords)
    else:
        return compute_simple_bounds(tiles)

    polygon_vertices = [[float(x), float(y)] for x, y in coords[:-1]]

    all_x = [v[0] for v in polygon_vertices]
    all_y = [v[1] for v in polygon_vertices]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    avg_z = sum(t['z'] for t in tiles) / len(tiles)

    return {
        'type': 'polygon',
        'vertices': polygon_vertices,
        'vertex_count': len(polygon_vertices),
        'bounds': {
            'min': {'x': min_x, 'y': min_y, 'z': avg_z},
            'max': {'x': max_x, 'y': max_y, 'z': avg_z},
            'center': {
                'x': (min_x + max_x) / 2,
                'y': (min_y + max_y) / 2,
                'z': avg_z
            },
            'size': {
                'x': max_x - min_x,
                'y': max_y - min_y,
                'z': 0.0
            }
        },
        'area': float(unified_polygon.area)
    }


def compute_simple_bounds(tiles: List[Dict]) -> Dict:
    """Fallback: compute simple rectangular bounding box."""
    if not tiles:
        return {}

    all_min_x = [t['bbox_min'][0] for t in tiles]
    all_min_y = [t['bbox_min'][1] for t in tiles]
    all_max_x = [t['bbox_max'][0] for t in tiles]
    all_max_y = [t['bbox_max'][1] for t in tiles]

    min_x = min(all_min_x)
    min_y = min(all_min_y)
    max_x = max(all_max_x)
    max_y = max(all_max_y)

    avg_z = sum(t['z'] for t in tiles) / len(tiles)

    vertices = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

    return {
        'type': 'polygon',
        'vertices': vertices,
        'vertex_count': 4,
        'bounds': {
            'min': {'x': min_x, 'y': min_y, 'z': avg_z},
            'max': {'x': max_x, 'y': max_y, 'z': avg_z},
            'center': {
                'x': (min_x + max_x) / 2,
                'y': (min_y + max_y) / 2,
                'z': avg_z
            },
            'size': {
                'x': max_x - min_x,
                'y': max_y - min_y,
                'z': 0.0
            }
        },
        'area': (max_x - min_x) * (max_y - min_y)
    }


def create_warehouse_layout(csv_path: Path, output_json_path: Optional[Path] = None) -> Dict:
    """
    Create hierarchical warehouse layout JSON.

    Hierarchy:
    - Floor
      - Functional Zones (Receiving, Staging, Pallet Truck, Hub Robot, Forklift, Storage, General)
        - Aisles
          - Shelves
            - Shelf Sections
    """
    # Parse data
    floor_tiles = parse_floor_tiles(csv_path)
    rackshields = parse_rackshields(csv_path)
    aisle_signs = parse_aisle_signs(csv_path)
    rack_shelves = parse_rack_shelves(csv_path)
    rack_frames = parse_rack_frames(csv_path)
    receiving_wall = parse_asset_by_name(csv_path, 'sm_wall_a01_01')
    staging_tapes = parse_staging_tapes(csv_path)
    staging_bounds = compute_staging_zone_bounds(staging_tapes)
    reflective_tapes = parse_reflective_tapes(csv_path)
    tape_zone_rects = detect_tape_zones(reflective_tapes)
    zone_indicators = parse_zone_indicator_assets(csv_path)
    identified_tape_zones = identify_tape_zones(tape_zone_rects, zone_indicators)

    # Compute floor and clip to wall boundary
    floor_polygon = compute_floor_polygon(floor_tiles)
    boundary_walls = parse_boundary_walls(csv_path)
    wall_boundary = compute_wall_boundary(boundary_walls)
    if wall_boundary:
        floor_polygon = clip_floor_to_boundary(floor_polygon, wall_boundary)
    floor_top_z = max(t['bbox_max_z'] for t in floor_tiles) if floor_tiles else 0.0

    # Detect shelf areas (XY bounds defined by rackshield pairs)
    shelf_areas = detect_shelf_areas(rackshields)

    # Detect shelves with nested shelf sections
    shelves = detect_shelves_and_sections(shelf_areas, rack_shelves, rack_frames, floor_top_z)

    # Detect aisles (uses shelf bounds for adjacency)
    aisles = detect_aisles(aisle_signs, shelves)

    # Build adjacent_aisles reverse lookup for shelves
    shelf_to_aisles: Dict[str, List[str]] = {}
    for aisle in aisles:
        for sid in aisle.get('adjacent_shelves', []):
            shelf_to_aisles.setdefault(sid, []).append(aisle['id'])
    for shelf in shelves:
        shelf['adjacent_aisles'] = shelf_to_aisles.get(shelf['id'], [])

    # Restructure sections: flat list -> dict indexed by level and horizontal position
    for shelf in shelves:
        sections_list = shelf.get('sections', [])
        if not sections_list:
            shelf['sections'] = {}
            continue

        # Group by level_index
        levels: Dict[int, List[Dict]] = {}
        for sec in sections_list:
            lvl = sec['level_index']
            levels.setdefault(lvl, []).append(sec)

        # Sort each level by horizontal position along the shelf
        sort_key = 'x' if shelf['orientation'] == 'horizontal' else 'y'
        sections_dict: Dict[str, Dict[str, Dict]] = {}
        for lvl, secs in sorted(levels.items()):
            secs.sort(key=lambda s: s['bounds']['min'][sort_key])
            level_dict: Dict[str, Dict] = {}
            for h_idx, sec in enumerate(secs):
                level_dict[str(h_idx)] = sec
            sections_dict[str(lvl)] = level_dict

        shelf['sections'] = sections_dict

    # Detect functional zones
    functional_zones = detect_functional_zones(
        floor_polygon, receiving_wall, staging_bounds,
        identified_tape_zones, aisles, shelves)

    # Add functional_zone reference to each aisle and shelf
    zone_lookup = {}
    for zone in functional_zones:
        for aid in zone['aisles']:
            zone_lookup[('aisle', aid)] = zone['id']
        for sid in zone['shelves']:
            zone_lookup[('shelf', sid)] = zone['id']
    for aisle in aisles:
        aisle['functional_zone'] = zone_lookup.get(('aisle', aisle['id']))
    for shelf in shelves:
        shelf['functional_zone'] = zone_lookup.get(('shelf', shelf['id']))

    # Embed full aisle and shelf objects inside their zones
    aisle_map = {a['id']: a for a in aisles}
    shelf_map = {s['id']: s for s in shelves}
    for zone in functional_zones:
        zone_aisle_ids = zone['aisles']
        zone['aisles'] = {aid: aisle_map[aid] for aid in zone_aisle_ids
                          if aid in aisle_map}
        zone_shelf_ids = zone['shelves']
        zone['shelves'] = {sid: shelf_map[sid] for sid in zone_shelf_ids
                           if sid in shelf_map}

    # Compute total section count
    total_sections = sum(s['section_count'] for s in shelves)

    # Create layout
    warehouse_layout = {
        'metadata': {
            'source_csv': str(csv_path),
            'description': 'Hierarchical warehouse layout with floor, functional zones, aisles, shelves, and shelf sections',
            'coordinate_system': 'Isaac Sim world frame (X: forward, Y: left, Z: up)',
            'units': 'meters',
            'floor_tile_count': len(floor_tiles),
            'boundary_wall_count': len(boundary_walls),
            'floor_top_z': floor_top_z,
            'rackshield_count': len(rackshields),
            'rackshelf_count': len(rack_shelves),
            'rackframe_count': len(rack_frames),
            'aislesign_count': len(aisle_signs),
            'packing_tape_count': len(staging_tapes),
            'reflective_tape_count': len(reflective_tapes),
            'tape_zone_count': len(identified_tape_zones),
            'functional_zone_count': len(functional_zones),
            'aisle_count': len(aisles),
            'shelf_count': len(shelves),
            'shelf_section_count': total_sections
        },
        'floor': floor_polygon,
        'functional_zones': functional_zones
    }

    # Write to JSON
    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open('w', encoding='utf-8') as f:
            json.dump(warehouse_layout, f, indent=2)

        print(f"Warehouse layout written to: {output_json_path}")
        print(f"Floor: Combined {len(floor_tiles)} tiles into unified polygon")
        if floor_polygon:
            bounds = floor_polygon.get('bounds', {})
            if bounds:
                print(f"  Bounds: X[{bounds['min']['x']:.2f}, {bounds['max']['x']:.2f}], "
                      f"Y[{bounds['min']['y']:.2f}, {bounds['max']['y']:.2f}]")
                print(f"  Area: {floor_polygon.get('area', 0):.2f} m²")
                print(f"  Floor top Z: {floor_top_z:.4f}")

        print(f"\nFunctional Zones: {len(functional_zones)}")
        for zone in functional_zones:
            zb = zone['bounds']
            print(f"  {zone['id']} ({zone['name']}): "
                  f"X[{zb['min']['x']:.2f}, {zb['max']['x']:.2f}], "
                  f"Y[{zb['min']['y']:.2f}, {zb['max']['y']:.2f}], "
                  f"area={zone['area']:.1f} m², "
                  f"{len(zone['aisles'])} aisles, {len(zone['shelves'])} shelves")

        print(f"\nAisles: Detected {len(aisles)} aisles from {len(aisle_signs)} aisle signs")
        for aisle in aisles:
            print(f"  {aisle['id']}: {aisle['orientation']}, "
                  f"length={aisle['length']:.2f}m, width={aisle['width']:.2f}m, "
                  f"zone={aisle['functional_zone']}")

        print(f"\nShelves: {len(shelves)} shelves from {len(rackshields)} rackshields")
        for shelf in shelves:
            print(f"  {shelf['id']}: {shelf['orientation']}, "
                  f"Z=[{shelf['bounds']['min']['z']:.2f}, {shelf['bounds']['max']['z']:.2f}]m, "
                  f"{shelf['section_count']} sections, zone={shelf['functional_zone']}")

        print(f"\nShelf Sections: {total_sections} total from {len(rack_shelves)} rack shelves")

    return warehouse_layout


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate warehouse layout JSON from object_lists.csv'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'warehouse_layout' / 'object_lists.csv',
        help='Path to object_lists.csv'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'warehouse_layout' / 'warehouse_layout.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--max-distance',
        type=float,
        default=20.0,
        help='Maximum distance between paired rackshields (meters)'
    )
    parser.add_argument(
        '--alignment-tolerance',
        type=float,
        default=2.0,
        help='Alignment tolerance for shield pairing (meters)'
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        exit(1)

    create_warehouse_layout(args.csv, args.output)
