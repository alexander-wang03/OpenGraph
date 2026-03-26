#!/usr/bin/env python3
"""
Build TierGraph - Hierarchical Warehouse Scene Graph

Loads OpenGraph results (objects with 3D positions, captions, features) and
assigns them to a 5-level warehouse hierarchy using geometric containment.

This replaces OpenGraph's flat MST-based scene graph with TierGraph's
structured hierarchy: Zone → Aisle → Shelf → Section → Object

Usage:
    python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01

Author: awang (TierGraph thesis)
Date: 2026-02-10
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import gzip
import pickle
import hydra
import json
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from some_class.map_calss import MapObjectList
from utils.warehouse_graph_builder import WarehouseGraphBuilder
from utils.coordinate_alignment import get_aligned_warehouse_layout


def load_opengraph_results(result_path: Path):
    """
    Load OpenGraph results (objects with 3D point clouds and captions).

    Args:
        result_path: Path to full_pcd.pkl.gz

    Returns:
        objects: MapObjectList with OpenGraph detections
        bg_objects: Background objects (or None)
    """
    print(f"Loading OpenGraph results from {result_path}...")

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])

        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError(f"Unknown results type: {type(results)}")

    print(f"Loaded {len(objects)} objects from OpenGraph")
    if bg_objects:
        print(f"Loaded {len(bg_objects)} background objects")

    return objects, bg_objects


def load_frame_captions(caption_dir):
    """
    Load all captions from caption files.

    Returns:
        dict: {frame_idx: [list of captions for that frame]}
    """
    import gzip
    import pickle

    caption_dir = Path(caption_dir)
    if not caption_dir.exists():
        print(f"Warning: Caption directory not found: {caption_dir}")
        return {}

    caption_files = sorted(caption_dir.glob('cap_*.pkl.gz'))
    frame_captions = {}

    for cap_file in caption_files:
        # Extract frame index from filename (e.g., cap_000010.pkl.gz -> 10)
        frame_idx_str = cap_file.stem.split('_')[1]
        frame_idx = int(frame_idx_str)

        try:
            with gzip.open(cap_file, 'rb') as f:
                cap_data = pickle.load(f)

            if 'captions' in cap_data:
                frame_captions[frame_idx] = cap_data['captions']
        except Exception as e:
            print(f"  Warning: Could not load {cap_file.name}: {e}")

    print(f"Loaded captions from {len(frame_captions)} frames")
    return frame_captions


def extract_object_centroid(pcd, z_offset=0.781):
    """
    Extract 3D centroid from Open3D point cloud and apply Z offset.

    The Z offset accounts for the robot base_link being elevated above the floor.
    In robot frame, floor is at Z≈-0.781m. We add +0.781m to bring it to Z=0.0m,
    matching the warehouse layout (which is offset from -0.14m to 0.0m separately).

    Args:
        pcd: open3d.geometry.PointCloud
        z_offset: Z offset to add (default 0.781m, computed from floor Z in point clouds)

    Returns:
        centroid: np.ndarray [x, y, z] with Z offset applied
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.array([0.0, 0.0, 0.0])

    centroid = np.mean(points, axis=0)
    # Apply Z offset to align with warehouse layout
    centroid[2] += z_offset

    return centroid


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    """Main function to build TierGraph from OpenGraph results."""

    print("\n" + "="*60)
    print("BUILDING TIERGRAPH - Hierarchical Warehouse Scene Graph")
    print("="*60 + "\n")

    # Paths
    result_path = Path(cfg.result_path)
    warehouse_layout_path = "/home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json"
    output_path = Path(cfg.scenegraph_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine sequence directory
    sequence_dir = Path(cfg.basedir) / cfg.sequence

    # Load OpenGraph results
    objects, bg_objects = load_opengraph_results(result_path)

    # Load captions from caption directory
    caption_dir = Path(cfg.basedir) / cfg.sequence.replace(cfg.basedir, '').strip('/')
    caption_dir = Path(cfg.basedir).parent / f"warehouse_{cfg.sequence}" / "caption"
    frame_captions = load_frame_captions(caption_dir)

    # Load warehouse layout and align to robot coordinates
    print(f"\nLoading warehouse layout from {warehouse_layout_path}...")
    aligned_layout = get_aligned_warehouse_layout(warehouse_layout_path, sequence_dir)

    # Save aligned layout for debugging
    aligned_layout_path = sequence_dir / "warehouse_layout_aligned.json"
    with open(aligned_layout_path, 'w') as f:
        json.dump(aligned_layout, f, indent=2)
    print(f"Saved aligned layout to: {aligned_layout_path}")

    # Initialize TierGraph builder with aligned layout
    # Save to temp file and load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(aligned_layout, f, indent=2)
        temp_layout_path = f.name

    builder = WarehouseGraphBuilder(temp_layout_path)

    # Clean up temp file
    import os
    os.unlink(temp_layout_path)

    # Geometric structural filter using warehouse layout geometry.
    #
    # Walls:  centroid XY is outside the warehouse floor polygon
    #         (the floor polygon gives the building boundary in robot-relative XY).
    # Floor:  centroid Z < FLOOR_Z_THRESHOLD after the +0.781m Z offset.
    #         Zone bounds start at Z=0.0 after layout alignment, so floor-level
    #         objects end up at Z ≈ 0–0.3m in centroid space.
    from matplotlib.path import Path as MplPath
    floor_vertices = np.array(aligned_layout['floor']['vertices'])  # (N, 2) XY
    floor_polygon = MplPath(floor_vertices)
    FLOOR_Z_THRESHOLD = 0.4  # metres; centroids below this are floor tiles
    MAX_EXTENT_THRESHOLD = 5.0  # metres; objects larger than this in any axis
                                # are structural (walls, ceilings, floor slabs)

    # Assign each OpenGraph object to the hierarchy
    print(f"\nAssigning {len(objects)} objects to warehouse hierarchy...")
    assigned_count = 0
    unassigned_count = 0
    filtered_count = 0

    for i, obj in enumerate(objects):
        # Extract object data
        object_id = f"object_{i}"
        pcd = obj['pcd']
        centroid = extract_object_centroid(pcd)

        # Get caption - try multiple sources
        caption = f"Object {i}"  # Default

        # Source 1: Caption stored in object
        if 'captions' in obj and obj['captions']:
            caption = obj['captions'][0]
        elif 'caption' in obj and obj['caption']:
            caption = obj['caption']
        # Source 2: Try to get from frame captions (approximate matching)
        elif frame_captions:
            # Use modulo to cycle through available captions
            all_captions = []
            for frame_caps in frame_captions.values():
                all_captions.extend(frame_caps)
            if all_captions:
                caption = all_captions[i % len(all_captions)]

        # Geometric structural filter: skip walls and floor tiles
        centroid_xy = centroid[:2]
        if not floor_polygon.contains_point(centroid_xy):
            print(f"  [FILTERED wall]  object_{i}: XY={centroid_xy} outside warehouse polygon")
            filtered_count += 1
            continue
        if centroid[2] < FLOOR_Z_THRESHOLD:
            # Exempt objects that fall inside a non-General functional zone —
            # they could be low pallets or boxes sitting on the floor.
            in_functional_zone = any(
                zone['id'] != 'zone_general'
                and zone['bounds']['min']['x'] <= centroid[0] <= zone['bounds']['max']['x']
                and zone['bounds']['min']['y'] <= centroid[1] <= zone['bounds']['max']['y']
                for zone in aligned_layout['functional_zones']
            )
            if not in_functional_zone:
                print(f"  [FILTERED floor] object_{i}: Z={centroid[2]:.3f}m below threshold, not in functional zone")
                filtered_count += 1
                continue

        # Extent filter: reject objects whose bounding box exceeds the
        # threshold in any axis — these are walls, ceilings, or floor slabs
        # that passed the centroid-based filters above.
        points = np.asarray(pcd.points)
        if len(points) > 0:
            extent = points.max(axis=0) - points.min(axis=0)
            max_extent = extent.max()
            if max_extent > MAX_EXTENT_THRESHOLD:
                print(f"  [FILTERED extent] object_{i}: max extent {max_extent:.1f}m > {MAX_EXTENT_THRESHOLD}m "
                      f"({extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f})")
                filtered_count += 1
                continue

        # Assign to hierarchy
        success, path = builder.assign_object_to_hierarchy(
            object_id=object_id,
            object_position=centroid,
            object_caption=caption,
            object_data=obj
        )

        if success:
            assigned_count += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(objects)}] {caption[:30]:30s} → {path}")
        else:
            unassigned_count += 1
            print(f"  [WARNING] Could not assign object {i}: {caption[:30]} at {centroid}")

    print(f"\nAssignment complete:")
    print(f"  Filtered (structural): {filtered_count} objects")
    print(f"  Assigned: {assigned_count} / {len(objects) - filtered_count} remaining objects")
    print(f"  Unassigned: {unassigned_count} objects")

    # Process background objects (if any)
    if bg_objects and len(bg_objects) > 0:
        print(f"\nProcessing {len(bg_objects)} background objects...")
        for i, bg_obj in enumerate(bg_objects):
            object_id = f"bg_object_{i}"
            pcd = bg_obj['pcd']
            centroid = extract_object_centroid(pcd)
            captions = bg_obj.get('captions', [])
            caption = captions[0] if captions else f"Background {i}"

            success, path = builder.assign_object_to_hierarchy(
                object_id=object_id,
                object_position=centroid,
                object_caption=caption,
                object_data=bg_obj
            )

            if success:
                assigned_count += 1

    # Print summary
    builder.print_summary()

    # Export hierarchical scene graph
    print(f"Exporting TierGraph to {output_path}...")
    builder.export_graph(str(output_path))

    # Compare with OpenGraph baseline
    print("\n" + "="*60)
    print("COMPARISON: OpenGraph (Baseline) vs TierGraph")
    print("="*60)
    print(f"OpenGraph structure:")
    print(f"  - Flat MST graph")
    print(f"  - {len(objects)} nodes (objects only)")
    print(f"  - ~{len(objects)-1} edges (MST has N-1 edges)")
    print(f"  - No hierarchical organization")
    print(f"\nTierGraph structure:")
    print(f"  - 5-level hierarchy (Zone→Aisle→Shelf→Section→Object)")
    print(f"  - {len(builder.nodes)} total nodes (infrastructure + objects)")
    print(f"  - {len(builder.edges)} edges (containment relationships)")
    print(f"  - Structured organization for efficient queries")
    print("="*60 + "\n")

    print("✓ TierGraph construction complete!")


if __name__ == "__main__":
    main()
