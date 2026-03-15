#!/usr/bin/env python3
"""
Compress TierGraph using the Agglomerative Information Bottleneck.

Takes an existing TierGraph JSON and a set of task queries, then compresses
semantically redundant objects into clusters — while preserving task-relevant
distinctions — using the hierarchy-aware AIB algorithm.

The adjacency graph for IB is derived from the TierGraph hierarchy (shelf /
aisle / zone groupings), not from bounding-box spatial proximity.  This
ensures merges never cross warehouse structural boundaries.

Usage:
    python script/compress_tiergraph.py \\
        --config-name=isaac_warehouse \\
        sequence=02 \\
        ib_config=config/ib_config.yaml

Output:
    results/warehouse_{seq}/pcd/object_relations_compressed.json

Author: awang (TierGraph thesis)
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import gzip
import json
import pickle
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from some_class.map_calss import MapObjectList
from utils.ib.ib_cluster    import ClusterIB, ClusterIBConfig
from utils.ib.tiergraph_bridge import (
    extract_clip_features,
    build_ib_graph,
    apply_clusters_to_tiergraph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tiergraph(path: Path) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict) or 'nodes' not in data:
        raise ValueError(
            f"{path} does not look like a TierGraph JSON (missing 'nodes'). "
            "Run build_tiergraph.py first."
        )
    return data


def load_opengraph_objects(result_path: Path) -> MapObjectList:
    with gzip.open(result_path, 'rb') as f:
        results = pickle.load(f)

    objects = MapObjectList()
    if isinstance(results, dict):
        objects.load_serializable(results['objects'])
    elif isinstance(results, list):
        objects.load_serializable(results)
    else:
        raise ValueError(f"Unexpected full_pcd type: {type(results)}")
    return objects


def encode_tasks_sbert(task_strings, sbert_model_name: str) -> np.ndarray:
    """
    Encode task query strings with SBERT (same model as OpenGraph 'ft' features).

    Returns (T, D) float64 numpy array.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(sbert_model_name)
    embeddings = model.encode(task_strings, convert_to_numpy=True,
                              normalize_embeddings=True)
    return embeddings.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-component IB
# ---------------------------------------------------------------------------

def run_ib_on_component(component_obj_ids, all_obj_id_to_row,
                         region_features, task_features, config):
    """
    Run ClusterIB on one connected component of the IB graph.

    Args:
        component_obj_ids: list of obj_id strings in this component
        all_obj_id_to_row: {obj_id: row_index} into region_features
        region_features:   (N_total, D) all object features
        task_features:     (T, D) task embeddings
        config:            ClusterIBConfig

    Returns:
        list[list[obj_id]]: cluster assignments for this component
    """
    # Gather features for this component
    rows = [all_obj_id_to_row[oid] for oid in component_obj_ids
            if oid in all_obj_id_to_row]
    if len(rows) < 2:
        # Single object or no features — return as-is
        return [[oid] for oid in component_obj_ids
                if oid in all_obj_id_to_row]

    comp_features = region_features[rows]  # (K, D)

    # Build a fully-connected graph over this component
    import networkx as nx
    G = nx.Graph()
    for oid in component_obj_ids:
        if oid in all_obj_id_to_row:
            G.add_node(oid, position=[0.0, 0.0, 0.0])
    for i, oid_i in enumerate(component_obj_ids):
        for oid_j in component_obj_ids[i + 1:]:
            if oid_i in all_obj_id_to_row and oid_j in all_obj_id_to_row:
                G.add_edge(oid_i, oid_j)

    ib = ClusterIB(config)
    ib.setup_py_x(comp_features, task_features)
    ib.initialize_nx_graph(G)
    return ib.find_clusters()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config",
            config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("COMPRESS TIERGRAPH — Agglomerative Information Bottleneck")
    print("=" * 60 + "\n")

    # ---- paths ----
    tiergraph_path   = Path(cfg.scenegraph_path)
    result_path      = Path(cfg.result_path)
    ib_config_path   = Path(cfg.get('ib_config', 'config/ib_config.yaml'))
    output_path      = tiergraph_path.parent / 'object_relations_compressed.json'

    # Resolve relative ib_config_path against script directory
    if not ib_config_path.is_absolute():
        ib_config_path = (
            Path(__file__).parent.parent / ib_config_path
        )

    print(f"TierGraph input:  {tiergraph_path}")
    print(f"Full PCD input:   {result_path}")
    print(f"IB config:        {ib_config_path}")
    print(f"Output:           {output_path}\n")

    # ---- load data ----
    print("Loading TierGraph...")
    tiergraph = load_tiergraph(tiergraph_path)
    n_objects = sum(1 for n in tiergraph['nodes'] if n['type'] == 'object')
    print(f"  {len(tiergraph['nodes'])} nodes total, {n_objects} object nodes")

    print("Loading OpenGraph objects (for CLIP/SBERT features)...")
    objects = load_opengraph_objects(result_path)
    print(f"  {len(objects)} objects loaded")

    # ---- load IB config ----
    ib_config = ClusterIBConfig(str(ib_config_path))
    print(f"\nIB config: delta={ib_config.delta}, "
          f"sims_thres={ib_config.sims_thres}, "
          f"top_k={ib_config.top_k}")

    # ---- load task strings ----
    import yaml
    with open(ib_config_path, 'r') as f:
        ib_cfg_raw = yaml.safe_load(f)
    task_strings = ib_cfg_raw.get('tasks', [
        'find a box', 'locate a package', 'identify items on shelf'])
    print(f"\nTask queries ({len(task_strings)}):")
    for t in task_strings:
        print(f"  • {t}")

    # ---- encode tasks ----
    print(f"\nEncoding tasks with SBERT ({cfg.sbert_path})...")
    task_features = encode_tasks_sbert(task_strings, cfg.sbert_path)
    print(f"  Task features: {task_features.shape}")

    # ---- extract object features ----
    print("\nExtracting object SBERT features from MapObjectList...")
    region_features, valid_obj_indices = extract_clip_features(objects)
    print(f"  Valid features: {len(valid_obj_indices)} / {len(objects)} objects "
          f"(dim={region_features.shape[1] if region_features.ndim == 2 else '?'})")

    if len(valid_obj_indices) == 0:
        print("ERROR: No valid features found. Cannot run IB compression.")
        return

    # Build mapping: obj_id → row in region_features
    obj_id_to_row = {f"object_{idx}": row
                     for row, idx in enumerate(valid_obj_indices)}

    # ---- build hierarchy-derived IB graph ----
    print("\nBuilding hierarchy-derived IB adjacency graph...")
    G, group_map = build_ib_graph(tiergraph, objects, valid_obj_indices)

    if G.number_of_nodes() == 0:
        print("WARNING: IB graph has no nodes. "
              "Check that TierGraph has object nodes with valid IDs.")
        return

    # ---- run IB per connected component ----
    print("\nRunning Agglomerative IB compression...")
    components = list(nx.connected_components(G))
    print(f"  {len(components)} connected components (merge groups)")

    all_clusters = []
    total_merged = 0

    for comp_idx, comp_nodes in enumerate(components):
        comp_list = list(comp_nodes)
        if len(comp_list) == 1:
            # Singleton: no compression possible
            all_clusters.append(comp_list)
            continue

        comp_clusters = run_ib_on_component(
            comp_list, obj_id_to_row,
            region_features, task_features,
            ib_config
        )
        all_clusters.extend(comp_clusters)

        n_merged = len(comp_list) - len(comp_clusters)
        if n_merged > 0:
            total_merged += n_merged
            print(f"  Component {comp_idx:3d}: {len(comp_list):3d} objects → "
                  f"{len(comp_clusters):3d} clusters  "
                  f"(merged {n_merged})")

    # ---- print compression summary ----
    # Use IB graph node count (objects actually in TierGraph), not all feature objects
    total_objs     = G.number_of_nodes()
    total_clusters = len(all_clusters)
    ratio          = total_objs / max(total_clusters, 1)

    print(f"\n{'=' * 50}")
    print(f"COMPRESSION SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Objects before: {total_objs}")
    print(f"  Clusters after: {total_clusters}")
    print(f"  Compression:    {ratio:.2f}x  "
          f"({total_objs - total_clusters} objects merged)")

    multi = [c for c in all_clusters if len(c) > 1]
    if multi:
        print(f"\n  Multi-object clusters ({len(multi)}):")
        for cl in sorted(multi, key=len, reverse=True)[:10]:
            print(f"    {len(cl)} members: {cl}")
        if len(multi) > 10:
            print(f"    ... ({len(multi) - 10} more)")

    # ---- apply clusters to TierGraph ----
    print("\nApplying cluster assignments to TierGraph...")
    compressed = apply_clusters_to_tiergraph(
        tiergraph   = tiergraph,
        clusters    = all_clusters,
        objects     = objects,
        valid_obj_indices = valid_obj_indices,
        task_names  = task_strings,
    )

    # ---- save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(compressed, f, indent=2)

    print(f"\nCompressed TierGraph saved to: {output_path}")
    print(f"  Nodes:   {compressed['statistics']['total_nodes']}")
    print(f"  Edges:   {compressed['statistics']['total_edges']}")
    print(f"  by type: {compressed['statistics']['nodes_by_type']}")
    print("\n✓ IB compression complete!")


import networkx as nx   # needed by run_ib_on_component

if __name__ == "__main__":
    main()
