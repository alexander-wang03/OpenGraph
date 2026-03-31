#!/usr/bin/env python3
"""
Sweep IB compression delta — Experiment 2.3

Varies the IB delta parameter (0.05 → 0.30), runs compression and retrieval
evaluation at each value, and collects results into a summary CSV + table.

Optionally runs two task sets:
  1. Generic tasks (from ib_config.yaml) — realistic baseline
  2. Eval-derived tasks (from warehouse_task_list.csv) — upper bound

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/sweep_compression.py --config-name=isaac_warehouse sequence=03

Author: awang (TierGraph thesis)
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import copy
import csv
import gzip
import json
import pickle
import time
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import yaml
from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from some_class.map_calss import MapObjectList
from utils.coordinate_alignment import load_absolute_first_pose
from utils.ib.ib_cluster import ClusterIB, ClusterIBConfig
from utils.ib.tiergraph_bridge import (
    extract_clip_features,
    build_ib_graph,
    apply_clusters_to_tiergraph,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DELTA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

GENERIC_TASKS = [
    "find a cardboard box",
    "locate a package or parcel",
    "identify products on a shelf",
    "find storage bins or containers",
    "locate items on a pallet",
]

# Zone keyword mapping (same as query_tiergraph.py)
ZONE_KEYWORDS = {
    "zone_staging": [
        "staging area", "staging zone", "packing area", "packing zone",
    ],
    "zone_receiving": [
        "receiving area", "receiving zone", "unloading area", "unloading zone",
        "dock",
    ],
    "zone_storage": [
        "shelf area", "shelf zone", "storage area", "storage zone",
        "aisle", "shelves", "rack",
    ],
    "zone_forklift": [
        "forklift area", "forklift zone", "yellow-marked area",
        "forklift parking",
    ],
    "zone_hub_robot": [
        "hub robot", "robot area", "robot zone",
    ],
}

# TAP-feasible exclusion patterns (same as query_tiergraph.py)
TAP_EXCLUDE_PATTERNS = [
    'labeled "Digital Twin"',
    "labeled 'Digital Twin'",
    '"EQP" symbol',
    '"UNK" symbol',
    "star-shaped symbol",
    "FRAGILE labeled",
    "first aid kit",
    "disinfectant",
    "flashlight",
    "fire extinguisher",
    "small bottles",
    "electrical box",
    "dock-high doors",
]


def is_tap_feasible(question: str) -> bool:
    q_lower = question.lower()
    for pattern in TAP_EXCLUDE_PATTERNS:
        if pattern.lower() in q_lower:
            return False
    return True


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


def parse_query_csv(csv_path: str, T_inv: np.ndarray,
                    z_offset: float = 0.781) -> list:
    queries = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            center_str = row.get("Center", "").strip()
            if not center_str:
                continue
            question = row.get("Question", "").strip()
            if not question:
                continue
            center_str = center_str.strip("[]")
            parts = [float(x.strip()) for x in center_str.split(",")]
            if len(parts) != 3:
                continue
            gt_global = np.array(parts)
            gt_homo = np.array([*gt_global, 1.0])
            gt_robot = (T_inv @ gt_homo)[:3]
            gt_robot[2] += z_offset
            qtype = row.get("Type (binary, position, time, text)", "").strip()
            queries.append({
                "question": question,
                "query_type": qtype,
                "gt_global": gt_global,
                "gt_robot": gt_robot,
                "tap_feasible": is_tap_feasible(question),
            })
    return queries


# ---------------------------------------------------------------------------
# Retrieval (copied core logic from query_tiergraph.py)
# ---------------------------------------------------------------------------

def get_object_centroid(node: dict) -> np.ndarray:
    c = node["bounds"]["center"]
    return np.array([c["x"], c["y"], c["z"]])


def _detect_zone(query_text: str) -> str:
    query_lower = query_text.lower()
    for zone_id, keywords in ZONE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return zone_id
    return None


def _find_ancestor_zone(node: dict, node_lookup: dict) -> str:
    current = node
    for _ in range(10):
        if current["type"] == "zone":
            return current["id"]
        parent_id = current.get("parent_id")
        if parent_id is None or parent_id not in node_lookup:
            return None
        current = node_lookup[parent_id]
    return None


def flat_retrieval(q_emb, obj_features, top_k=5):
    q = torch.from_numpy(q_emb).float().unsqueeze(0)
    o = torch.from_numpy(obj_features).float()
    sims = F.cosine_similarity(q, o, dim=1)
    scores, indices = torch.topk(sims, min(top_k, len(sims)))
    return list(zip(indices.tolist(), scores.tolist()))


def hierarchical_retrieval(q_emb, obj_features, obj_node_ids, node_lookup,
                            query_text="", top_k=5, zone_top_n=2,
                            container_top_n=5):
    q = torch.from_numpy(q_emb).float()

    # Zone level
    detected_zone = _detect_zone(query_text)
    zone_objects = {}
    for fi, nid in enumerate(obj_node_ids):
        node = node_lookup.get(nid)
        if node is None:
            continue
        zone_id = _find_ancestor_zone(node, node_lookup)
        if zone_id:
            zone_objects.setdefault(zone_id, []).append(fi)

    if detected_zone and detected_zone in zone_objects:
        candidate_idxs = zone_objects[detected_zone]
    else:
        zone_scores = []
        for zid, feat_idxs in zone_objects.items():
            zone_feat = torch.from_numpy(
                obj_features[feat_idxs]).float().mean(dim=0)
            score = F.cosine_similarity(
                q.unsqueeze(0), zone_feat.unsqueeze(0)).item()
            zone_scores.append((zid, score, feat_idxs))
        zone_scores.sort(key=lambda x: x[1], reverse=True)
        candidate_idxs = []
        for _, _, fi_list in zone_scores[:zone_top_n]:
            candidate_idxs.extend(fi_list)

    if not candidate_idxs:
        return flat_retrieval(q_emb, obj_features, top_k)

    # Container level
    container_objects = {}
    for fi in candidate_idxs:
        nid = obj_node_ids[fi]
        node = node_lookup.get(nid)
        if node is None:
            continue
        parent_id = node.get("parent_id", "unknown")
        container_objects.setdefault(parent_id, []).append(fi)

    container_scores = []
    for cid, feat_idxs in container_objects.items():
        cfeat = torch.from_numpy(
            obj_features[feat_idxs]).float().mean(dim=0)
        score = F.cosine_similarity(
            q.unsqueeze(0), cfeat.unsqueeze(0)).item()
        container_scores.append((cid, score, feat_idxs))
    container_scores.sort(key=lambda x: x[1], reverse=True)

    final_idxs = []
    for _, _, fi_list in container_scores[:container_top_n]:
        final_idxs.extend(fi_list)

    if not final_idxs:
        return flat_retrieval(q_emb, obj_features, top_k)

    # Object level
    final_feats = torch.from_numpy(obj_features[final_idxs]).float()
    sims = F.cosine_similarity(q.unsqueeze(0), final_feats, dim=1)
    k = min(top_k, len(sims))
    scores, local_indices = torch.topk(sims, k)

    return [(final_idxs[li], sc)
            for li, sc in zip(local_indices.tolist(), scores.tolist())]


def evaluate_retrieval(results, obj_node_ids, node_lookup, gt_robot,
                        threshold=5.0, ks=(1, 3, 5)):
    eval_result = {f"hit@{k}": 0 for k in ks}
    eval_result["min_dist"] = float("inf")

    for rank, (feat_idx, score) in enumerate(results):
        nid = obj_node_ids[feat_idx]
        node = node_lookup.get(nid)
        if node is None:
            continue
        centroid = get_object_centroid(node)
        dist = np.linalg.norm(centroid - gt_robot)
        if dist < eval_result["min_dist"]:
            eval_result["min_dist"] = dist
        if dist <= threshold:
            for k in ks:
                if rank < k:
                    eval_result[f"hit@{k}"] = 1

    return eval_result


# ---------------------------------------------------------------------------
# Build retrieval inputs from a (possibly compressed) TierGraph
# ---------------------------------------------------------------------------

def build_retrieval_inputs(graph, objects):
    """
    Build feature matrix and node IDs for retrieval.
    Works with both original (object nodes) and compressed (object + cluster
    nodes) TierGraphs.

    For cluster nodes, uses the mean_feature from metadata.
    For object nodes, uses the ft vector from full_pcd.

    Returns:
        obj_features: (N, D) numpy array
        obj_node_ids: list of node IDs (object_X or cluster_X)
        node_lookup:  {node_id: node_dict}
    """
    node_lookup = {n["id"]: n for n in graph["nodes"]}
    obj_features = []
    obj_node_ids = []

    for node in graph["nodes"]:
        if node["type"] == "object":
            parts = node["id"].split("_")
            if parts[0] != "object" or len(parts) != 2:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            if idx < len(objects) and "ft" in objects[idx]:
                ft = objects[idx]["ft"]
                if isinstance(ft, torch.Tensor):
                    ft = ft.numpy()
                if ft.ndim == 2:
                    ft = ft[0]
                obj_features.append(ft)
                obj_node_ids.append(node["id"])

        elif node["type"] == "cluster":
            meta = node.get("metadata", {})
            mean_feat = meta.get("mean_feature")
            if mean_feat is not None:
                ft = np.array(mean_feat)
                if ft.ndim == 2:
                    ft = ft[0]
                obj_features.append(ft)
                obj_node_ids.append(node["id"])

    if not obj_features:
        return np.zeros((0, 384)), [], node_lookup

    return np.stack(obj_features), obj_node_ids, node_lookup


# ---------------------------------------------------------------------------
# IB compression at a given delta
# ---------------------------------------------------------------------------

def compress_at_delta(tiergraph, objects, valid_obj_indices, region_features,
                       task_features, task_strings, delta, sims_thres=0.20):
    """Run IB compression with a specific delta value. Returns compressed graph."""
    # Create a temporary config with the given delta
    import tempfile, os
    tmp_cfg = {
        'delta': delta,
        'sims_thres': sims_thres,
        'top_k_tasks': 1,
        'cumulative': False,
        'use_lerf_loss': False,
        'lerf_loss_cannonical_phrases': [],
        'tasks': task_strings,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False) as f:
        yaml.dump(tmp_cfg, f)
        tmp_path = f.name

    try:
        config = ClusterIBConfig(tmp_path)
        obj_id_to_row = {f"object_{idx}": row
                         for row, idx in enumerate(valid_obj_indices)}

        G, group_map = build_ib_graph(tiergraph, objects, valid_obj_indices)
        if G.number_of_nodes() == 0:
            return None, 0, 0

        components = list(nx.connected_components(G))
        all_clusters = []
        for comp_nodes in components:
            comp_list = list(comp_nodes)
            if len(comp_list) == 1:
                all_clusters.append(comp_list)
                continue
            rows = [obj_id_to_row[oid] for oid in comp_list
                    if oid in obj_id_to_row]
            if len(rows) < 2:
                for oid in comp_list:
                    if oid in obj_id_to_row:
                        all_clusters.append([oid])
                continue

            comp_features = region_features[rows]
            G_comp = nx.Graph()
            for oid in comp_list:
                if oid in obj_id_to_row:
                    G_comp.add_node(oid, position=[0.0, 0.0, 0.0])
            for i, oid_i in enumerate(comp_list):
                for oid_j in comp_list[i + 1:]:
                    if oid_i in obj_id_to_row and oid_j in obj_id_to_row:
                        G_comp.add_edge(oid_i, oid_j)

            ib = ClusterIB(config)
            ib.setup_py_x(comp_features, task_features)
            ib.initialize_nx_graph(G_comp)
            all_clusters.extend(ib.find_clusters())

        n_objects = G.number_of_nodes()
        n_clusters = len(all_clusters)

        compressed = apply_clusters_to_tiergraph(
            tiergraph=copy.deepcopy(tiergraph),
            clusters=all_clusters,
            objects=objects,
            valid_obj_indices=valid_obj_indices,
            task_names=task_strings,
        )
        return compressed, n_objects, n_clusters
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Run evaluation on a graph
# ---------------------------------------------------------------------------

def run_eval(graph, objects, queries, sbert):
    """Run flat + hierarchical retrieval evaluation on a graph.
    Returns dict with hit counts."""
    obj_features, obj_node_ids, node_lookup = build_retrieval_inputs(
        graph, objects)

    if len(obj_node_ids) == 0:
        return None

    ks = (1, 3, 5)
    flat_hits = {k: 0 for k in ks}
    hier_hits = {k: 0 for k in ks}
    n = len(queries)

    # Also track for filtered subsets
    flat_hits_feasible = {k: 0 for k in ks}
    hier_hits_feasible = {k: 0 for k in ks}
    n_feasible = sum(1 for q in queries if q["tap_feasible"])

    # Oracle reachability
    obj_nodes_for_oracle = [node_lookup[nid] for nid in obj_node_ids
                             if nid in node_lookup]
    reachable = 0
    reachable_feasible = 0
    flat_hits_reach = {k: 0 for k in ks}
    hier_hits_reach = {k: 0 for k in ks}
    flat_hits_feas_reach = {k: 0 for k in ks}
    hier_hits_feas_reach = {k: 0 for k in ks}

    query_reachable = []
    for q in queries:
        min_d = min((np.linalg.norm(get_object_centroid(n) - q["gt_robot"])
                     for n in obj_nodes_for_oracle), default=float("inf"))
        is_reach = min_d <= 5.0
        query_reachable.append(is_reach)
        if is_reach:
            reachable += 1
            if q["tap_feasible"]:
                reachable_feasible += 1

    for qi, q in enumerate(queries):
        q_emb = sbert.encode(q["question"], normalize_embeddings=True)

        # Flat
        flat_res = flat_retrieval(q_emb, obj_features, top_k=5)
        flat_ev = evaluate_retrieval(flat_res, obj_node_ids, node_lookup,
                                      q["gt_robot"])

        # Hierarchical
        hier_res = hierarchical_retrieval(q_emb, obj_features, obj_node_ids,
                                           node_lookup,
                                           query_text=q["question"],
                                           top_k=5)
        hier_ev = evaluate_retrieval(hier_res, obj_node_ids, node_lookup,
                                      q["gt_robot"])

        for k in ks:
            flat_hits[k] += flat_ev[f"hit@{k}"]
            hier_hits[k] += hier_ev[f"hit@{k}"]

            if q["tap_feasible"]:
                flat_hits_feasible[k] += flat_ev[f"hit@{k}"]
                hier_hits_feasible[k] += hier_ev[f"hit@{k}"]

            if query_reachable[qi]:
                flat_hits_reach[k] += flat_ev[f"hit@{k}"]
                hier_hits_reach[k] += hier_ev[f"hit@{k}"]

                if q["tap_feasible"]:
                    flat_hits_feas_reach[k] += flat_ev[f"hit@{k}"]
                    hier_hits_feas_reach[k] += hier_ev[f"hit@{k}"]

    return {
        "n_queries": n,
        "n_feasible": n_feasible,
        "n_reachable": reachable,
        "n_feasible_reachable": reachable_feasible,
        "n_nodes": len(obj_node_ids),
        "flat_hits": flat_hits,
        "hier_hits": hier_hits,
        "flat_hits_feasible": flat_hits_feasible,
        "hier_hits_feasible": hier_hits_feasible,
        "flat_hits_reach": flat_hits_reach,
        "hier_hits_reach": hier_hits_reach,
        "flat_hits_feas_reach": flat_hits_feas_reach,
        "hier_hits_feas_reach": hier_hits_feas_reach,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config",
            config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 70)
    print("SWEEP: IB Compression Delta vs Retrieval Precision")
    print("=" * 70 + "\n")

    # Paths
    tiergraph_path = Path(cfg.scenegraph_path)
    result_path = Path(cfg.result_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    query_csv = str(
        Path(__file__).parent.parent / "data" / "warehouse_layout"
        / "warehouse_task_list.csv"
    )
    output_dir = tiergraph_path.parent
    output_csv = output_dir / "sweep_compression_results.csv"

    # Load SBERT
    print("Loading SBERT model...")
    sbert = SentenceTransformer(cfg.sbert_path)

    # Load TierGraph
    print(f"Loading TierGraph from {tiergraph_path}...")
    tiergraph = load_tiergraph(tiergraph_path)
    n_objects = sum(1 for n in tiergraph['nodes'] if n['type'] == 'object')
    print(f"  {n_objects} object nodes")

    # Load objects
    print(f"Loading objects from {result_path}...")
    objects = load_objects(result_path)
    print(f"  {len(objects)} objects in full_pcd")

    # Extract features
    print("Extracting SBERT features...")
    region_features, valid_obj_indices = extract_clip_features(objects)
    print(f"  {len(valid_obj_indices)} valid features (dim={region_features.shape[1]})")

    # Encode task sets
    print("Encoding task sets...")
    generic_task_features = sbert.encode(
        GENERIC_TASKS, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float64)

    # Build eval-derived tasks from query CSV position queries
    T_first = load_absolute_first_pose(sequence_dir)
    T_inv = np.linalg.inv(T_first)
    all_queries = parse_query_csv(query_csv, T_inv)
    position_queries = [q for q in all_queries if q["query_type"] == "position"]
    print(f"  {len(position_queries)} position queries for evaluation")

    eval_task_strings = list(set(q["question"] for q in position_queries))
    eval_task_features = sbert.encode(
        eval_task_strings, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float64)
    print(f"  Generic tasks: {len(GENERIC_TASKS)}")
    print(f"  Eval-derived tasks: {len(eval_task_strings)}")

    # Run baseline (uncompressed TierGraph)
    print("\n" + "-" * 70)
    print("BASELINE: Uncompressed TierGraph")
    print("-" * 70)
    baseline = run_eval(tiergraph, objects, position_queries, sbert)
    n_base = baseline["n_queries"]
    print(f"  {baseline['n_nodes']} nodes, "
          f"{baseline['n_reachable']}/{n_base} reachable")
    for k in (1, 3, 5):
        fp = 100 * baseline["flat_hits"][k] / n_base
        hp = 100 * baseline["hier_hits"][k] / n_base
        print(f"  P@{k}: flat={fp:.1f}%  hier={hp:.1f}%")

    # Sweep
    results_rows = []
    task_sets = {
        "generic": (GENERIC_TASKS, generic_task_features),
        "eval": (eval_task_strings, eval_task_features),
    }

    for task_label, (task_strings, task_features) in task_sets.items():
        print(f"\n{'=' * 70}")
        print(f"TASK SET: {task_label} ({len(task_strings)} tasks)")
        print(f"{'=' * 70}")

        for delta in DELTA_VALUES:
            print(f"\n--- delta={delta:.2f} ({task_label}) ---")

            t0 = time.time()
            compressed, n_obj_before, n_clusters = compress_at_delta(
                tiergraph, objects, valid_obj_indices,
                region_features, task_features, task_strings,
                delta=delta,
            )
            t_compress = time.time() - t0

            if compressed is None:
                print("  SKIP: compression failed")
                continue

            by_type = compressed["statistics"]["nodes_by_type"]
            n_obj_after = by_type.get("object", 0) + by_type.get("cluster", 0)
            ratio = n_obj_before / max(n_obj_after, 1)
            print(f"  {n_obj_before} objects → {n_obj_after} units "
                  f"({ratio:.2f}x compression, {t_compress:.1f}s)")

            # Evaluate
            ev = run_eval(compressed, objects, position_queries, sbert)
            if ev is None:
                print("  SKIP: no retrievable nodes")
                continue

            n = ev["n_queries"]
            nr = ev["n_reachable"]
            nf = ev["n_feasible"]
            nfr = ev["n_feasible_reachable"]

            row = {
                "task_set": task_label,
                "delta": delta,
                "objects_before": n_obj_before,
                "units_after": n_obj_after,
                "compression_ratio": round(ratio, 2),
                "n_clusters": by_type.get("cluster", 0),
                "n_singletons": by_type.get("object", 0),
            }

            # All queries
            for method, hits in [("flat", ev["flat_hits"]),
                                  ("hier", ev["hier_hits"])]:
                for k in (1, 3, 5):
                    row[f"{method}_P@{k}_all"] = round(
                        100 * hits[k] / n, 1) if n > 0 else 0

            # Reachable
            for method, hits in [("flat", ev["flat_hits_reach"]),
                                  ("hier", ev["hier_hits_reach"])]:
                for k in (1, 3, 5):
                    row[f"{method}_P@{k}_reach"] = round(
                        100 * hits[k] / nr, 1) if nr > 0 else 0

            # TAP-feasible + reachable
            for method, hits in [("flat", ev["flat_hits_feas_reach"]),
                                  ("hier", ev["hier_hits_feas_reach"])]:
                for k in (1, 3, 5):
                    row[f"{method}_P@{k}_feas_reach"] = round(
                        100 * hits[k] / nfr, 1) if nfr > 0 else 0

            results_rows.append(row)

            # Print summary line
            fp1 = row["flat_P@1_all"]
            hp1 = row["hier_P@1_all"]
            fp1r = row["flat_P@1_reach"]
            hp1r = row["hier_P@1_reach"]
            print(f"  P@1 all:  flat={fp1}%  hier={hp1}%")
            print(f"  P@1 reach: flat={fp1r}%  hier={hp1r}%")

    # Add baseline row
    base_row = {
        "task_set": "none",
        "delta": 0.0,
        "objects_before": baseline["n_nodes"],
        "units_after": baseline["n_nodes"],
        "compression_ratio": 1.0,
        "n_clusters": 0,
        "n_singletons": baseline["n_nodes"],
    }
    for method, hits in [("flat", baseline["flat_hits"]),
                          ("hier", baseline["hier_hits"])]:
        for k in (1, 3, 5):
            base_row[f"{method}_P@{k}_all"] = round(
                100 * hits[k] / n_base, 1) if n_base > 0 else 0
    nr = baseline["n_reachable"]
    nfr = baseline["n_feasible_reachable"]
    for method, hits in [("flat", baseline["flat_hits_reach"]),
                          ("hier", baseline["hier_hits_reach"])]:
        for k in (1, 3, 5):
            base_row[f"{method}_P@{k}_reach"] = round(
                100 * hits[k] / nr, 1) if nr > 0 else 0
    for method, hits in [("flat", baseline["flat_hits_feas_reach"]),
                          ("hier", baseline["hier_hits_feas_reach"])]:
        for k in (1, 3, 5):
            base_row[f"{method}_P@{k}_feas_reach"] = round(
                100 * hits[k] / nfr, 1) if nfr > 0 else 0
    results_rows.insert(0, base_row)

    # Print final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    header = (f"{'Tasks':<8} {'Delta':>5} {'Units':>5} {'Ratio':>5} "
              f"{'F-P@1':>6} {'H-P@1':>6} {'F-P@1r':>7} {'H-P@1r':>7} "
              f"{'F-P@5':>6} {'H-P@5':>6}")
    print(header)
    print("-" * len(header))
    for row in results_rows:
        print(f"{row['task_set']:<8} {row['delta']:>5.2f} "
              f"{row['units_after']:>5} {row['compression_ratio']:>5.2f} "
              f"{row['flat_P@1_all']:>5.1f}% {row['hier_P@1_all']:>5.1f}% "
              f"{row['flat_P@1_reach']:>6.1f}% {row['hier_P@1_reach']:>6.1f}% "
              f"{row['flat_P@5_all']:>5.1f}% {row['hier_P@5_all']:>5.1f}%")
    print("=" * 70)

    # Save CSV
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nResults saved to: {output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
