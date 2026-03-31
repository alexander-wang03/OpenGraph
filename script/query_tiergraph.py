#!/usr/bin/env python3
"""
Query TierGraph — Flat vs Hierarchical Retrieval Evaluation

Evaluates object retrieval on TierGraph using two strategies:
  1. Flat retrieval:  SBERT cosine similarity against ALL object features
  2. Hierarchical:    Cascading zone → shelf/section → object search
     - Zone-keyword matching: queries mentioning a zone name restrict
       search to that zone (e.g. "staging area" → zone_packing)

Ground truth comes from the warehouse task list CSV (Isaac Sim global coords).
Success = retrieved object centroid within 5m of GT position.

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/query_tiergraph.py --config-name=isaac_warehouse sequence=02

Author: awang (TierGraph thesis)
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import csv
import gzip
import json
import pickle
import re
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from some_class.map_calss import MapObjectList
from utils.coordinate_alignment import load_absolute_first_pose


# Zone keyword mapping: query text patterns → zone IDs
# When a query mentions one of these, restrict hierarchical search to that zone
# ---------------------------------------------------------------------------
# TAP-feasible query filter
# ---------------------------------------------------------------------------
# Queries requiring fine-grained label reading, text recognition, or
# identification of small wall-mounted items are beyond TAP's visual
# segmentation capability.  We exclude them for the "filtered" evaluation.

TAP_EXCLUDE_PATTERNS = [
    # Text/label reading required
    'labeled "Digital Twin"',
    "labeled 'Digital Twin'",
    '"EQP" symbol',
    '"UNK" symbol',
    "star-shaped symbol",
    "FRAGILE labeled",
    # Small wall-mounted items (not reliably segmented by TAP)
    "first aid kit",
    "disinfectant",
    "flashlight",
    "fire extinguisher",
    # Small indistinguishable items
    "small bottles",
    # Structural features (not objects)
    "electrical box",
    "dock-high doors",
]


def is_tap_feasible(question: str) -> bool:
    """Return True if the query targets visually distinct objects that
    TAP can reasonably segment and caption (no label reading, no tiny
    wall-mounted items)."""
    q_lower = question.lower()
    for pattern in TAP_EXCLUDE_PATTERNS:
        if pattern.lower() in q_lower:
            return False
    return True


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_objects(result_path: Path) -> MapObjectList:
    """Load full_pcd.pkl.gz and return MapObjectList."""
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    objects = MapObjectList()
    if isinstance(results, dict):
        objects.load_serializable(results["objects"])
    elif isinstance(results, list):
        objects.load_serializable(results)
    return objects


def load_tiergraph(json_path: Path) -> dict:
    """Load TierGraph JSON and build lookup structures."""
    with open(json_path, "r") as f:
        graph = json.load(f)

    node_lookup = {n["id"]: n for n in graph["nodes"]}
    return graph, node_lookup


def get_object_centroid(node: dict) -> np.ndarray:
    """Extract centroid from a TierGraph node's bounds."""
    c = node["bounds"]["center"]
    return np.array([c["x"], c["y"], c["z"]])


# ---------------------------------------------------------------------------
# Query set parsing
# ---------------------------------------------------------------------------

def parse_query_csv(csv_path: str, T_inv: np.ndarray, z_offset: float = 0.781) -> list:
    """
    Parse the warehouse task list CSV into a query set.

    Filters to rows that have a ground-truth Center coordinate.
    Transforms GT from Isaac Sim global frame → robot-relative frame,
    then applies the same +0.781m Z offset used for object centroids.

    Returns list of dicts: {uuid, question, gt_global, gt_robot, query_type}
    """
    queries = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            center_str = row.get("Center", "").strip()
            if not center_str:
                continue  # skip queries without GT position

            question = row.get("Question", "").strip()
            if not question:
                continue

            # Parse "[x, y, z]"
            center_str = center_str.strip("[]")
            parts = [float(x.strip()) for x in center_str.split(",")]
            if len(parts) != 3:
                continue
            gt_global = np.array(parts)

            # Transform global → robot-relative
            gt_homo = np.array([*gt_global, 1.0])
            gt_robot = (T_inv @ gt_homo)[:3]
            # Apply same Z offset as extract_object_centroid
            gt_robot[2] += z_offset

            queries.append({
                "uuid": row.get("UUID", ""),
                "question": question,
                "query_type": row.get("Type (binary, position, time, text)", "").strip(),
                "gt_global": gt_global,
                "gt_robot": gt_robot,
            })

    return queries


# ---------------------------------------------------------------------------
# Retrieval: Flat
# ---------------------------------------------------------------------------

def flat_retrieval(query_embedding: np.ndarray,
                   object_features: np.ndarray,
                   top_k: int = 5) -> list:
    """
    Flat SBERT cosine similarity retrieval against all objects.

    Args:
        query_embedding: (384,) SBERT embedding of the query
        object_features: (N, 384) stacked object ft vectors
        top_k: number of results to return

    Returns:
        List of (object_index_in_features, score) sorted by descending score
    """
    q = torch.from_numpy(query_embedding).float().unsqueeze(0)  # (1, 384)
    o = torch.from_numpy(object_features).float()               # (N, 384)
    sims = F.cosine_similarity(q, o, dim=1)                     # (N,)
    scores, indices = torch.topk(sims, min(top_k, len(sims)))
    return list(zip(indices.tolist(), scores.tolist()))


# ---------------------------------------------------------------------------
# Retrieval: Hierarchical (cascading)
# ---------------------------------------------------------------------------

def _detect_zone_from_query(query_text: str) -> str:
    """
    Check if the query text explicitly mentions a warehouse zone.

    Returns zone_id if found, None otherwise.
    """
    query_lower = query_text.lower()
    for zone_id, keywords in ZONE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return zone_id
    return None


def hierarchical_retrieval(query_embedding: np.ndarray,
                           object_features: np.ndarray,
                           obj_indices: list,
                           node_lookup: dict,
                           query_text: str = "",
                           top_k: int = 5,
                           zone_top_n: int = 2,
                           container_top_n: int = 5) -> list:
    """
    Cascading hierarchical retrieval: zone → container → object.

    1. If query mentions a specific zone, restrict to that zone.
       Otherwise, mean-pool object features per zone → pick top-N zones.
    2. Mean-pool object features per parent container (shelf/section/aisle)
       within shortlisted zones → pick top-N containers
    3. Rank individual objects within shortlisted containers

    Args:
        query_embedding: (384,) SBERT embedding
        object_features: (N, 384) stacked ft vectors, indexed same as obj_indices
        obj_indices: list of int indices (into full_pcd) for each row in object_features
        node_lookup: TierGraph node dict keyed by node ID
        query_text: original query string (for zone keyword matching)
        top_k: final number of objects to return
        zone_top_n: how many zones to keep at level 1
        container_top_n: how many containers to keep at level 2

    Returns:
        List of (object_index_in_features, score)
    """
    q = torch.from_numpy(query_embedding).float()  # (384,)

    # Build index: zone_id → list of (feat_idx, obj_node_id)
    zone_objects = {}   # zone_id → [(feat_idx, obj_node_id), ...]
    for feat_idx, obj_idx in enumerate(obj_indices):
        obj_id = f"object_{obj_idx}"
        if obj_id not in node_lookup:
            continue
        node = node_lookup[obj_id]
        # Walk up to find the zone
        zone_id = _find_ancestor_zone(node, node_lookup)
        if zone_id is None:
            continue
        zone_objects.setdefault(zone_id, []).append((feat_idx, obj_id))

    if not zone_objects:
        return flat_retrieval(query_embedding, object_features, top_k)

    # Level 1: zone selection
    # If the query explicitly mentions a zone, force that zone
    detected_zone = _detect_zone_from_query(query_text)
    if detected_zone and detected_zone in zone_objects:
        shortlisted_zones = [detected_zone]
    else:
        # Score each zone by mean-pooled object features
        zone_scores = []
        for zone_id, members in zone_objects.items():
            feat_idxs = [m[0] for m in members]
            zone_feat = torch.from_numpy(object_features[feat_idxs]).float().mean(dim=0)
            score = F.cosine_similarity(q.unsqueeze(0), zone_feat.unsqueeze(0)).item()
            zone_scores.append((zone_id, score))
        zone_scores.sort(key=lambda x: x[1], reverse=True)
        shortlisted_zones = [z[0] for z in zone_scores[:zone_top_n]]

    # Collect objects from shortlisted zones
    candidate_feat_idxs = []
    for zone_id in shortlisted_zones:
        candidate_feat_idxs.extend([m[0] for m in zone_objects[zone_id]])

    if not candidate_feat_idxs:
        return flat_retrieval(query_embedding, object_features, top_k)

    # Level 2: score by parent container (shelf/section/aisle)
    container_objects = {}  # parent_id → [feat_idx, ...]
    for feat_idx in candidate_feat_idxs:
        obj_id = f"object_{obj_indices[feat_idx]}"
        node = node_lookup.get(obj_id)
        if node is None:
            continue
        parent_id = node.get("parent_id", "unknown")
        container_objects.setdefault(parent_id, []).append(feat_idx)

    container_scores = []
    for container_id, feat_idxs in container_objects.items():
        container_feat = torch.from_numpy(object_features[feat_idxs]).float().mean(dim=0)
        score = F.cosine_similarity(q.unsqueeze(0), container_feat.unsqueeze(0)).item()
        container_scores.append((container_id, score, feat_idxs))
    container_scores.sort(key=lambda x: x[1], reverse=True)

    # Keep top-N containers
    final_feat_idxs = []
    for _, _, feat_idxs in container_scores[:container_top_n]:
        final_feat_idxs.extend(feat_idxs)

    if not final_feat_idxs:
        return flat_retrieval(query_embedding, object_features, top_k)

    # Level 3: rank individual objects
    final_feats = torch.from_numpy(object_features[final_feat_idxs]).float()
    sims = F.cosine_similarity(q.unsqueeze(0), final_feats, dim=1)
    k = min(top_k, len(sims))
    scores, local_indices = torch.topk(sims, k)

    results = []
    for li, sc in zip(local_indices.tolist(), scores.tolist()):
        results.append((final_feat_idxs[li], sc))
    return results


def _find_ancestor_zone(node: dict, node_lookup: dict, max_depth: int = 10) -> str:
    """Walk up the hierarchy to find the zone ancestor."""
    current = node
    for _ in range(max_depth):
        if current["type"] == "zone":
            return current["id"]
        parent_id = current.get("parent_id")
        if parent_id is None or parent_id not in node_lookup:
            return None
        current = node_lookup[parent_id]
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(results: list,
                       obj_indices: list,
                       node_lookup: dict,
                       gt_robot: np.ndarray,
                       threshold: float = 5.0,
                       ks: tuple = (1, 3, 5)) -> dict:
    """
    Compute precision@k for a single query.

    A hit at rank i means the retrieved object's centroid is within
    `threshold` metres of the ground truth position.

    Returns dict with keys: hit@1, hit@3, hit@5, min_dist, best_obj_id
    """
    eval_result = {f"hit@{k}": 0 for k in ks}
    eval_result["min_dist"] = float("inf")
    eval_result["best_obj_id"] = None
    eval_result["distances"] = []

    for rank, (feat_idx, score) in enumerate(results):
        obj_idx = obj_indices[feat_idx]
        obj_id = f"object_{obj_idx}"
        node = node_lookup.get(obj_id)
        if node is None:
            eval_result["distances"].append(float("inf"))
            continue

        centroid = get_object_centroid(node)
        dist = np.linalg.norm(centroid - gt_robot)
        eval_result["distances"].append(dist)

        if dist < eval_result["min_dist"]:
            eval_result["min_dist"] = dist
            eval_result["best_obj_id"] = obj_id

        if dist <= threshold:
            for k in ks:
                if rank < k:
                    eval_result[f"hit@{k}"] = 1

    return eval_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("QUERY TIERGRAPH — Flat vs Hierarchical Retrieval")
    print("=" * 60 + "\n")

    # --- Paths ---
    result_path = Path(cfg.result_path)
    tiergraph_path = Path(cfg.scenegraph_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    query_csv = str(
        Path(__file__).parent.parent / "data" / "warehouse_layout" / "warehouse_task_list.csv"
    )
    output_dir = tiergraph_path.parent
    output_csv = output_dir / "retrieval_results.csv"

    # --- Load SBERT model ---
    print("Loading SBERT model...")
    sbert = SentenceTransformer(cfg.sbert_path)

    # --- Load TierGraph ---
    print(f"Loading TierGraph from {tiergraph_path}...")
    graph, node_lookup = load_tiergraph(tiergraph_path)
    object_nodes = [n for n in graph["nodes"] if n["type"] == "object"]
    print(f"  {len(object_nodes)} object nodes in TierGraph")

    # --- Load full_pcd for ft vectors ---
    print(f"Loading object features from {result_path}...")
    objects = load_objects(result_path)
    print(f"  {len(objects)} objects in full_pcd")

    # Build feature matrix for objects that exist in TierGraph
    obj_indices = []  # index into full_pcd
    obj_features = []
    for node in object_nodes:
        obj_id = node["id"]
        # Extract numeric index: "object_17" → 17, skip "bg_object_X"
        parts = obj_id.split("_")
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
                ft = ft[0]  # (1, 384) → (384,)
            obj_indices.append(idx)
            obj_features.append(ft)

    obj_features = np.stack(obj_features)  # (N, 384)
    print(f"  {len(obj_indices)} objects with ft vectors")

    # --- Load first pose for coordinate transform ---
    T_first = load_absolute_first_pose(sequence_dir)
    T_inv = np.linalg.inv(T_first)

    # --- Parse queries ---
    print(f"\nParsing queries from {query_csv}...")
    queries = parse_query_csv(query_csv, T_inv)
    print(f"  {len(queries)} queries with GT positions")

    # Filter to position-type queries (most reliable GT)
    position_queries = [q for q in queries if q["query_type"] == "position"]
    all_gt_queries = queries  # all queries with GT positions
    print(f"  {len(position_queries)} position-type queries")
    print(f"  {len(all_gt_queries)} total queries with GT positions (all types)")

    # Tag each query as TAP-feasible or not
    for q in position_queries:
        q["tap_feasible"] = is_tap_feasible(q["question"])
    n_feasible = sum(1 for q in position_queries if q["tap_feasible"])
    n_excluded = len(position_queries) - n_feasible
    print(f"  {n_feasible} TAP-feasible queries, {n_excluded} excluded "
          f"(label reading / fine-grained / structural)")

    # --- Oracle analysis: which queries even have a nearby object? ---
    print("\n--- Oracle Analysis ---")
    reachable = 0
    for q in position_queries:
        min_d = float("inf")
        for node in object_nodes:
            c = get_object_centroid(node)
            d = np.linalg.norm(c - q["gt_robot"])
            min_d = min(min_d, d)
        if min_d <= 5.0:
            reachable += 1
    print(f"  {reachable}/{len(position_queries)} position queries have a TierGraph "
          f"object within 5m of GT (oracle upper bound)")

    # --- Run retrieval ---
    ks = (1, 3, 5)
    flat_hits = {k: 0 for k in ks}
    hier_hits = {k: 0 for k in ks}
    results_rows = []

    print(f"\nRunning retrieval on {len(position_queries)} queries...\n")
    print(f"{'#':>3} {'Method':>6} {'Hit@1':>5} {'Hit@3':>5} {'Hit@5':>5} "
          f"{'MinDist':>8} {'BestObj':>12}  Question")
    print("-" * 110)

    for qi, query in enumerate(position_queries):
        # Encode query
        q_emb = sbert.encode(query["question"], normalize_embeddings=True)

        # Flat retrieval
        flat_results = flat_retrieval(q_emb, obj_features, top_k=5)
        flat_eval = evaluate_retrieval(flat_results, obj_indices, node_lookup,
                                       query["gt_robot"], threshold=5.0, ks=ks)

        # Hierarchical retrieval
        hier_results = hierarchical_retrieval(q_emb, obj_features, obj_indices,
                                              node_lookup,
                                              query_text=query["question"],
                                              top_k=5)
        hier_eval = evaluate_retrieval(hier_results, obj_indices, node_lookup,
                                       query["gt_robot"], threshold=5.0, ks=ks)

        # Accumulate
        for k in ks:
            flat_hits[k] += flat_eval[f"hit@{k}"]
            hier_hits[k] += hier_eval[f"hit@{k}"]

        # Print per-query results
        q_short = query["question"][:50]
        print(f"{qi+1:3d}   flat {flat_eval['hit@1']:5d} {flat_eval['hit@3']:5d} "
              f"{flat_eval['hit@5']:5d} {flat_eval['min_dist']:8.2f} "
              f"{flat_eval['best_obj_id'] or 'N/A':>12s}  {q_short}")
        print(f"      hier {hier_eval['hit@1']:5d} {hier_eval['hit@3']:5d} "
              f"{hier_eval['hit@5']:5d} {hier_eval['min_dist']:8.2f} "
              f"{hier_eval['best_obj_id'] or 'N/A':>12s}")

        # Build CSV row
        results_rows.append({
            "query_idx": qi,
            "question": query["question"],
            "query_type": query["query_type"],
            "tap_feasible": int(query.get("tap_feasible", True)),
            "gt_global_x": query["gt_global"][0],
            "gt_global_y": query["gt_global"][1],
            "gt_global_z": query["gt_global"][2],
            "gt_robot_x": query["gt_robot"][0],
            "gt_robot_y": query["gt_robot"][1],
            "gt_robot_z": query["gt_robot"][2],
            "flat_hit@1": flat_eval["hit@1"],
            "flat_hit@3": flat_eval["hit@3"],
            "flat_hit@5": flat_eval["hit@5"],
            "flat_min_dist": flat_eval["min_dist"],
            "flat_best_obj": flat_eval["best_obj_id"],
            "hier_hit@1": hier_eval["hit@1"],
            "hier_hit@3": hier_eval["hit@3"],
            "hier_hit@5": hier_eval["hit@5"],
            "hier_min_dist": hier_eval["min_dist"],
            "hier_best_obj": hier_eval["best_obj_id"],
        })

    # --- Summary ---
    n = len(position_queries)
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Queries evaluated: {n}")
    print(f"Oracle reachable:  {reachable}/{n} queries have GT object within 5m")
    print(f"Success threshold: 5.0m")
    print(f"TierGraph objects: {len(obj_indices)}")
    print()

    # All queries
    print("--- All position queries ---")
    print(f"{'Metric':<15} {'Flat':>8} {'Hierarchical':>14} {'Delta':>8}")
    print("-" * 50)
    for k in ks:
        flat_pct = 100 * flat_hits[k] / n if n > 0 else 0
        hier_pct = 100 * hier_hits[k] / n if n > 0 else 0
        delta = hier_pct - flat_pct
        print(f"Precision@{k:<5} {flat_pct:7.1f}% {hier_pct:13.1f}% {delta:+7.1f}%")

    # Reachable-only queries (oracle-filtered)
    if reachable > 0:
        flat_reach = {k: 0 for k in ks}
        hier_reach = {k: 0 for k in ks}
        for row in results_rows:
            # Check if this query is reachable
            gt = np.array([row["gt_robot_x"], row["gt_robot_y"], row["gt_robot_z"]])
            min_d = min(np.linalg.norm(get_object_centroid(n) - gt)
                        for n in object_nodes)
            if min_d <= 5.0:
                for k in ks:
                    flat_reach[k] += row[f"flat_hit@{k}"]
                    hier_reach[k] += row[f"hier_hit@{k}"]

        print()
        print(f"--- Reachable queries only ({reachable} queries with GT object in range) ---")
        print(f"{'Metric':<15} {'Flat':>8} {'Hierarchical':>14} {'Delta':>8}")
        print("-" * 50)
        for k in ks:
            flat_pct = 100 * flat_reach[k] / reachable
            hier_pct = 100 * hier_reach[k] / reachable
            delta = hier_pct - flat_pct
            print(f"Precision@{k:<5} {flat_pct:7.1f}% {hier_pct:13.1f}% {delta:+7.1f}%")

    # TAP-feasible filtered queries
    filtered_rows = [r for r in results_rows if r.get("tap_feasible", 1)]
    n_filt = len(filtered_rows)
    if n_filt > 0:
        # Count filtered reachable
        filt_reachable = 0
        flat_filt = {k: 0 for k in ks}
        hier_filt = {k: 0 for k in ks}
        flat_filt_reach = {k: 0 for k in ks}
        hier_filt_reach = {k: 0 for k in ks}
        for row in filtered_rows:
            for k in ks:
                flat_filt[k] += row[f"flat_hit@{k}"]
                hier_filt[k] += row[f"hier_hit@{k}"]
            gt = np.array([row["gt_robot_x"], row["gt_robot_y"], row["gt_robot_z"]])
            min_d = min(np.linalg.norm(get_object_centroid(n) - gt)
                        for n in object_nodes)
            if min_d <= 5.0:
                filt_reachable += 1
                for k in ks:
                    flat_filt_reach[k] += row[f"flat_hit@{k}"]
                    hier_filt_reach[k] += row[f"hier_hit@{k}"]

        print()
        print(f"--- TAP-feasible queries only ({n_filt} queries, "
              f"{n_excluded} excluded) ---")
        print(f"{'Metric':<15} {'Flat':>8} {'Hierarchical':>14} {'Delta':>8}")
        print("-" * 50)
        for k in ks:
            flat_pct = 100 * flat_filt[k] / n_filt
            hier_pct = 100 * hier_filt[k] / n_filt
            delta = hier_pct - flat_pct
            print(f"Precision@{k:<5} {flat_pct:7.1f}% {hier_pct:13.1f}% {delta:+7.1f}%")

        if filt_reachable > 0:
            print()
            print(f"--- TAP-feasible + reachable ({filt_reachable} queries) ---")
            print(f"{'Metric':<15} {'Flat':>8} {'Hierarchical':>14} {'Delta':>8}")
            print("-" * 50)
            for k in ks:
                flat_pct = 100 * flat_filt_reach[k] / filt_reachable
                hier_pct = 100 * hier_filt_reach[k] / filt_reachable
                delta = hier_pct - flat_pct
                print(f"Precision@{k:<5} {flat_pct:7.1f}% {hier_pct:13.1f}% {delta:+7.1f}%")

        # Print excluded queries for transparency
        print()
        print(f"Excluded queries ({n_excluded}):")
        for row in results_rows:
            if not row.get("tap_feasible", 1):
                print(f"  - {row['question'][:80]}")

    print("=" * 60)

    # --- Save CSV ---
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nDetailed results saved to: {output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
