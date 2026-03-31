#!/usr/bin/env python3
"""
Full Evaluation Benchmark Suite — TierGraph (TODO 2.4)

Runs all three benchmark categories in one script:

  1. RETRIEVAL ACCURACY
     - Flat OpenGraph (no hierarchy) vs TierGraph hierarchical
     - Full TierGraph vs compressed TierGraph (at default delta)
     - Reports P@1, P@3, P@5 across all/reachable/TAP-feasible splits

  2. COMPRESSION EFFICIENCY
     - Node reduction ratio at the configured delta
     - (Full delta sweep is in sweep_compression.py)

  3. QUERY LATENCY
     - Wall-clock time per query: flat vs hierarchical
     - Synthetic scale test: extrapolate to 100–10,000 objects

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/benchmark_full.py --config-name=isaac_warehouse sequence=03

Author: awang (TierGraph thesis)
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import csv
import gzip
import json
import pickle
import time
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from some_class.map_calss import MapObjectList
from utils.coordinate_alignment import load_absolute_first_pose


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

TAP_EXCLUDE_PATTERNS = [
    'labeled "Digital Twin"', "labeled 'Digital Twin'",
    '"EQP" symbol', '"UNK" symbol', "star-shaped symbol",
    "FRAGILE labeled", "first aid kit", "disinfectant", "flashlight",
    "fire extinguisher", "small bottles", "electrical box", "dock-high doors",
]

SCALE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_tap_feasible(question: str) -> bool:
    q_lower = question.lower()
    for p in TAP_EXCLUDE_PATTERNS:
        if p.lower() in q_lower:
            return False
    return True


def load_objects(path: Path) -> MapObjectList:
    with gzip.open(path, 'rb') as f:
        results = pickle.load(f)
    objects = MapObjectList()
    if isinstance(results, dict):
        objects.load_serializable(results["objects"])
    elif isinstance(results, list):
        objects.load_serializable(results)
    return objects


def get_centroid(node: dict) -> np.ndarray:
    c = node["bounds"]["center"]
    return np.array([c["x"], c["y"], c["z"]])


def parse_queries(csv_path: str, T_inv: np.ndarray, z_offset=0.781) -> list:
    queries = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            center_str = row.get("Center", "").strip()
            if not center_str:
                continue
            question = row.get("Question", "").strip()
            if not question:
                continue
            parts = [float(x) for x in center_str.strip("[]").split(",")]
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
                "gt_robot": gt_robot,
                "tap_feasible": is_tap_feasible(question),
            })
    return [q for q in queries if q["query_type"] == "position"]


def build_retrieval_inputs(graph, objects):
    """Build feature matrix from object + cluster nodes."""
    node_lookup = {n["id"]: n for n in graph["nodes"]}
    features, node_ids = [], []

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
                features.append(ft)
                node_ids.append(node["id"])

        elif node["type"] == "cluster":
            mf = node.get("metadata", {}).get("mean_feature")
            if mf is not None:
                ft = np.array(mf)
                if ft.ndim == 2:
                    ft = ft[0]
                features.append(ft)
                node_ids.append(node["id"])

    if not features:
        return np.zeros((0, 384)), [], node_lookup
    return np.stack(features), node_ids, node_lookup


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

def _detect_zone(text: str) -> str:
    text_lower = text.lower()
    for zid, kws in ZONE_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                return zid
    return None


def _find_zone(node, lookup):
    cur = node
    for _ in range(10):
        if cur["type"] == "zone":
            return cur["id"]
        pid = cur.get("parent_id")
        if not pid or pid not in lookup:
            return None
        cur = lookup[pid]
    return None


def flat_retrieval(q_emb, features, top_k=5):
    q = torch.from_numpy(q_emb).float().unsqueeze(0)
    o = torch.from_numpy(features).float()
    sims = F.cosine_similarity(q, o, dim=1)
    scores, indices = torch.topk(sims, min(top_k, len(sims)))
    return list(zip(indices.tolist(), scores.tolist()))


def hierarchical_retrieval(q_emb, features, node_ids, lookup,
                            query_text="", top_k=5,
                            zone_top_n=2, container_top_n=5):
    q = torch.from_numpy(q_emb).float()

    # Zone level
    detected = _detect_zone(query_text)
    zone_objs = {}
    for fi, nid in enumerate(node_ids):
        node = lookup.get(nid)
        if not node:
            continue
        zid = _find_zone(node, lookup)
        if zid:
            zone_objs.setdefault(zid, []).append(fi)

    if detected and detected in zone_objs:
        cands = zone_objs[detected]
    else:
        zscores = []
        for zid, fis in zone_objs.items():
            zfeat = torch.from_numpy(features[fis]).float().mean(dim=0)
            sc = F.cosine_similarity(q.unsqueeze(0), zfeat.unsqueeze(0)).item()
            zscores.append((zid, sc, fis))
        zscores.sort(key=lambda x: x[1], reverse=True)
        cands = []
        for _, _, fis in zscores[:zone_top_n]:
            cands.extend(fis)

    if not cands:
        return flat_retrieval(q_emb, features, top_k)

    # Container level
    cont_objs = {}
    for fi in cands:
        node = lookup.get(node_ids[fi])
        if not node:
            continue
        pid = node.get("parent_id", "unk")
        cont_objs.setdefault(pid, []).append(fi)

    cscores = []
    for cid, fis in cont_objs.items():
        cfeat = torch.from_numpy(features[fis]).float().mean(dim=0)
        sc = F.cosine_similarity(q.unsqueeze(0), cfeat.unsqueeze(0)).item()
        cscores.append((cid, sc, fis))
    cscores.sort(key=lambda x: x[1], reverse=True)

    final = []
    for _, _, fis in cscores[:container_top_n]:
        final.extend(fis)

    if not final:
        return flat_retrieval(q_emb, features, top_k)

    # Object level
    ff = torch.from_numpy(features[final]).float()
    sims = F.cosine_similarity(q.unsqueeze(0), ff, dim=1)
    k = min(top_k, len(sims))
    scores, lidx = torch.topk(sims, k)
    return [(final[li], sc) for li, sc in zip(lidx.tolist(), scores.tolist())]


def eval_retrieval(results, node_ids, lookup, gt_robot,
                    threshold=5.0, ks=(1, 3, 5)):
    ev = {f"hit@{k}": 0 for k in ks}
    ev["min_dist"] = float("inf")
    for rank, (fi, _) in enumerate(results):
        node = lookup.get(node_ids[fi])
        if not node:
            continue
        dist = np.linalg.norm(get_centroid(node) - gt_robot)
        ev["min_dist"] = min(ev["min_dist"], dist)
        if dist <= threshold:
            for k in ks:
                if rank < k:
                    ev[f"hit@{k}"] = 1
    return ev


# ---------------------------------------------------------------------------
# Run full evaluation on a graph
# ---------------------------------------------------------------------------

def run_full_eval(graph, objects, queries, sbert, measure_latency=False):
    """Returns metrics dict + per-query latency arrays."""
    features, node_ids, lookup = build_retrieval_inputs(graph, objects)
    if len(node_ids) == 0:
        return None

    ks = (1, 3, 5)
    n = len(queries)

    # Pre-compute oracle reachability
    retrievable_nodes = [lookup[nid] for nid in node_ids if nid in lookup]
    reachable_mask = []
    for q in queries:
        min_d = min((np.linalg.norm(get_centroid(nd) - q["gt_robot"])
                     for nd in retrievable_nodes), default=float("inf"))
        reachable_mask.append(min_d <= 5.0)

    # Accumulators for all splits
    splits = {
        "all": lambda qi: True,
        "reach": lambda qi: reachable_mask[qi],
        "feasible": lambda qi: queries[qi]["tap_feasible"],
        "feas_reach": lambda qi: (reachable_mask[qi] and
                                   queries[qi]["tap_feasible"]),
    }
    counts = {s: 0 for s in splits}
    flat_hits = {s: {k: 0 for k in ks} for s in splits}
    hier_hits = {s: {k: 0 for k in ks} for s in splits}

    for s, pred in splits.items():
        counts[s] = sum(1 for qi in range(n) if pred(qi))

    flat_times = []
    hier_times = []

    for qi, q in enumerate(queries):
        q_emb = sbert.encode(q["question"], normalize_embeddings=True)

        # Flat
        t0 = time.perf_counter()
        flat_res = flat_retrieval(q_emb, features, top_k=5)
        flat_times.append(time.perf_counter() - t0)
        flat_ev = eval_retrieval(flat_res, node_ids, lookup, q["gt_robot"])

        # Hierarchical
        t0 = time.perf_counter()
        hier_res = hierarchical_retrieval(q_emb, features, node_ids, lookup,
                                           query_text=q["question"], top_k=5)
        hier_times.append(time.perf_counter() - t0)
        hier_ev = eval_retrieval(hier_res, node_ids, lookup, q["gt_robot"])

        for s, pred in splits.items():
            if pred(qi):
                for k in ks:
                    flat_hits[s][k] += flat_ev[f"hit@{k}"]
                    hier_hits[s][k] += hier_ev[f"hit@{k}"]

    result = {
        "n_nodes": len(node_ids),
        "n_reachable": counts["reach"],
        "n_feasible": counts["feasible"],
        "n_feasible_reachable": counts["feas_reach"],
    }

    for s in splits:
        c = counts[s]
        for k in ks:
            result[f"flat_P@{k}_{s}"] = round(100 * flat_hits[s][k] / c, 1) if c else 0
            result[f"hier_P@{k}_{s}"] = round(100 * hier_hits[s][k] / c, 1) if c else 0

    result["flat_latency_ms"] = round(np.mean(flat_times) * 1000, 3)
    result["hier_latency_ms"] = round(np.mean(hier_times) * 1000, 3)
    result["flat_latency_std_ms"] = round(np.std(flat_times) * 1000, 3)
    result["hier_latency_std_ms"] = round(np.std(hier_times) * 1000, 3)

    return result


# ---------------------------------------------------------------------------
# Synthetic latency scaling test
# ---------------------------------------------------------------------------

def latency_scale_test(sbert, dim=384):
    """Measure flat and hierarchical retrieval time at varying object counts.
    Uses synthetic random features to isolate compute cost from data loading."""
    print("\n" + "=" * 60)
    print("LATENCY SCALE TEST (synthetic features)")
    print("=" * 60)

    q_emb = np.random.randn(dim).astype(np.float32)
    q_emb /= np.linalg.norm(q_emb)

    rows = []
    for n_obj in SCALE_SIZES:
        features = np.random.randn(n_obj, dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.clip(norms, 1e-8, None)

        # Flat timing (average over multiple runs for stability)
        n_runs = max(10, 1000 // n_obj)
        times_flat = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            q = torch.from_numpy(q_emb).float().unsqueeze(0)
            o = torch.from_numpy(features).float()
            sims = F.cosine_similarity(q, o, dim=1)
            torch.topk(sims, min(5, len(sims)))
            times_flat.append(time.perf_counter() - t0)

        # Hierarchical timing (simulate 2-zone narrowing: search 30% of objects)
        n_zone_subset = max(5, n_obj * 3 // 10)
        zone_indices = list(range(n_zone_subset))
        # Container narrowing: top 5 containers of ~5 objects each
        n_container_subset = min(25, n_zone_subset)
        container_indices = list(range(n_container_subset))

        times_hier = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            # Zone scoring (mean-pool per zone, score 7 zones)
            n_zones = 7
            zone_size = max(1, n_obj // n_zones)
            for z in range(n_zones):
                start = z * zone_size
                end = min(start + zone_size, n_obj)
                if start >= n_obj:
                    break
                zfeat = torch.from_numpy(features[start:end]).float().mean(dim=0)
                F.cosine_similarity(q, zfeat.unsqueeze(0))

            # Container scoring on subset
            subset_feats = torch.from_numpy(features[zone_indices]).float()
            n_containers = max(1, len(zone_indices) // 5)
            for ci in range(min(n_containers, 20)):
                start = ci * 5
                end = min(start + 5, len(zone_indices))
                cfeat = subset_feats[start:end].mean(dim=0)
                F.cosine_similarity(q, cfeat.unsqueeze(0))

            # Final object ranking on container subset
            final_feats = torch.from_numpy(
                features[container_indices]).float()
            sims = F.cosine_similarity(q, final_feats, dim=1)
            torch.topk(sims, min(5, len(sims)))
            times_hier.append(time.perf_counter() - t0)

        flat_ms = np.mean(times_flat) * 1000
        hier_ms = np.mean(times_hier) * 1000
        speedup = flat_ms / hier_ms if hier_ms > 0 else 0

        rows.append({
            "n_objects": n_obj,
            "flat_ms": round(flat_ms, 3),
            "hier_ms": round(hier_ms, 3),
            "speedup": round(speedup, 2),
        })
        print(f"  {n_obj:>6d} objects: flat={flat_ms:>8.3f}ms  "
              f"hier={hier_ms:>8.3f}ms  speedup={speedup:.2f}x")

    return rows


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_comparison_table(label, r1, r1_name, r2, r2_name):
    """Print a precision comparison table for two conditions."""
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  {r1_name}: {r1['n_nodes']} nodes  |  "
          f"{r2_name}: {r2['n_nodes']} nodes")
    print(f"  Reachable: {r1['n_reachable']}  |  "
          f"TAP-feasible+reachable: {r1['n_feasible_reachable']}")

    for split, split_label in [
        ("all", "All queries (46)"),
        ("reach", "Reachable only"),
        ("feas_reach", "TAP-feasible + reachable"),
    ]:
        print(f"\n  --- {split_label} ---")
        print(f"  {'Metric':<10} {r1_name+' flat':>14} {r1_name+' hier':>14} "
              f"{r2_name+' flat':>14} {r2_name+' hier':>14}")
        print(f"  {'-' * 60}")
        for k in (1, 3, 5):
            v1f = r1[f"flat_P@{k}_{split}"]
            v1h = r1[f"hier_P@{k}_{split}"]
            v2f = r2[f"flat_P@{k}_{split}"]
            v2h = r2[f"hier_P@{k}_{split}"]
            print(f"  P@{k:<7} {v1f:>13.1f}% {v1h:>13.1f}% "
                  f"{v2f:>13.1f}% {v2h:>13.1f}%")

    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config",
            config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "=" * 65)
    print("  FULL EVALUATION BENCHMARK SUITE — TierGraph")
    print("=" * 65 + "\n")

    # Paths
    tiergraph_path = Path(cfg.scenegraph_path)
    result_path = Path(cfg.result_path)
    sequence_dir = Path(cfg.basedir) / cfg.sequence
    compressed_path = tiergraph_path.parent / "object_relations_compressed.json"
    query_csv = str(
        Path(__file__).parent.parent / "data" / "warehouse_layout"
        / "warehouse_task_list.csv"
    )
    output_dir = tiergraph_path.parent
    output_csv = output_dir / "benchmark_full_results.csv"

    # Load shared data
    print("Loading SBERT model...")
    sbert = SentenceTransformer(cfg.sbert_path)

    print(f"Loading objects from {result_path}...")
    objects = load_objects(result_path)
    print(f"  {len(objects)} objects")

    T_first = load_absolute_first_pose(sequence_dir)
    T_inv = np.linalg.inv(T_first)

    print(f"Parsing queries...")
    queries = parse_queries(query_csv, T_inv)
    n_feasible = sum(1 for q in queries if q["tap_feasible"])
    print(f"  {len(queries)} position queries ({n_feasible} TAP-feasible)")

    # =====================================================================
    # 1. RETRIEVAL ACCURACY — Full TierGraph (flat vs hierarchical)
    # =====================================================================
    print("\n" + "#" * 65)
    print("  BENCHMARK 1: RETRIEVAL ACCURACY")
    print("#" * 65)

    print(f"\nLoading TierGraph from {tiergraph_path}...")
    with open(tiergraph_path, 'r') as f:
        tiergraph = json.load(f)
    n_obj = sum(1 for n in tiergraph['nodes'] if n['type'] == 'object')
    print(f"  {n_obj} object nodes")

    print("Running evaluation on full TierGraph...")
    full_result = run_full_eval(tiergraph, objects, queries, sbert)

    # Print standalone results
    print(f"\n--- Full TierGraph ({full_result['n_nodes']} nodes) ---")
    for split, label in [("all", "All"), ("reach", "Reachable"),
                          ("feas_reach", "Feasible+Reachable")]:
        fp1 = full_result[f"flat_P@1_{split}"]
        hp1 = full_result[f"hier_P@1_{split}"]
        fp5 = full_result[f"flat_P@5_{split}"]
        hp5 = full_result[f"hier_P@5_{split}"]
        delta1 = hp1 - fp1
        print(f"  {label:<22s} P@1: flat={fp1:5.1f}%  hier={hp1:5.1f}% "
              f"(Δ{delta1:+.1f}%)  P@5: flat={fp5:5.1f}%  hier={hp5:5.1f}%")

    # =====================================================================
    # 2. RETRIEVAL ACCURACY — Compressed TierGraph
    # =====================================================================
    compressed_result = None
    if compressed_path.exists():
        print(f"\nLoading compressed TierGraph from {compressed_path}...")
        with open(compressed_path, 'r') as f:
            compressed = json.load(f)
        by_type = compressed["statistics"]["nodes_by_type"]
        print(f"  {by_type.get('cluster', 0)} clusters + "
              f"{by_type.get('object', 0)} singletons")

        print("Running evaluation on compressed TierGraph...")
        compressed_result = run_full_eval(compressed, objects, queries, sbert)

        print_comparison_table(
            "Full TierGraph vs Compressed TierGraph",
            full_result, "Full",
            compressed_result, "Comp",
        )

        comp_info = compressed.get("metadata", {}).get("compression", {})
        orig = comp_info.get("original_object_count", n_obj)
        after = (by_type.get("cluster", 0) + by_type.get("object", 0))
        ratio = orig / max(after, 1)
        print(f"\n  Compression: {orig} objects → {after} units "
              f"({ratio:.2f}x reduction)")
    else:
        print(f"\n  No compressed TierGraph found at {compressed_path} — skipping.")

    # =====================================================================
    # 3. QUERY LATENCY — Real data
    # =====================================================================
    print("\n" + "#" * 65)
    print("  BENCHMARK 2: QUERY LATENCY")
    print("#" * 65)

    print(f"\n--- Latency on real data ({full_result['n_nodes']} nodes) ---")
    print(f"  Flat:         {full_result['flat_latency_ms']:>8.3f} ms/query "
          f"(± {full_result['flat_latency_std_ms']:.3f})")
    print(f"  Hierarchical: {full_result['hier_latency_ms']:>8.3f} ms/query "
          f"(± {full_result['hier_latency_std_ms']:.3f})")

    if full_result['flat_latency_ms'] > 0:
        ratio = full_result['hier_latency_ms'] / full_result['flat_latency_ms']
        print(f"  Ratio:        {ratio:.2f}x "
              f"({'slower' if ratio > 1 else 'faster'})")

    if compressed_result:
        print(f"\n--- Latency on compressed data ({compressed_result['n_nodes']} nodes) ---")
        print(f"  Flat:         {compressed_result['flat_latency_ms']:>8.3f} ms/query "
              f"(± {compressed_result['flat_latency_std_ms']:.3f})")
        print(f"  Hierarchical: {compressed_result['hier_latency_ms']:>8.3f} ms/query "
              f"(± {compressed_result['hier_latency_std_ms']:.3f})")

    # Synthetic scale test
    scale_rows = latency_scale_test(sbert)

    # =====================================================================
    # 4. COMPRESSION EFFICIENCY summary
    # =====================================================================
    print("\n" + "#" * 65)
    print("  BENCHMARK 3: COMPRESSION EFFICIENCY")
    print("#" * 65)

    if compressed_result and compressed_path.exists():
        with open(compressed_path, 'r') as f:
            comp_graph = json.load(f)
        comp_info = comp_graph.get("metadata", {}).get("compression", {})
        bt = comp_graph["statistics"]["nodes_by_type"]

        print(f"\n  Original objects:  {comp_info.get('original_object_count', '?')}")
        print(f"  Clusters created:  {bt.get('cluster', 0)}")
        print(f"  Singletons kept:   {bt.get('object', 0)}")
        print(f"  Total units after: {bt.get('cluster', 0) + bt.get('object', 0)}")
        print(f"  Objects merged:    {comp_info.get('objects_merged', '?')}")
        print(f"  Compression ratio: "
              f"{comp_info.get('original_object_count', 0) / max(bt.get('cluster', 0) + bt.get('object', 0), 1):.2f}x")
        print(f"\n  (Full delta sweep results in sweep_compression_results.csv)")
    else:
        print("\n  No compressed TierGraph available.")

    # =====================================================================
    # Save all results
    # =====================================================================
    print("\n" + "#" * 65)
    print("  SAVING RESULTS")
    print("#" * 65)

    # Build CSV rows
    csv_rows = []

    # Full TierGraph row
    row = {"condition": "full_tiergraph", **full_result}
    csv_rows.append(row)

    # Compressed row
    if compressed_result:
        row = {"condition": "compressed_tiergraph", **compressed_result}
        csv_rows.append(row)

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Retrieval results: {output_csv}")

    # Save latency scale test
    scale_csv = output_dir / "latency_scale_test.csv"
    if scale_rows:
        with open(scale_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scale_rows[0].keys())
            writer.writeheader()
            writer.writerows(scale_rows)
        print(f"  Latency scale test: {scale_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
