#!/usr/bin/env python3
"""Generate scalability table across all sequences."""

import json, gzip, pickle, csv, os
from pathlib import Path

seqs = ['01', '02', '03', '04', '05']
results_base = Path(__file__).parent.parent.parent / 'results'
data_base = Path(__file__).parent.parent / 'data' / 'isaac_warehouse'

header = ['Sequence', 'Frames', 'Raw Objects', 'Filtered Objects', 'Compressed Units', 'Compression Ratio']
rows = []

print(f"{'Seq':>4} | {'Frames':>7} | {'Raw Obj':>8} | {'Filtered':>9} | {'Compressed':>11} | {'Ratio':>6}")
print("-" * 65)

for seq in seqs:
    pcd_dir = results_base / f"warehouse_{seq}" / "pcd"

    # Count frames
    img_dir = data_base / seq / "image_2"
    n_frames = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0

    # Raw object count from full_pcd
    pcd_path = pcd_dir / "full_pcd.pkl.gz"
    n_raw = 0
    if pcd_path.exists():
        with gzip.open(pcd_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "objects" in data:
            n_raw = len(data["objects"])
        elif isinstance(data, list):
            n_raw = len(data)

    # Filtered object count from TierGraph
    tg_path = pcd_dir / "object_relations.json"
    n_filtered = 0
    if tg_path.exists():
        with open(tg_path) as f:
            tg = json.load(f)
        if isinstance(tg, dict) and "nodes" in tg:
            n_filtered = sum(1 for n in tg["nodes"] if n.get("type") == "object")

    # Compressed count
    comp_path = pcd_dir / "object_relations_compressed.json"
    n_compressed = "-"
    ratio = "-"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        if isinstance(comp, dict) and "nodes" in comp:
            clusters = sum(1 for n in comp["nodes"] if n.get("type") == "cluster")
            singletons = sum(1 for n in comp["nodes"] if n.get("type") == "object")
            n_compressed = clusters + singletons
            if n_compressed > 0 and n_filtered > 0:
                ratio = f"{n_filtered / n_compressed:.2f}x"

    print(f"{seq:>4} | {n_frames:>7} | {n_raw:>8} | {n_filtered:>9} | {str(n_compressed):>11} | {str(ratio):>6}")
    rows.append([seq, n_frames, n_raw, n_filtered, n_compressed, ratio])

# Save CSV
out_csv = results_base / "scalability_table.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"\nSaved to {out_csv}")
