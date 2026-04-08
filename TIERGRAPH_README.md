# TierGraph: Hierarchical Scene Graphs for Warehouse Object Retrieval

**Author**: Alexander Wang  
**Supervisor**: Steven Waslander (TRAIL Lab, University of Toronto)  
**Thesis**: B.A.Sc. Thesis, April 2026

TierGraph extends [OpenGraph](https://github.com/BIT-DYN/OpenGraph) with a 5-level warehouse hierarchy, structurally-constrained Information Bottleneck compression, and a cascading hierarchical retrieval algorithm. It is evaluated in an Isaac Sim warehouse with a Clearpath Husky robot.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data Collection (Isaac Sim)](#4-data-collection-isaac-sim)
5. [OpenGraph Perception Pipeline](#5-opengraph-perception-pipeline)
6. [TierGraph Pipeline](#6-tiergraph-pipeline)
7. [Evaluation and Benchmarking](#7-evaluation-and-benchmarking)
8. [Configuration](#8-configuration)
9. [Results and Outputs](#9-results-and-outputs)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

TierGraph organizes detected objects into a hierarchy that mirrors the physical warehouse layout:

```
Zone (receiving, staging, storage, forklift, hub_robot, pallet_truck, general)
 └─ Shelf (storage zone only: 7 shelves)
     └─ Section (118 sections across all shelves)
         └─ Object
 └─ Aisle (storage zone only: 6 aisles — sibling to shelves, not nested)
     └─ Object
 └─ Object (non-storage zones: objects are direct zone children)
```

**Note on storage zone structure**: Aisles and shelves are siblings under the storage zone. An object in an aisle walkway is assigned Zone → Aisle → Object. An object on a shelf is assigned Zone → Shelf → Section → Object (or Zone → Shelf → Object if no section matches).

The three thesis contributions implemented here are:

1. **Hierarchical scene graph construction** (`build_tiergraph.py`, `warehouse_graph_builder.py`) — geometric containment assignment using the known warehouse floor plan.
2. **Structurally-constrained IB compression** (`compress_tiergraph.py`, `utils/ib/`) — Agglomerative IB that only merges objects within the same structural unit (section, aisle, or zone).
3. **Cascading hierarchical retrieval** (`query_tiergraph.py`) — zone keyword matching → container mean-pool scoring → object cosine similarity.

---

## 2. Repository Structure

```
OpenGraph/
├── config/
│   └── isaac_warehouse.yaml        # Main config (edit sequence number here)
├── script/
│   ├── collect_isaac_data.py       # ROS2 data collector (Terminal 3)
│   ├── verify_collection.py        # Check collected frames with RGB overlay
│   ├── main_gen_cap.py             # Stage 1: VLM captioning + segmentation
│   ├── main_gen_pc.py              # Stage 2: 3D point cloud fusion
│   ├── build_scenegraph.py         # OpenGraph baseline flat scene graph
│   ├── build_tiergraph.py          # TierGraph: hierarchical graph construction
│   ├── compress_tiergraph.py       # IB compression
│   ├── query_tiergraph.py          # Flat + hierarchical retrieval evaluation
│   ├── benchmark_full.py           # Full benchmark: accuracy + latency
│   ├── sweep_compression.py        # Delta sweep (0.05–0.30) across task sets
│   ├── plot_sweep.py               # Plot sweep results
│   ├── scalability_table.py        # Node count table across sequences
│   ├── visualize.py                # OpenGraph baseline visualizer
│   ├── visualize_tiergraph.py      # TierGraph 3D visualizer (bounding boxes)
│   ├── visualize_hierarchy_tree.py # 2D logical hierarchy tree (PNG)
│   ├── visualize_hierarchy_3d.py   # 3D hierarchy overlay (spheres + edges)
│   └── visualize_compressed.py     # Compressed cluster visualizer
├── utils/
│   ├── warehouse_graph_builder.py  # Core: HierarchyNode, WarehouseGraphBuilder
│   ├── coordinate_alignment.py     # Isaac Sim global ↔ robot-relative transforms
│   ├── merge.py                    # OpenGraph object merging utilities
│   └── ib/
│       ├── ib_cluster.py           # Agglomerative IB algorithm (ClusterIB)
│       ├── tiergraph_bridge.py     # IB graph construction with structural constraints
│       └── information_metrics.py  # Shannon entropy, JS-divergence, mutual information
└── data/
    └── isaac_warehouse/            # Raw collected data (SemanticKITTI format)
        └── {sequence}/
            ├── image_2/            # RGB frames
            ├── velodyne/           # Depth point clouds (as .bin)
            ├── poses.txt           # Robot poses (robot-relative frame)
            └── calib.txt           # Camera calibration
```

Warehouse layout (shared with the navigation stack):
```
../Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json
```

Results are written to:
```
../results/warehouse_{sequence}/
    caption/    # VLM outputs
    vis/        # Annotated frame visualizations
    pcd/        # 3D maps and TierGraph outputs
        full_pcd.pkl.gz                     # OpenGraph object list
        object_relations.json               # TierGraph (full)
        object_relations_compressed.json    # TierGraph (IB-compressed)
        benchmark_full_results.csv          # Benchmark results
        sweep_compression_results.csv       # Delta sweep results
        latency_scale_test.csv              # Latency scaling data
        sweep_compression_plot.png          # Sweep figure
        hierarchy_3d_overlay.png            # 3D hierarchy figure
```

---

## 3. Environment Setup

Two conda environments are used:

| Environment | Used for |
|-------------|---------|
| `isaacsim` | Data collection only (`collect_isaac_data.py`) — needs ROS2 |
| `opengraph-cu118` | Everything else (OpenGraph pipeline, TierGraph, evaluation) |

```bash
conda activate opengraph-cu118
cd /home/awang/Documents/TRAILbot/OpenGraph
```

Model weights are pre-downloaded to:
- `~/opengraph_models/GroundingDINO/` — GroundingDINO
- `third_parties/weights/tag2text_swin_14m.pth` — Tag2Text
- `third_parties/weights/tap_vit_l_v1_0.pkl` + `merged_2560.pkl` — TAP
- SBERT (`sentence-transformers/all-MiniLM-L6-v2`) is loaded from HuggingFace automatically

---

## 4. Data Collection (Isaac Sim)

Data collection requires three terminals running simultaneously.

### Terminal 1 — Isaac Sim

```bash
cd /home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation
./run.sh sim.py --usd <warehouse_usd_path>
# or launch the GUI directly:
./isaac-sim.sh
```

### Terminal 2 — Navigation

```bash
cd /home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation
source install/setup.bash
ros2 launch husky_sim_navigation husky_navigation_ground_truth_with_amcl_complex_warehouse.launch.py

# Control the robot (in a separate terminal or tab):
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# Or use RViz "2D Goal Pose" button to set navigation goals.
```

Drive the robot through the full warehouse to ensure coverage of all zones. For Sequence 05 (the primary evaluation sequence), the robot traversed ~5,000 frames at stride 10, covering all 46 ground-truth query locations.

### Terminal 3 — Data Collection

```bash
cd /home/awang/Documents/TRAILbot/OpenGraph
conda activate isaacsim
python3 script/collect_isaac_data.py --mode depth --sequence <N> --use_sim_time
# Press Ctrl+C when done.
```

Replace `<N>` with the sequence number (e.g. `05`). Data is saved to `data/isaac_warehouse/<N>/` in SemanticKITTI format.

**Verify the collection** (in the `opengraph-cu118` environment):
```bash
conda activate opengraph-cu118
python script/verify_collection.py --sequence <N>
```
This overlays depth data on RGB frames and writes annotated images to confirm the RGB-D alignment is correct.

### Collected Sequences

| Seq | Frames | Raw Objects | Filtered | Notes |
|-----|--------|-------------|----------|-------|
| 01  | 240    | 25          | 2        | Sparse coverage |
| 02  | 484    | 45          | 21       | Early test |
| 03  | 687    | 137         | 78       | First benchmarked |
| 04  | 694    | 262         | 184      | Near-complete |
| 05  | 5,019  | 991         | 706      | **Primary — all 46 queries reachable** |

---

## 5. OpenGraph Perception Pipeline

The OpenGraph pipeline converts raw RGB-D frames into a list of detected objects with 3D point clouds, captions, and SBERT feature vectors. Run this before TierGraph.

Set the sequence in `config/isaac_warehouse.yaml` (field: `sequence`) or pass it on the command line.

```bash
conda activate opengraph-cu118
cd /home/awang/Documents/TRAILbot/OpenGraph

# Stage 1: VLM captioning and segmentation (~30–60 min per sequence)
# Tag2Text generates tags; GroundingDINO detects objects; TAP segments them.
python script/main_gen_cap.py --config-name=isaac_warehouse

# Stage 2: 3D point cloud fusion (~20–30 min per sequence)
# Projects detections onto depth point clouds and fuses across frames.
torchrun --nproc_per_node=1 script/main_gen_pc.py --config-name=isaac_warehouse

# Output: results/warehouse_{sequence}/pcd/full_pcd.pkl.gz
```

To view the raw OpenGraph output (flat scene graph, no warehouse hierarchy):
```bash
python script/build_scenegraph.py --config-name=isaac_warehouse
python script/visualize.py --config-name=isaac_warehouse
# Keys: B=background, C=class color, R=RGB, F=text query, I=instance, G=scene graph
```

---

## 6. TierGraph Pipeline

Run these after the OpenGraph pipeline has produced `full_pcd.pkl.gz`.

### 6.1 Build the Hierarchy

```bash
python script/build_tiergraph.py --config-name=isaac_warehouse
```

This applies structural filtering (wall/floor/extent filters) and assigns each surviving object to the correct position in the warehouse hierarchy via geometric containment checks. Output: `results/warehouse_{sequence}/pcd/object_relations.json`.

### 6.2 Visualize the Hierarchy

**2D logical tree** (saves PNG):
```bash
python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/tiergraph_hierarchy.png
```

**3D warehouse view** (interactive Open3D window, bounding boxes):
```bash
python script/visualize_tiergraph.py --config-name=isaac_warehouse
# Keys: 1=zones, 2=aisles, 3=shelves, 4=sections, 5=hierarchy edges,
#       6=color by level, I=color by instance, R=RGB
```

**3D hierarchy overlay** (spheres at hierarchy node centers, edges, objects colored by zone):
```bash
python script/visualize_hierarchy_3d.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/hierarchy_3d_overlay.png
```

### 6.3 IB Compression

```bash
python script/compress_tiergraph.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/object_relations_compressed.json
```

The IB delta threshold and task set are configured in `isaac_warehouse.yaml`. Default: `delta=0.15`, generic task set.

Visualize the compressed graph:
```bash
python script/visualize_compressed.py --config-name=isaac_warehouse

# Or view the compressed hierarchy tree:
python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse \
    'scenegraph_path=/home/awang/Documents/TRAILbot/results/warehouse_<N>/pcd/object_relations_compressed.json'
```

---

## 7. Evaluation and Benchmarking

All evaluation scripts require `object_relations.json` (and optionally `object_relations_compressed.json`) to exist.

### 7.1 Single Query Run

Interactive retrieval for a single sequence:
```bash
python script/query_tiergraph.py --config-name=isaac_warehouse
```

Prints flat vs. hierarchical Precision@1/3/5 results for all 46 queries and the Tag2Text-feasible subset (30 queries).

### 7.2 Full Benchmark

Runs flat and hierarchical retrieval, latency measurements, and synthetic scaling in one script:
```bash
python script/benchmark_full.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/benchmark_full_results.csv
#         results/warehouse_{sequence}/pcd/latency_scale_test.csv
```

### 7.3 Compression Delta Sweep

Sweeps delta from 0.05 to 0.30 for both the generic and eval-derived task sets:
```bash
python script/sweep_compression.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/sweep_compression_results.csv
```

Plot the sweep results:
```bash
python script/plot_sweep.py --config-name=isaac_warehouse
# Output: results/warehouse_{sequence}/pcd/sweep_compression_plot.png
```

### 7.4 Scalability Table

Node counts across all sequences:
```bash
python script/scalability_table.py --config-name=isaac_warehouse
```

---

## 8. Configuration

All scripts use [Hydra](https://hydra.cc/) with `config/isaac_warehouse.yaml`. The most commonly changed fields are:

| Field | Default | Description |
|-------|---------|-------------|
| `sequence` | `"05"` | Sequence to process (must be a quoted string) |
| `stride` | `10` | Process every Nth frame |
| `start` / `end` | `0` / `-1` | Frame range (`-1` = all) |
| `max_depth` | `15` | Max point cloud depth in metres |
| `sim_threshold` | `0.6` | Object fusion similarity threshold |
| `bg_rate` | `0.80` | Background classification threshold |
| `dbscan_eps` | `0.3` | DBSCAN cluster radius for denoising |
| `voxel_size` | `0.05` | Voxel size for point cloud downsampling |

Override any field on the command line:
```bash
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=03
python script/sweep_compression.py --config-name=isaac_warehouse sequence=05
```

---

## 9. Results and Outputs

Primary results are in `results/warehouse_05/pcd/` (Sequence 05 is the main evaluation sequence).

### Key files

| File | Description |
|------|-------------|
| `full_pcd.pkl.gz` | OpenGraph object list (input to TierGraph) |
| `object_relations.json` | Full TierGraph (hierarchy + object nodes) |
| `object_relations_compressed.json` | IB-compressed TierGraph |
| `benchmark_full_results.csv` | P@k and latency for all conditions |
| `sweep_compression_results.csv` | P@k and ratio across delta 0.05–0.30 |
| `latency_scale_test.csv` | Latency from 50 to 10,000 objects |
| `sweep_compression_plot.png` | Four-panel sweep figure |
| `hierarchy_3d_overlay.png` | 3D hierarchy overlay figure |
| `tiergraph_hierarchy.png` | 2D hierarchy tree diagram |

### Summary results (Sequence 05, 706 objects, 46 queries)

| Condition | P@1 | P@5 |
|-----------|-----|-----|
| Flat retrieval (full graph) | 26.1% | 50.0% |
| Hierarchical retrieval (full graph) | 28.3% | 54.3% |
| Hierarchical + IB compression (generic, δ=0.25, 228 units, 3.10x) | 54.5% | 75.8% |
| Hierarchical + IB compression (eval, δ=0.20, 163 units, 4.33x) | 54.5% | 75.8% |

Latency crossover between flat and hierarchical retrieval: ~6,300 objects. At 10,000 objects, hierarchical is 1.75× faster.

---

## 10. Key Design Decisions

**Coordinate system.** Isaac Sim uses a global frame (X forward, Y left, Z up) with floor at Z = −0.14 m. The robot's first-frame pose is extracted from `poses.txt` and used to compute an inverse transform `T_inv` that maps all layout bounding boxes into the robot-relative frame. Object centroids from OpenGraph receive a +0.781 m Z offset (robot base_link elevation) to align with the layout frame. Layout bounds receive a +0.14 m Z offset to shift the floor to Z = 0.

**Structural filtering.** Three filters remove non-inventory detections before hierarchy assignment: (1) wall filter — centroid XY outside the warehouse floor polygon; (2) floor filter — centroid Z < 0.4 m (with exception for objects inside a functional zone); (3) extent filter — bounding box > 6.0 m in any axis. On Sequence 05, this removes 285 of 991 raw objects (28.8%).

**Storage zone geometry.** The storage zone is a rotated rectangle and uses polygon containment (`matplotlib.path.Path.contains_point`) rather than a simple AABB. All other zones use AABB containment.

**IB merge constraints.** Objects can only merge with others in the same structural unit: same section, same aisle, or same zone (for non-storage zones). This is enforced by constructing a fully-connected IB adjacency graph within each unit and zero connections across units. The constraint prevents clusters from spanning physically separate locations.

**Tag2Text-feasible queries.** 16 of the 46 evaluation queries target objects that Tag2Text cannot reliably detect (label reading, small wall-mounted items, fine-grained identification). These are excluded from the "feasible" subset (30 queries) to give a fairer view of retrieval performance.

---

## 11. Troubleshooting

**`FileNotFoundError: full_pcd.pkl.gz`**  
Run `main_gen_cap.py` and `main_gen_pc.py` first. Verify the sequence number matches what is in `isaac_warehouse.yaml`.

**`FileNotFoundError: object_relations.json`**  
Run `build_tiergraph.py` before any evaluation or compression script.

**`FileNotFoundError: warehouse_layout.json`**  
The layout file is in `../Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/`. Verify the `Isaac-sim-husky-navigation` repo is checked out at the same level as `TRAILbot/OpenGraph`.

**"Object not in any zone" warnings**  
Expected for objects that pass structural filtering but fall outside all zone bounding boxes (edge cases near zone boundaries). These are logged but not discarded — they fall back to the nearest zone by distance.

**CUDA out of memory during `main_gen_pc.py`**  
Reduce the sequence stride (increase `stride` in config) or process a shorter frame range with `start`/`end`.

**`main_gen_cap.py` is slow**  
Expected: ~30–60 minutes per 500 frames. The Tag2Text, GroundingDINO, and TAP models all run per frame. Use `stride` to skip frames.

**Compressed graph is identical to the full graph**  
The IB algorithm may have converged early if `delta` is too small or the sequence has very few objects. Try increasing `delta` (e.g. 0.20 or 0.25) or check that `compress_tiergraph.py` printed non-zero merge counts.

---

## Citation

For the upstream OpenGraph paper:

```bibtex
@article{opengraph,
  title={OpenGraph: Open-Vocabulary Hierarchical 3D Graph Representation in Large-Scale Outdoor Environments},
  author={Deng, Yinan and Wang, Jiahui and Zhao, Jingyu and Tian, Xinyu and Chen, Guangyan and Yang, Yi and Yue, Yufeng},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  volume={9},
  number={10},
  pages={8402--8409},
}
```

For TierGraph:

```bibtex
@thesis{wang2026tiergraph,
  title={TierGraph: Hierarchical Scene Graphs for Structured Warehouse Object Retrieval},
  author={Wang, Alexander},
  year={2026},
  school={University of Toronto},
  type={B.A.Sc. Thesis},
}
```
