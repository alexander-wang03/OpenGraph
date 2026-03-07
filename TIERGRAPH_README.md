# TierGraph: Hierarchical Scene Graphs for Warehouse Robotics

**Author**: awang
**Date**: 2026-02-10
**Thesis Project**: 5-Level Hierarchical Scene Graph for Warehouse Navigation

---

## Overview

TierGraph extends OpenGraph with a hierarchical 5-level scene graph structure designed specifically for warehouse environments:

```
Zone (Storage, Receiving, Staging, etc.)
 └─ Aisle (Corridors between shelves)
     └─ Shelf (Shelving units)
         └─ Section (Tiers/levels on shelves)
             └─ Object (Detected items)
```

**Key Innovation**: Replace OpenGraph's flat MST-based scene graph with explicit warehouse semantics and hierarchical organization.

---

## Quick Start

### 1. Test TierGraph (No Data Required)

```bash
cd /home/awang/Documents/TRAILbot/OpenGraph
python script/test_tiergraph.py
```

This tests the hierarchy building and object assignment logic with synthetic test objects.

### 2. Full Pipeline (OpenGraph → TierGraph)

```bash
# Step 1: Generate captions (VLM + segmentation)
python script/main_gen_cap.py --config-name=isaac_warehouse sequence=01

# Step 2: 3D point cloud mapping
torchrun --nproc_per_node=1 script/main_gen_pc.py --config-name=isaac_warehouse sequence=01

# Step 3: Build TierGraph (replaces build_scenegraph.py)
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01

# Optional: Visualize (currently shows OpenGraph objects, not TierGraph hierarchy)
python script/visualize.py --config-name=isaac_warehouse sequence=01
```

### 3. TierGraph Only (OpenGraph Results Already Exist)

If you've already run steps 1-2 and just want to rebuild the hierarchy:

```bash
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01
```

### 4. Visualize TierGraph

**3D Interactive Visualization** (shows infrastructure bounding boxes + objects):
```bash
python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01
```

Controls:
- `[1]` Toggle zone bounding boxes (red)
- `[2]` Toggle aisle bounding boxes (green)
- `[3]` Toggle shelf bounding boxes (blue)
- `[4]` Toggle section bounding boxes (yellow)
- `[5]` Toggle hierarchy edges (parent→child connections)
- `[6]` Color by hierarchy level
- `[I]` Color by instance (random colors)
- `[R]` Color by RGB (original colors)

**2D Hierarchy Tree** (shows logical structure):
```bash
python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=01
```

Saves PNG: `results/warehouse_{sequence}/pcd/tiergraph_hierarchy.png`

---

## Architecture

### Files

| File | Purpose |
|------|---------|
| `utils/warehouse_graph_builder.py` | Core TierGraph implementation |
| `script/build_tiergraph.py` | Main script to build hierarchy from OpenGraph results |
| `script/test_tiergraph.py` | Unit test with synthetic objects |
| `script/visualize_tiergraph.py` | 3D interactive visualization with bounding boxes |
| `script/visualize_hierarchy_tree.py` | 2D tree diagram of hierarchy structure |
| `config/isaac_warehouse.yaml` | Warehouse-tuned config (dataset_type: isaac) |

### Algorithm

**For each OpenGraph object** (with 3D centroid from point cloud):

1. **Find containing Zone**: Check which functional zone (storage/receiving/staging) contains the point
2. **Find containing Aisle**: Check which aisle (corridor) in that zone contains the point
3. **Find containing Shelf**: Check which shelving unit contains the point
4. **Find containing Section**: Check which section (tier/level) on that shelf contains the point
5. **Create hierarchy edges**: Zone→Aisle→Shelf→Section→Object

All checks use 3D axis-aligned bounding box containment.

---

## Input/Output

### Input 1: OpenGraph Results
Path: `OpenGraph/results/warehouse_{sequence}/pcd/full_pcd.pkl.gz`

Contains:
- List of detected objects
- Each object has: 3D point cloud, caption, CLIP features, detection confidence

### Input 2: Warehouse Layout
Path: `Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json`

Contains:
- 7 functional zones (storage, receiving, staging, etc.)
- 6 aisles in storage zone
- 7 shelves in storage zone
- 118 sections (tiers) across all shelves
- 3D bounding boxes for all infrastructure

### Output: TierGraph JSON
Path: `OpenGraph/results/warehouse_{sequence}/pcd/object_relations.json`

Contains:
- Nodes: All zones, aisles, shelves, sections, and objects (250+ nodes)
- Edges: Containment relationships (parent → child)
- Statistics: Node counts by type, max depth, etc.

---

## Comparison: OpenGraph vs TierGraph

| Aspect | OpenGraph (Baseline) | TierGraph (Yours) |
|--------|---------------------|------------------|
| **Structure** | Flat MST graph | 5-level hierarchy |
| **Organization** | Spatial proximity only | Warehouse semantics |
| **Nodes** | Objects only (~100) | Infrastructure + objects (~250) |
| **Edges** | N-1 (MST) | Containment (parent→child) |
| **Queries** | BFS/DFS traversal | O(1) hierarchical lookup |
| **Semantics** | Generic scene graph | Zone/Aisle/Shelf structure |

**Example Query**:
- OpenGraph: "Find objects near X" → Traverse MST edges from X
- TierGraph: "Find objects in Aisle 2, Shelf 3, Section 0" → Direct lookup in hierarchy

---

## Data Collection Status

| Sequence | Frames | Status | Location |
|----------|--------|--------|----------|
| 01 | 240 | ✓ Collected | `data/isaac_warehouse/01/` |
| 06 | 217 | ✓ Collected | `data/isaac_warehouse/06/` |
| 07 | 412 | ✓ Collected | `data/isaac_warehouse/07/` |
| 08 | 276 | ✓ Collected | `data/isaac_warehouse/08/` |

All sequences collected with fixed IsaacDataset (no Tr transform on poses).

---

## Thesis Contribution

### Implementation Strategy

1. **Fixed Warehouse Layout**: Uses `warehouse_layout.json` extracted from Isaac Sim
   - Semi-automated extraction from Isaac Sim asset hierarchy
   - Manual specification of functional zones (receiving, storage, etc.)
   - Provides 3D bounding boxes for all infrastructure

2. **Geometric Assignment**: Objects assigned via containment checks (not learned)
   - Isolates hierarchy contribution (no confounding perception errors)
   - Guaranteed correct assignment (point-in-bbox is deterministic)
   - Clean baseline comparison (OpenGraph and TierGraph get identical detections)

3. **5-Level Hierarchy**: Explicit warehouse structure
   - Zone: Functional areas (storage, receiving, staging)
   - Aisle: Corridors between shelves
   - Shelf: Shelving units
   - Section: Tiers/levels on shelves (height-based)
   - Object: Detected items from OpenGraph

### Evaluation Metrics

**Baseline (OpenGraph)**: Run `build_scenegraph.py` on same data
**TierGraph (Yours)**: Run `build_tiergraph.py` on same data

Compare:
1. **Query efficiency**: Time to find objects in specific locations
2. **Structure**: Flat MST vs 5-level hierarchy (node/edge counts)
3. **Semantic accuracy**: Does hierarchy match ground truth warehouse layout?
4. **Scalability**: How does each approach handle more objects/sequences?

---

## Troubleshooting

### "Object not in any zone"
- Object is outside warehouse bounds (e.g., on floor outside storage area)
- Check object position: Is it within any zone's bounding box?
- Solution: Objects outside zones are not assigned (expected behavior)

### "ModuleNotFoundError: warehouse_graph_builder"
- Make sure you're in the OpenGraph directory: `cd /home/awang/Documents/TRAILbot/OpenGraph`
- Verify the file exists: `ls utils/warehouse_graph_builder.py`

### "FileNotFoundError: warehouse_layout.json"
- Hardcoded path in `build_tiergraph.py` line 71
- Verify file exists: `ls /home/awang/Documents/TRAILbot/Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json`

### No objects assigned to sections
- Check Z coordinate (height) of objects
- Sections have specific Z bounds for tiers (bottom, middle, top)
- Objects on floor may not reach section bounds (assigned to shelf instead)

---

## Next Steps

### 1. Run TierGraph on All Sequences
```bash
for seq in 01 06 07 08; do
    python script/build_tiergraph.py --config-name=isaac_warehouse sequence=$seq
done
```

### 2. Evaluate vs OpenGraph Baseline
- Run OpenGraph's `build_scenegraph.py` on same data
- Compare node/edge counts, structure, query performance
- Document differences for thesis

### 3. Implement Query Interface
Create `query_tiergraph.py` to demonstrate hierarchical queries:
- "Find all objects in Storage Zone"
- "Find objects in Aisle 2, Shelf 3"
- "Find objects in Section 0 (bottom tier)"

### 4. Visualization (✓ COMPLETE)
Two visualization scripts are provided:

**3D Interactive** (`visualize_tiergraph.py`):
- Shows warehouse infrastructure as colored bounding boxes
- Objects colored by instance or hierarchy level
- Toggle infrastructure layers with keyboard
- View parent→child hierarchy edges

**2D Tree Diagram** (`visualize_hierarchy_tree.py`):
- Shows logical hierarchy structure
- Color-coded by level (zone, aisle, shelf, section, object)
- Saves PNG for thesis figures

---

## Citation

If you use this code, please cite:

```
@thesis{wang2026tiergraph,
  title={TierGraph: Hierarchical Scene Graphs for Warehouse Robot Navigation},
  author={Wang, [Your Name]},
  year={2026},
  school={[Your University]}
}
```

---

## Contact

- Author: awang
- Project: TierGraph (Thesis)
- Advisor: [Advisor Name]
- Date: February 2026

---

**Status**: ✓ Implementation complete, ready for evaluation
