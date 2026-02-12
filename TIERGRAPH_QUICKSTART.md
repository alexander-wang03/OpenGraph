# TierGraph Quick Start Guide

**Goal**: Build and visualize hierarchical warehouse scene graphs from Isaac Sim data.

---

## Prerequisites

✓ Isaac Sim warehouse data collected (sequences 01, 06, 07, 08)
✓ OpenGraph environment activated: `conda activate opengraph-cu118`
✓ In OpenGraph directory: `cd /home/awang/Documents/TRAILbot/OpenGraph`

---

## Option 1: Test with Synthetic Data (No Real Data Required)

```bash
# Test TierGraph implementation with synthetic objects
python script/test_tiergraph.py
```

**Output**: Shows that hierarchy building works correctly.

---

## Option 2: Full Pipeline (Real Data)

### Step 1: Generate Captions & Segmentation
```bash
python script/main_gen_cap.py --config-name=isaac_warehouse sequence=01
```
⏱️ Time: ~30 min for 240 frames
📁 Output: `results/warehouse_01/caption/`, `results/warehouse_01/vis/`

### Step 2: 3D Point Cloud Mapping
```bash
torchrun --nproc_per_node=1 script/main_gen_pc.py --config-name=isaac_warehouse sequence=01
```
⏱️ Time: ~20 min
📁 Output: `results/warehouse_01/pcd/full_pcd.pkl.gz`

### Step 3: Build TierGraph (Your Contribution!)
```bash
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01
```
⏱️ Time: ~10 seconds
📁 Output: `results/warehouse_01/pcd/object_relations.json`

### Step 4a: 3D Visualization
```bash
python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01
```

**Interactive Controls**:
- `[1]` - Toggle zones (red boxes)
- `[2]` - Toggle aisles (green boxes)
- `[3]` - Toggle shelves (blue boxes)
- `[4]` - Toggle sections (yellow boxes)
- `[5]` - Toggle hierarchy edges
- `[6]` - Color by hierarchy level
- `[I]` - Color by instance
- `[R]` - Original RGB colors
- `[Q]` - Quit

### Step 4b: 2D Tree Diagram
```bash
python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=01
```
📁 Output: `results/warehouse_01/pcd/tiergraph_hierarchy.png`

---

## Option 3: Build All Sequences (For Evaluation)

```bash
# Process all sequences
for seq in 01 06 07 08; do
    echo "=== Processing sequence $seq ==="

    # If not already processed, run captions + mapping:
    # python script/main_gen_cap.py --config-name=isaac_warehouse sequence=$seq
    # torchrun --nproc_per_node=1 script/main_gen_pc.py --config-name=isaac_warehouse sequence=$seq

    # Build TierGraph
    python script/build_tiergraph.py --config-name=isaac_warehouse sequence=$seq

    # Generate tree diagram
    python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=$seq
done
```

---

## Compare: OpenGraph vs TierGraph

### OpenGraph Baseline (Flat MST):
```bash
python script/build_scenegraph.py --config-name=isaac_warehouse sequence=01
python script/visualize.py --config-name=isaac_warehouse sequence=01
```

### TierGraph (Your Thesis):
```bash
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01
python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01
```

**Key Differences**:
| Aspect | OpenGraph | TierGraph |
|--------|-----------|-----------|
| Structure | Flat MST | 5-level hierarchy |
| Nodes | ~100 objects | ~250 (infrastructure + objects) |
| Edges | ~99 (proximity) | ~240 (containment) |
| Semantics | Generic | Warehouse-specific |
| Queries | Traversal | Direct lookup |

---

## Troubleshooting

### "CUDA out of memory"
- Already fixed: `caption_merge_ft: False` in config
- If still happening: Reduce batch size or use smaller sequences

### "FileNotFoundError: full_pcd.pkl.gz"
- Run steps 1-2 first (caption generation + 3D mapping)
- Make sure you're using the correct sequence number

### "Object not in any zone"
- Expected behavior for objects outside warehouse bounds
- Check object position in output logs

### Visualization shows empty scene
- Make sure OpenGraph results exist: `ls results/warehouse_01/pcd/full_pcd.pkl.gz`
- Make sure TierGraph was built: `ls results/warehouse_01/pcd/object_relations.json`

### Sections not visible in 3D view
- Press `[4]` to toggle sections (118 sections = lots of boxes!)
- They're hidden by default to avoid clutter

---

## For Your Thesis

### Figures to Include:
1. **2D Hierarchy Tree** (`tiergraph_hierarchy.png`) - Shows full 5-level structure
2. **3D Warehouse View** (screenshot from `visualize_tiergraph.py`) - Infrastructure + objects
3. **Comparison Table** (OpenGraph vs TierGraph stats from terminal output)

### Evaluation Metrics:
```bash
# After building both OpenGraph and TierGraph on same sequence:
# 1. Node/edge counts (in terminal output)
# 2. Query performance (implement query_tiergraph.py)
# 3. Semantic accuracy (manual inspection of assignments)
```

### Commands for All Evaluation Data:
```bash
# Build TierGraph for all sequences
for seq in 01 06 07 08; do
    python script/build_tiergraph.py --config-name=isaac_warehouse sequence=$seq > tiergraph_$seq.log
done

# Build OpenGraph baseline for comparison
for seq in 01 06 07 08; do
    python script/build_scenegraph.py --config-name=isaac_warehouse sequence=$seq > opengraph_$seq.log
done

# Extract statistics from logs for thesis table
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Test | `python script/test_tiergraph.py` |
| Build | `python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01` |
| 3D View | `python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01` |
| 2D Tree | `python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=01` |
| Baseline | `python script/build_scenegraph.py --config-name=isaac_warehouse sequence=01` |

---

**Status**: ✅ Implementation complete, ready for thesis evaluation!
