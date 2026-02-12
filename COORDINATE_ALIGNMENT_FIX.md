# Coordinate Alignment Fix for TierGraph Visualization

## Problem

The warehouse layout (`warehouse_layout.json`) is in **Isaac Sim global coordinates**, but the OpenGraph point clouds are in **robot-relative coordinates** (first frame at origin). This causes misalignment in visualization.

## Root Cause

`collect_isaac_data.py` applies this transform:
```python
relative_pose = first_pose_inv @ absolute_pose
```

This makes the first frame identity (0,0,0), but the warehouse layout remains in global coordinates.

## Solution

Two approaches:

### Option A: Re-collect Data (Recommended)

The updated `collect_isaac_data.py` now saves `first_pose.txt` automatically.

**Steps**:
1. Re-run data collection with the updated collector:
   ```bash
   ros2 run -- python3 /home/awang/Documents/TRAILbot/OpenGraph/script/collect_isaac_data.py \
       --sequence 09 \
       --mode depth \
       --save_hz 10
   ```

2. This creates: `data/isaac_warehouse/09/first_pose.txt`

3. Build TierGraph (it will auto-align):
   ```bash
   python script/build_tiergraph.py --config-name=isaac_warehouse sequence=09
   python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=09
   ```

### Option B: Fix Existing Data (Quickest)

If you don't want to re-collect, create `first_pose.txt` manually.

**Steps**:

1. **Find your robot's starting position** in Isaac Sim:
   - Option 1: Check Isaac Sim UI for robot's initial pose
   - Option 2: Look at ROS `/tf` topic at the start of your recording
   - Option 3: Estimate from the warehouse map

2. **Create first_pose.txt**:
   ```bash
   # Example: Robot started at (x=-10, y=5, z=0, yaw=1.57 rad)
   python script/create_first_pose_from_rosbag.py \
       --sequence 01 \
       --manual \
       --x -10.0 \
       --y 5.0 \
       --z 0.0 \
       --yaw 1.57
   ```

3. **Rebuild TierGraph**:
   ```bash
   python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01
   python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01
   ```

## How It Works

1. **Data Collection** (`collect_isaac_data.py`):
   - Saves relative poses (first frame = identity)
   - ALSO saves `first_pose.txt` (absolute first pose in Isaac Sim coordinates)

2. **TierGraph Building** (`build_tiergraph.py`):
   - Loads `first_pose.txt`
   - Transforms warehouse layout: `global_coords @ inv(first_pose) = robot_coords`
   - Saves aligned layout to `warehouse_layout_aligned.json`
   - Builds hierarchy using aligned coordinates

3. **Visualization** (`visualize_tiergraph.py`):
   - Loads TierGraph (already in robot coordinates)
   - Loads OpenGraph point clouds (in robot coordinates)
   - Everything aligns!

## Files Modified

| File | Change |
|------|--------|
| `script/collect_isaac_data.py` | Now saves `first_pose.txt` |
| `utils/coordinate_alignment.py` | NEW: Transform utilities |
| `script/build_tiergraph.py` | Uses aligned warehouse layout |
| `script/test_coordinate_alignment.py` | NEW: Test alignment |
| `script/create_first_pose_from_rosbag.py` | NEW: Manual pose creation |

## Testing

**Test alignment without re-collecting**:
```bash
python script/test_coordinate_alignment.py
```

This will warn you if `first_pose.txt` is missing.

## Debugging

**Check if first_pose.txt exists**:
```bash
ls data/isaac_warehouse/01/first_pose.txt
```

**View first_pose.txt**:
```bash
cat data/isaac_warehouse/01/first_pose.txt
# Should show 12 numbers (3x4 matrix flattened)
```

**Check aligned layout**:
```bash
# After building TierGraph:
ls data/isaac_warehouse/01/warehouse_layout_aligned.json
```

**Compare global vs aligned**:
```python
import json

# Global (Isaac Sim)
global_layout = json.load(open('Isaac-sim-husky-navigation/src/isaaccomponentspython/Data/warehouse_layout.json'))
storage = next(z for z in global_layout['functional_zones'] if z['id'] == 'zone_storage')
print("Global:", storage['bounds']['center'])

# Aligned (robot-relative)
aligned_layout = json.load(open('OpenGraph/data/isaac_warehouse/01/warehouse_layout_aligned.json'))
storage = next(z for z in aligned_layout['functional_zones'] if z['id'] == 'zone_storage')
print("Aligned:", storage['bounds']['center'])
```

## Quick Reference

| Task | Command |
|------|---------|
| **Test alignment** | `python script/test_coordinate_alignment.py` |
| **Create first_pose.txt** | `python script/create_first_pose_from_rosbag.py --sequence 01 --manual --x X --y Y --yaw YAW` |
| **Rebuild TierGraph** | `python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01` |
| **Visualize** | `python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01` |

## For Your Sequences

You have 4 existing sequences (01, 06, 07, 08). To fix them:

```bash
# For each sequence, create first_pose.txt
# (You need to find the starting position for each)

for seq in 01 06 07 08; do
    # TODO: Replace X, Y, YAW with actual values
    python script/create_first_pose_from_rosbag.py \
        --sequence $seq \
        --manual \
        --x X \
        --y Y \
        --yaw YAW

    # Rebuild TierGraph with alignment
    python script/build_tiergraph.py --config-name=isaac_warehouse sequence=$seq
done
```

## Alternative: Skip Alignment (For Testing Only)

If you just want to see TierGraph structure without correct spatial alignment:

1. The hierarchy building still works (objects get assigned to zones/aisles/shelves)
2. Just the 3D positions won't match perfectly
3. Good enough for demonstrating the concept, but not for thesis figures

---

**Status**: ✅ Fix implemented, awaiting first_pose.txt creation or re-collection
