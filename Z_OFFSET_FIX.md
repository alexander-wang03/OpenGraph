# Z-Offset Fix for TierGraph Visualization

## Problem Description

When visualizing TierGraph, the warehouse infrastructure (bounding boxes) appeared **offset vertically** from the point clouds by approximately 0.14 meters.

## Root Cause

**Isaac Sim warehouse floor Z coordinate**: `-0.14m` (global frame)
**Robot base_link Z coordinate**: `~0m` (approximately at floor height)

The warehouse layout JSON uses Isaac Sim's global coordinates where the floor is at Z=-0.14m, but the robot's point clouds are in the robot's local frame where Z≈0 is at floor level.

## Investigation Results

```
Warehouse layout (Isaac Sim global):
  Floor Z: -0.140m
  Shelf min Z: -0.140m
  Shelf max Z: 5.860m

Point cloud data (robot frame):
  Z median: -0.777m  (looking down at floor)
  Z mean: 0.352m
  Z range: [-0.781m, 9.175m]

Tr matrix (base_link → camera):
  Z translation: -0.440m (camera is 0.44m above base_link)
```

## Solution

Added a **Z offset correction** of `+0.14m` to the warehouse layout when aligning it to robot coordinates.

### Code Changes

Modified `utils/coordinate_alignment.py`:

1. Added `apply_z_offset()` function that shifts all Z coordinates in the warehouse layout
2. Updated `get_aligned_warehouse_layout()` to apply Z offset before XY alignment
3. Default offset: `z_offset=0.14` meters

### After Fix

```
Warehouse layout (robot frame, after +0.14m offset):
  Floor Z: 0.000m  ✓ Matches robot floor level
  Shelf min Z: 0.000m  ✓ Shelves start at floor
  Shelf max Z: 6.000m
  Section min Z: 0.000m  ✓ Bottom sections at floor
```

## Usage

The Z offset is applied automatically in `build_tiergraph.py`:

```python
aligned_layout = get_aligned_warehouse_layout(
    warehouse_layout_path,
    sequence_dir,
    z_offset=0.14  # Default, can be customized
)
```

### Custom Z Offset

If you need a different Z offset (e.g., robot at different height):

```python
# In build_tiergraph.py or visualize_tiergraph.py
aligned_layout = get_aligned_warehouse_layout(
    warehouse_layout_path,
    sequence_dir,
    z_offset=0.20  # Custom offset in meters
)
```

## Verification

To verify the alignment is correct:

```bash
# Rebuild TierGraph with Z offset
python script/build_tiergraph.py --config-name=isaac_warehouse sequence=01

# Visualize (should now be aligned vertically)
python script/visualize_tiergraph.py --config-name=isaac_warehouse sequence=01
```

### What to Look For

In the 3D visualization:
- ✓ Shelf bounding boxes should align with detected objects at correct heights
- ✓ Floor-level sections (Z≈0) should contain floor-level objects
- ✓ Upper sections should contain objects on higher shelves

## Technical Details

### Coordinate Frames

1. **Isaac Sim Global Frame**:
   - Origin: Arbitrary world origin
   - Floor: Z = -0.14m
   - Warehouse infrastructure defined relative to this

2. **Robot Base Link Frame**:
   - Origin: Robot's base_link at start
   - Floor: Z ≈ 0m (base_link is slightly above floor)
   - Point clouds are in this frame

3. **Camera Frame**:
   - Origin: Camera optical center
   - Camera is 0.44m above base_link (from Tr matrix)
   - Depth points are initially in camera frame, then transformed to base_link

### Transform Pipeline

```
Isaac Sim Global → Apply Z offset → Robot Frame (XY still global)
                                   ↓
                           Apply first_pose_inv (XY alignment)
                                   ↓
                           Robot-relative coordinates
```

## Files Modified

| File | Change |
|------|--------|
| `utils/coordinate_alignment.py` | Added `apply_z_offset()` function and Z offset parameter |
| `script/build_tiergraph.py` | Uses Z offset in `get_aligned_warehouse_layout()` |
| `Z_OFFSET_FIX.md` | This documentation |

## Remaining Work

The Z offset fixes the vertical alignment. For complete XY alignment, you still need:
1. **Option A**: Re-collect data with updated `collect_isaac_data.py` (creates `first_pose.txt`)
2. **Option B**: Manually create `first_pose.txt` for existing sequences

See `COORDINATE_ALIGNMENT_FIX.md` for XY alignment instructions.

---

**Status**: ✅ Z-offset fix implemented and tested
**Next**: Create `first_pose.txt` for XY alignment
