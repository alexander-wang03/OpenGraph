# Warehouse Layout Generation Pipeline

This directory contains tools for generating and visualizing the warehouse layout from Isaac Sim export data.

## Files

### Scripts
- **`create_warehouse_layout.py`** - Parses `object_lists.csv` from Isaac Sim to generate hierarchical warehouse layout JSON
- **`visualize_warehouse_layout_3d.py`** - Creates 3D visualizations of the warehouse layout (matplotlib-based)

### Data
Located in `../data/warehouse_layout/`:
- **`object_lists.csv`** - Raw Isaac Sim object export (2.3 MB, 7000+ objects)
- **`warehouse_layout.json`** - Generated hierarchical warehouse layout (154 KB)

## Warehouse Layout Hierarchy

The layout JSON contains a 5-level hierarchy:
```
Floor
└── Functional Zones (Receiving, Packing, Storage, Pallet Truck, Hub Robot, Forklift, General)
    ├── Aisles (corridors between shelves)
    │   └── Shelves (racking units)
    │       └── Sections (vertical tiers: bottom, intermediate, top)
    └── Shelves (directly in zones without aisles)
        └── Sections
```

## Usage

### Generate warehouse_layout.json

```bash
cd /home/awang/Documents/TRAILbot/OpenGraph/warehouse_layout

# Generate layout from CSV
python create_warehouse_layout.py \
    --csv ../data/warehouse_layout/object_lists.csv \
    --output ../data/warehouse_layout/warehouse_layout.json
```

### Visualize the layout

```bash
# Single 3D view
python visualize_warehouse_layout_3d.py \
    --json ../data/warehouse_layout/warehouse_layout.json \
    --output ../data/warehouse_layout/warehouse_layout_3d.png

# Multi-view (3D + top + side views)
python visualize_warehouse_layout_3d.py \
    --json ../data/warehouse_layout/warehouse_layout.json \
    --output ../data/warehouse_layout/warehouse_layout_multiview.png \
    --multi-view

# Interactive view (won't save, just display)
python visualize_warehouse_layout_3d.py \
    --json ../data/warehouse_layout/warehouse_layout.json \
    --show
```

## Coordinate System

All coordinates are in **Isaac Sim global frame**:
- **X**: Forward (robot front)
- **Y**: Left (robot left side)
- **Z**: Up (vertical)
- **Units**: meters

## Integration with TierGraph

The warehouse layout is used by the TierGraph pipeline:
1. **Coordinate alignment** (`utils/coordinate_alignment.py`) transforms the layout from global to robot-relative frame
2. **TierGraph builder** (`utils/warehouse_graph_builder.py`) loads the aligned layout and assigns detected objects to the hierarchy
3. **Visualization** (`script/visualize_tiergraph.py`) displays objects colored by their hierarchical assignment

## Layout Statistics

From `warehouse_layout.json`:
- Floor area: ~1200 m²
- Functional zones: 7
- Aisles: 6
- Shelves: 7
- Shelf sections: 118
- Floor tiles: 160
- Rack shields: 56
- Rack shelves: 42

## Dependencies

Required packages:
- `numpy` - Array operations
- `matplotlib` - 2D/3D visualization
- `shapely` (optional) - Polygon operations for zone boundaries

Install with:
```bash
pip install numpy matplotlib shapely
```
