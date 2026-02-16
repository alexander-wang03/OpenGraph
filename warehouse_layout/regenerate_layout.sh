#!/bin/bash
# Regenerate warehouse_layout.json from object_lists.csv
# This is useful if you need to modify the layout generation logic

set -e

cd "$(dirname "$0")"

echo "Regenerating warehouse layout..."
python create_warehouse_layout.py \
    --csv ../data/warehouse_layout/object_lists.csv \
    --output ../data/warehouse_layout/warehouse_layout.json

echo ""
echo "Done! Generated warehouse_layout.json"
echo "Location: ../data/warehouse_layout/warehouse_layout.json"
echo ""
echo "To visualize the layout, run:"
echo "  python visualize_warehouse_layout_3d.py --show"
