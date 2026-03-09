#!/usr/bin/env python3
"""
Visualize TierGraph Hierarchy as an Interactive Tree

Creates an interactive 2D hierarchical tree diagram showing the warehouse structure.

Actual hierarchy (from warehouse_graph_builder.py):

  Non-storage zones (receiving, staging, forklift, hub_robot, general):
    Zone → Object   (direct — these are open floor areas, no aisles or shelves)

  Storage zone only:
    Infrastructure:  Zone → Aisle          (corridors between shelves)
                     Zone → Shelf → Section

    Object assignment (4 cases, storage zone only):
    1. On shelf with section → Zone → Shelf → Section → Object (full depth)
    2. On shelf, no section  → Zone → Shelf → Object
    3. In aisle walkway      → Zone → Aisle → Object
    4. Unmatched (fallback)  → Zone → Object

Features:
- Interactive zoom/pan to handle dense graphs
- Hover tooltips showing object types and captions
- Clickable nodes with detailed information
- Better layout to avoid overlapping nodes

Usage:
    python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=02

Output:
    - Interactive HTML: results/warehouse_{sequence}/tiergraph_hierarchy_interactive.html
    - Static PNG: results/warehouse_{sequence}/tiergraph_hierarchy.png

Author: awang (TierGraph thesis)
Date: 2026-02-16
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import json
import hydra
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from omegaconf import DictConfig
from collections import defaultdict


def load_tiergraph(graph_path):
    """Load TierGraph JSON."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def load_object_captions(result_path, num_objects):
    """
    Load object captions directly from OpenGraph results.

    Args:
        result_path: Path to full_pcd.pkl.gz
        num_objects: Number of objects expected

    Returns:
        dict: {object_id: caption_text}
    """
    import gzip
    import pickle
    import sys

    # Need to add OpenGraph path for imports
    sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")
    from some_class.map_calss import MapObjectList

    print(f"Loading captions from OpenGraph results: {result_path}...")

    try:
        with gzip.open(result_path, 'rb') as f:
            results = pickle.load(f)

        objects = MapObjectList()
        objects.load_serializable(results['objects'])

        print(f"  Loaded {len(objects)} objects from OpenGraph")

        # Extract captions from objects
        object_captions = {}
        objects_with_captions = 0

        for i, obj in enumerate(objects):
            obj_id = f"object_{i}"

            # Try to get caption from object data
            caption = None

            # Method 1: Direct caption field
            if 'caption' in obj:
                caption = obj['caption']
            # Method 2: Captions list
            elif 'captions' in obj and obj['captions']:
                caption = obj['captions'][0]
            # Method 3: Class name
            elif 'class' in obj:
                caption = obj['class']
            # Method 4: Check object data dict
            elif hasattr(obj, 'get'):
                for key in ['caption', 'captions', 'class', 'label', 'name']:
                    val = obj.get(key)
                    if val:
                        if isinstance(val, list) and len(val) > 0:
                            caption = val[0]
                        else:
                            caption = val
                        break

            if caption and isinstance(caption, str) and caption.strip():
                object_captions[obj_id] = caption.strip()
                objects_with_captions += 1
            else:
                # Use generic label
                object_captions[obj_id] = f"Object {i}"

        print(f"  Found captions for {objects_with_captions}/{len(objects)} objects")

        if objects_with_captions == 0:
            print("  WARNING: No captions found in OpenGraph results!")
            print("  This likely means main_gen_pc.py didn't merge captions properly.")
            print("  Objects will be labeled as 'Object #' in the visualization.")

        return object_captions

    except Exception as e:
        print(f"  Error loading captions: {e}")
        print(f"  Will use generic labels instead.")

        # Fallback to generic labels
        return {f"object_{i}": f"Object {i}" for i in range(num_objects)}


def build_hierarchy_stats(tiergraph):
    """
    Analyze hierarchy structure to understand zone-specific patterns.

    Non-storage zones always produce Zone → Object (direct_objects).
    The aisle_objects / shelf_objects / section_objects counters are
    meaningful only for zone_storage.

    Returns dict with statistics about each zone's hierarchy depth and structure.
    """
    nodes_dict = {node['id']: node for node in tiergraph['nodes']}
    stats = defaultdict(lambda: {
        'total_objects': 0,
        'direct_objects': 0,     # Zone → Object (non-storage zones always; storage fallback)
        'aisle_objects': 0,      # Zone → Aisle → Object  (storage zone only)
        'shelf_objects': 0,      # Zone → Aisle → Shelf → Object  (storage zone only)
        'section_objects': 0,    # Zone → Aisle → Shelf → Section → Object  (storage zone only)
        'aisles': set(),
        'shelves': set(),
        'sections': set()
    })

    for node in tiergraph['nodes']:
        if node['type'] == 'object':
            # Trace back to find zone and path
            path = []
            current = node
            while current:
                path.append(current['type'])
                parent_id = current.get('parent_id')
                if not parent_id:
                    break
                current = nodes_dict.get(parent_id)

            path = path[::-1]  # Reverse path

            # Find zone
            zone_id = None
            current = node
            while current:
                if current['type'] == 'zone':
                    zone_id = current['id']
                    break
                parent_id = current.get('parent_id')
                if not parent_id:
                    break
                current = nodes_dict.get(parent_id)

            if not zone_id:
                continue

            stats[zone_id]['total_objects'] += 1

            # Classify hierarchy depth.
            #
            # Non-storage zones:
            #   zone → object                      (always direct)
            #
            # Storage zone:
            #   zone → shelf → section → object    (full path — shelf matched)
            #   zone → shelf → object              (shelf matched, no section)
            #   zone → aisle → object              (in aisle walkway, no shelf match)
            #   zone → object                      (fallback — unmatched)
            if path == ['zone', 'object']:
                stats[zone_id]['direct_objects'] += 1
            elif path == ['zone', 'aisle', 'object']:
                # Storage zone only — in aisle walkway, not on a shelf
                stats[zone_id]['aisle_objects'] += 1
            elif path == ['zone', 'shelf', 'object']:
                # Storage zone only — on shelf XY area but no section match
                stats[zone_id]['shelf_objects'] += 1
            elif path == ['zone', 'shelf', 'section', 'object']:
                # Storage zone only — fully assigned on a shelf section
                stats[zone_id]['section_objects'] += 1

    return stats


def create_tree_layout(G, root_nodes, vertical_spacing=2.0, leaf_spacing=1.0):
    """
    Reingold-Tilford-style tree layout that guarantees no edge crossings.

    Leaves are placed consecutively left-to-right; each parent is centred
    over its children's x-range.  All nodes at the same BFS depth share
    the same y-coordinate so hierarchy levels are clearly aligned.
    """
    pos = {}
    leaf_counter = [0]  # list so the nested closure can mutate it

    # BFS depth so every node has a consistent y regardless of which root
    # subtree it belongs to.
    depths = {}
    for root in root_nodes:
        queue = [(root, 0)]
        visited = set()
        while queue:
            node, depth = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            depths[node] = depth
            for child in G.successors(node):
                if child not in visited:
                    queue.append((child, depth + 1))

    def assign_x(node):
        """Post-order DFS: place children first, then centre parent over them."""
        children = sorted(G.successors(node))
        if not children:
            x = leaf_counter[0] * leaf_spacing
            leaf_counter[0] += 1
        else:
            for child in children:
                assign_x(child)
            x = (pos[children[0]][0] + pos[children[-1]][0]) / 2
        pos[node] = (x, -depths[node] * vertical_spacing)

    for root in sorted(root_nodes):
        assign_x(root)
        leaf_counter[0] += 1  # small gap between separate root subtrees

    return pos


def get_node_color(node_type, zone_id=None):
    """Get color for a node based on its type and zone."""
    zone_colors = {
        'zone_storage': '#00B300',      # Green
        'zone_receiving': '#E68000',    # Orange
        'zone_staging': '#E600E6',      # Magenta
        'zone_pallet_truck': '#E6E600', # Yellow
        'zone_hub_robot': '#33B8E6',    # Cyan
        'zone_forklift': '#E69933',     # Brown
        'zone_general': '#808080',      # Gray
    }

    type_colors = {
        'zone': '#FF6B6B',      # Red
        'aisle': '#FFD700',     # Gold
        'shelf': '#4169E1',     # Royal Blue
        'section': '#FFA500',   # Orange
        'object': '#90EE90',    # Light Green
    }

    if node_type == 'zone' and zone_id:
        return zone_colors.get(zone_id, type_colors['zone'])

    return type_colors.get(node_type, '#CCCCCC')


def get_node_size(node_type):
    """Get size for a node based on its type."""
    sizes = {
        'zone': 30,
        'aisle': 20,
        'shelf': 20,
        'section': 15,
        'object': 10,
    }
    return sizes.get(node_type, 10)


def truncate_text(text, max_length=30):
    """Truncate text if too long."""
    if len(text) > max_length:
        return text[:max_length-3] + '...'
    return text


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "="*80)
    print("TIERGRAPH INTERACTIVE HIERARCHY TREE VISUALIZATION")
    print("="*80 + "\n")

    # Load TierGraph
    graph_path = Path(cfg.scenegraph_path)
    print(f"Loading TierGraph from {graph_path}...")
    tiergraph = load_tiergraph(graph_path)

    stats = tiergraph['statistics']

    # Load object captions from OpenGraph results
    result_path = Path(cfg.result_path)
    object_captions = load_object_captions(result_path, stats['nodes_by_type']['object'])
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Zones: {stats['nodes_by_type']['zone']}")
    print(f"  Aisles: {stats['nodes_by_type'].get('aisle', 0)}")
    print(f"  Shelves: {stats['nodes_by_type'].get('shelf', 0)}")
    print(f"  Sections: {stats['nodes_by_type'].get('section', 0)}")
    print(f"  Objects: {stats['nodes_by_type']['object']}")

    # Build NetworkX graph
    print("\nBuilding hierarchy tree...")
    G = nx.DiGraph()

    # Add nodes with attributes
    node_data = {}
    for node in tiergraph['nodes']:
        G.add_node(node['id'])

        # Use caption for objects if available
        node_name = node['name']
        if node['type'] == 'object' and node['id'] in object_captions:
            node_name = object_captions[node['id']]

        node_data[node['id']] = {
            'type': node['type'],
            'name': node_name,
            'id': node['id'],
            'original_name': node['name']  # Keep original for reference
        }

    # Add edges
    for node in tiergraph['nodes']:
        if node.get('parent_id'):
            G.add_edge(node['parent_id'], node['id'])

    # Print object captions summary
    if object_captions:
        print("\n" + "="*80)
        print("OBJECT LABELS (What TierGraph thinks each object is)")
        print("="*80)
        for obj_id in sorted(object_captions.keys(), key=lambda x: int(x.split('_')[1])):
            print(f"  {obj_id}: {object_captions[obj_id]}")
        print("="*80)

    # Analyze hierarchy structure
    print("\nAnalyzing zone-specific hierarchy patterns...")
    zone_stats = build_hierarchy_stats(tiergraph)

    for zone_id, zstats in sorted(zone_stats.items()):
        zone_name = zone_id.replace('zone_', '').title()
        is_storage = zone_id == 'zone_storage'
        print(f"\n  {zone_name} Zone:")
        print(f"    Total objects: {zstats['total_objects']}")
        if zstats['direct_objects'] > 0:
            label = "Zone → Object (fallback)" if is_storage else "Zone → Object (direct)"
            print(f"    {label}: {zstats['direct_objects']}")
        # Aisle / shelf / section breakdowns only apply to storage zone
        if is_storage:
            if zstats['aisle_objects'] > 0:
                print(f"    Zone → Aisle → Object: {zstats['aisle_objects']}")
            if zstats['shelf_objects'] > 0:
                print(f"    Zone → Aisle → Shelf → Object: {zstats['shelf_objects']}")
            if zstats['section_objects'] > 0:
                print(f"    Zone → Aisle → Shelf → Section → Object: {zstats['section_objects']}")

    # Find root nodes (zones with no parents)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    print(f"\nFound {len(root_nodes)} root zones")

    # Create hierarchical layout
    print("Computing hierarchical layout...")
    pos = create_tree_layout(G, root_nodes, vertical_spacing=2.0, leaf_spacing=1.0)

    # Compute dynamic canvas size so that visual spacing scales with node count
    MIN_PX_PER_NODE_H = 80
    MIN_PX_PER_NODE_V = 120

    # Depth (for height) via BFS
    from collections import Counter
    levels_count = Counter()
    root_set = set(root_nodes)
    queue = [(r, 0) for r in root_set]
    visited_l = set()
    while queue:
        n, lv = queue.pop(0)
        if n in visited_l:
            continue
        visited_l.add(n)
        levels_count[lv] += 1
        for ch in G.successors(n):
            if ch not in visited_l:
                queue.append((ch, lv + 1))

    max_depth = max(levels_count.keys()) + 1 if levels_count else 1
    n_leaves = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

    fig_width = max(2000, n_leaves * MIN_PX_PER_NODE_H + 300)
    fig_height = max(1200, max_depth * MIN_PX_PER_NODE_V + 300)

    # Compute explicit axis ranges from data to prevent Plotly auto-scaling
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_pad = (max(all_x) - min(all_x)) * 0.05 + 1.0
    y_pad = (max(all_y) - min(all_y)) * 0.05 + 1.0
    x_range = [min(all_x) - x_pad, max(all_x) + x_pad]
    y_range = [min(all_y) - y_pad, max(all_y) + y_pad]

    print(f"  Leaf nodes: {n_leaves} → canvas {fig_width}x{fig_height}px")

    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )

    # Create node traces (grouped by type for legend)
    node_traces = {}

    for node in G.nodes():
        node_type = node_data[node]['type']
        zone_id = node if node_type == 'zone' else None

        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [],
                'y': [],
                'text': [],
                'customdata': [],
                'color': get_node_color(node_type, zone_id),
                'size': get_node_size(node_type),
                'name': node_type.title()
            }

        x, y = pos[node]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)

        # Text label (shortened for display)
        display_name = truncate_text(node_data[node]['name'], 20)
        node_traces[node_type]['text'].append(display_name)

        # Hover information (full details)
        if node_type == 'object':
            hover_text = f"<b>🏷️ {node_data[node]['name']}</b><br>"
            hover_text += f"<i>Type: {node_type}</i><br>"
        else:
            hover_text = f"<b>{node_data[node]['name']}</b><br>"
            hover_text += f"Type: {node_type}<br>"

        hover_text += f"ID: {node}<br>"

        # Add parent/child info
        parents = list(G.predecessors(node))
        children = list(G.successors(node))
        if parents:
            hover_text += f"Parent: {parents[0]}<br>"
        if children:
            hover_text += f"Children: {len(children)}<br>"

        # Add hierarchy path for objects
        if node_type == 'object' and parents:
            path_parts = []
            current = node
            visited = set()
            while current and current not in visited:
                visited.add(current)
                path_parts.insert(0, current)
                preds = list(G.predecessors(current))
                current = preds[0] if preds else None
            hover_text += f"<br><b>Path:</b> {' → '.join(path_parts)}"

        node_traces[node_type]['customdata'].append(hover_text)

    # Create Plotly traces
    traces = edge_trace.copy()

    for node_type, data in node_traces.items():
        # Determine marker color based on zone
        if node_type == 'zone':
            # For zones, use zone-specific colors
            colors = [get_node_color('zone', node_id)
                     for node_id, nt in node_data.items()
                     if nt['type'] == 'zone']
            marker_color = colors
        else:
            marker_color = data['color']

        trace = go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            name=data['name'],
            text=data['text'],
            textposition='top center',
            textfont=dict(size=8, color='black'),
            hovertext=data['customdata'],
            hoverinfo='text',
            marker=dict(
                size=data['size'],
                color=marker_color,
                line=dict(width=2, color='black')
            )
        )
        traces.append(trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout with better defaults for large graphs
    title_text = f'TierGraph Hierarchy - Sequence {cfg.sequence}<br>'
    title_text += f'<sub>{stats["nodes_by_type"]["zone"]} Zones | '
    title_text += f'{stats["nodes_by_type"].get("aisle", 0)} Aisles | '
    title_text += f'{stats["nodes_by_type"].get("shelf", 0)} Shelves | '
    title_text += f'{stats["nodes_by_type"].get("section", 0)} Sections | '
    title_text += f'{stats["nodes_by_type"]["object"]} Objects</sub>'

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=100),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=False,
            showticklabels=False,
            title='Use mouse to zoom and pan • Double-click to reset view',
            range=x_range,
            autorange=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=False,
            showticklabels=False,
            range=y_range,
            autorange=False,
        ),
        plot_bgcolor='white',
        height=fig_height,
        width=fig_width,
        legend=dict(
            x=1.01,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11)
        )
    )

    # Add annotations explaining architecture
    fig.add_annotation(
        text=(
            "<b>Architecture:</b><br>"
            "• Non-storage zones: Zone → Object (direct — open floor, no aisles)<br>"
            "• Storage (on shelf + section): Zone → Shelf → Section → Object<br>"
            "• Storage (on shelf, no section): Zone → Shelf → Object<br>"
            "• Storage (in walkway): Zone → Aisle → Object<br>"
            "• Storage (fallback): Zone → Object<br>"
            "<i>Hover over nodes for details. Click and drag to pan. Scroll to zoom.</i>"
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        xanchor='left', yanchor='bottom',
        showarrow=False,
        bgcolor="rgba(255,255,200,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align='left'
    )

    # Save interactive HTML
    output_dir = Path(cfg.scenegraph_path).parent
    html_path = output_dir / 'tiergraph_hierarchy_interactive.html'
    fig.write_html(str(html_path))
    print(f"\n✓ Interactive HTML saved to: {html_path}")
    print(f"  Open in browser to explore interactively!")

    # Also save static PNG
    try:
        png_path = output_dir / 'tiergraph_hierarchy.png'
        fig.write_image(str(png_path), width=2400, height=1600)
        print(f"✓ Static PNG saved to: {png_path}")
    except Exception as e:
        print(f"  (Could not save PNG: {e})")
        print(f"  To enable PNG export: pip install kaleido")

    # Show in browser
    print("\nOpening visualization in browser...")
    fig.show()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
