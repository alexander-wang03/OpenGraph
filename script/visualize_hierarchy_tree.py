#!/usr/bin/env python3
"""
Visualize TierGraph Hierarchy as a Tree

Creates a 2D hierarchical tree diagram showing the warehouse structure:
Zone → Aisle → Shelf → Section → Object

This provides a complementary view to the 3D visualization, showing
the logical hierarchy structure clearly.

Usage:
    python script/visualize_hierarchy_tree.py --config-name=isaac_warehouse sequence=01

Output:
    Saves PNG file: results/warehouse_{sequence}/tiergraph_hierarchy.png

Author: awang (TierGraph thesis)
Date: 2026-02-10
"""

import sys
sys.path.append("/home/awang/Documents/TRAILbot/OpenGraph")

import json
import hydra
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig


def load_tiergraph(graph_path):
    """Load TierGraph JSON."""
    with open(graph_path, 'r') as f:
        return json.load(f)


def build_networkx_graph(tiergraph):
    """Convert TierGraph to NetworkX directed graph."""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in tiergraph['nodes']:
        G.add_node(node['id'], **{
            'type': node['type'],
            'name': node['name'],
        })

    # Add edges
    for edge in tiergraph['edges']:
        G.add_edge(edge['source'], edge['target'])

    return G


def get_node_colors(G):
    """Assign colors by hierarchy level."""
    colors_map = {
        'zone': '#FF6B6B',      # Red
        'aisle': '#4ECDC4',     # Teal
        'shelf': '#45B7D1',     # Blue
        'section': '#FFA07A',   # Orange
        'object': '#98D8C8',    # Light green
    }

    colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        colors.append(colors_map.get(node_type, '#CCCCCC'))

    return colors


def get_node_sizes(G):
    """Assign node sizes by hierarchy level."""
    sizes_map = {
        'zone': 3000,
        'aisle': 2000,
        'shelf': 1500,
        'section': 800,
        'object': 400,
    }

    sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        sizes.append(sizes_map.get(node_type, 500))

    return sizes


def hierarchical_layout(G):
    """
    Create a hierarchical layout for the tree.
    Levels: Zone (0) → Aisle (1) → Shelf (2) → Section (3) → Object (4)
    """
    pos = {}
    levels = {node: G.nodes[node].get('type', 'unknown') for node in G.nodes()}

    level_order = ['zone', 'aisle', 'shelf', 'section', 'object']
    level_y = {level: 5 - i for i, level in enumerate(level_order)}

    # Group nodes by level
    nodes_by_level = {level: [] for level in level_order}
    for node, node_type in levels.items():
        if node_type in nodes_by_level:
            nodes_by_level[node_type].append(node)

    # Position nodes
    for level, nodes in nodes_by_level.items():
        y = level_y[level]
        n = len(nodes)
        if n == 0:
            continue

        # Spread nodes horizontally
        x_spacing = 20.0 / max(n, 1)
        x_start = -10.0

        for i, node in enumerate(sorted(nodes)):
            x = x_start + i * x_spacing
            pos[node] = (x, y)

    return pos


@hydra.main(version_base=None, config_path="../config", config_name="isaac_warehouse")
def main(cfg: DictConfig):
    print("\n" + "="*60)
    print("TIERGRAPH HIERARCHY TREE VISUALIZATION")
    print("="*60 + "\n")

    # Load TierGraph
    graph_path = Path(cfg.scenegraph_path)
    print(f"Loading TierGraph from {graph_path}...")
    tiergraph = load_tiergraph(graph_path)

    stats = tiergraph['statistics']
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Zones: {stats['nodes_by_type']['zone']}")
    print(f"  Aisles: {stats['nodes_by_type']['aisle']}")
    print(f"  Shelves: {stats['nodes_by_type']['shelf']}")
    print(f"  Sections: {stats['nodes_by_type']['section']}")
    print(f"  Objects: {stats['nodes_by_type']['object']}")

    # Build NetworkX graph
    print("\nBuilding hierarchy tree...")
    G = build_networkx_graph(tiergraph)

    # Create layout
    print("Computing layout...")
    pos = hierarchical_layout(G)
    colors = get_node_colors(G)
    sizes = get_node_sizes(G)

    # Draw graph
    print("Rendering visualization...")
    plt.figure(figsize=(24, 14))

    # Draw edges first (so nodes are on top)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#888888',
        width=0.5,
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        arrowstyle='->',
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=sizes,
        alpha=0.9,
        linewidths=2,
        edgecolors='black',
    )

    # Draw labels (only for infrastructure, not all objects)
    labels = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        node_name = G.nodes[node].get('name', node)

        # Only label infrastructure (not objects to avoid clutter)
        if node_type in ['zone', 'aisle', 'shelf']:
            # Shorten long names
            if len(node_name) > 20:
                node_name = node_name[:17] + '...'
            labels[node] = node_name

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        font_color='black',
    )

    # Add title and legend
    plt.title(f'TierGraph Hierarchy - Sequence {cfg.sequence}', fontsize=20, fontweight='bold')

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                   markersize=15, label='Zone', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4',
                   markersize=13, label='Aisle', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1',
                   markersize=11, label='Shelf', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA07A',
                   markersize=9, label='Section', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#98D8C8',
                   markersize=7, label='Object', markeredgecolor='black', markeredgewidth=2),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.axis('off')
    plt.tight_layout()

    # Save figure
    output_dir = Path(cfg.scenegraph_path).parent
    output_path = output_dir / 'tiergraph_hierarchy.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Hierarchy tree saved to: {output_path}")

    # Also show interactively
    plt.show()

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
