#!/usr/bin/env python3
"""
Visualize the 3D warehouse layout JSON.
Shows floor polygon, aisles, and 3D shelf levels.
"""

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional, List
import argparse
import numpy as np


def load_warehouse_layout(json_path: Path) -> Dict:
    """Load warehouse layout JSON."""
    with json_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def extract_aisles_and_shelves(layout: Dict):
    """Extract aisles and shelves from nested zone structure."""
    aisles = []
    shelves = []
    for zone in layout.get('functional_zones', []):
        zone_aisles = zone.get('aisles', {})
        if isinstance(zone_aisles, dict):
            aisles.extend(zone_aisles.values())
        zone_shelves = zone.get('shelves', {})
        if isinstance(zone_shelves, dict):
            shelves.extend(zone_shelves.values())
    return aisles, shelves


def get_sections_list(shelf: Dict) -> List[Dict]:
    """Get flat list of sections from level-indexed dict structure."""
    sections = shelf.get('sections', {})
    if isinstance(sections, list):
        return sections
    result = []
    for level_dict in sections.values():
        if isinstance(level_dict, dict):
            result.extend(level_dict.values())
    return result


def create_box_vertices(bounds: Dict) -> np.ndarray:
    """
    Create vertices for a 3D box from bounds.
    Returns 8 vertices in order for proper face construction.
    """
    min_x, min_y, min_z = bounds['min']['x'], bounds['min']['y'], bounds['min']['z']
    max_x, max_y, max_z = bounds['max']['x'], bounds['max']['y'], bounds['max']['z']

    vertices = np.array([
        [min_x, min_y, min_z],  # 0: bottom-front-left
        [max_x, min_y, min_z],  # 1: bottom-front-right
        [max_x, max_y, min_z],  # 2: bottom-back-right
        [min_x, max_y, min_z],  # 3: bottom-back-left
        [min_x, min_y, max_z],  # 4: top-front-left
        [max_x, min_y, max_z],  # 5: top-front-right
        [max_x, max_y, max_z],  # 6: top-back-right
        [min_x, max_y, max_z],  # 7: top-back-left
    ])
    return vertices


def create_box_faces(vertices: np.ndarray) -> List[np.ndarray]:
    """Create the 6 faces of a box from 8 vertices."""
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[2], vertices[1]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[4], vertices[7], vertices[3]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    return faces


def visualize_3d_layout(
    layout: Dict,
    output_path: Optional[Path] = None,
    show_labels: bool = True,
    figsize: tuple = (16, 12),
    elev: float = 30,
    azim: float = -60
):
    """
    Create a 3D visualization of the warehouse layout.

    Shows:
    - Floor polygon (base)
    - Functional zones (colored 2D regions at floor level)
    - Aisles (2D rectangles at floor level)
    - Shelves (wireframe 3D boxes) with sections (filled 3D boxes)
    """
    floor = layout['floor']
    aisles, shelves = extract_aisles_and_shelves(layout)
    functional_zones = layout.get('functional_zones', [])
    metadata = layout['metadata']
    floor_bounds = floor['bounds']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Color schemes for shelf sections
    colors_horizontal = plt.cm.Blues(np.linspace(0.3, 0.8, 10))
    colors_vertical = plt.cm.Reds(np.linspace(0.3, 0.8, 10))

    # Zone color mapping
    zone_colors = {
        'receiving': ('sandybrown', 'saddlebrown'),
        'packing': ('plum', 'purple'),
        'pallet_truck': ('lightyellow', 'goldenrod'),
        'hub_robot': ('lightcyan', 'darkcyan'),
        'forklift': ('peachpuff', 'sienna'),
        'storage': ('lightskyblue', 'steelblue'),
        'general': ('palegreen', 'darkgreen')
    }

    # Draw floor polygon at Z = floor level
    floor_vertices = floor['vertices']
    floor_z = floor_bounds['min']['z']
    floor_3d = [[v[0], v[1], floor_z] for v in floor_vertices]
    floor_poly = Poly3DCollection([floor_3d], alpha=0.15, facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_collection3d(floor_poly)

    # Draw functional zones as colored polygons at floor level
    for zone in functional_zones:
        zone_verts = zone.get('vertices', [])
        if not zone_verts:
            continue
        facecolor, edgecolor = zone_colors.get(zone['type'], ('lightyellow', 'gray'))
        zone_3d = [[v[0], v[1], floor_z + 0.01] for v in zone_verts]
        zone_poly = Poly3DCollection(
            [zone_3d], alpha=0.25, facecolor=facecolor,
            edgecolor=edgecolor, linewidth=1.5
        )
        ax.add_collection3d(zone_poly)

        if show_labels:
            zb = zone['bounds']
            ax.text(zb['center']['x'], zb['center']['y'], floor_z + 0.15,
                   zone['name'], fontsize=8, color=edgecolor,
                   weight='bold', ha='center')

    # Draw aisles as thin 3D boxes at floor level
    for aisle in aisles:
        bounds = aisle['bounds']
        aisle_bounds_3d = {
            'min': {
                'x': bounds['min']['x'],
                'y': bounds['min']['y'],
                'z': floor_z
            },
            'max': {
                'x': bounds['max']['x'],
                'y': bounds['max']['y'],
                'z': floor_z + 0.1
            }
        }
        vertices = create_box_vertices(aisle_bounds_3d)
        faces = create_box_faces(vertices)
        aisle_poly = Poly3DCollection(faces, alpha=0.3, facecolor='lightyellow', edgecolor='darkgreen', linewidth=1)
        ax.add_collection3d(aisle_poly)

        if show_labels:
            center = bounds['center']
            ax.text(center['x'], center['y'], floor_z + 0.2, aisle['id'],
                   fontsize=7, color='darkgreen', weight='bold', ha='center')

    # Draw shelves and their sections
    for shelf in shelves:
        orientation = shelf['orientation']

        # Draw shelf outline as wireframe box
        shelf_vertices = create_box_vertices(shelf['bounds'])
        shelf_faces = create_box_faces(shelf_vertices)
        shelf_wireframe = Poly3DCollection(
            shelf_faces, alpha=0.03, facecolor='gray',
            edgecolor='darkgray', linewidth=0.8, linestyle='--'
        )
        ax.add_collection3d(shelf_wireframe)

        # Draw each shelf section as a colored filled box
        for section in get_sections_list(shelf):
            bounds = section['bounds']
            level_index = section['level_index']

            if orientation == 'horizontal':
                color = colors_horizontal[min(level_index, 9)]
            else:
                color = colors_vertical[min(level_index, 9)]

            vertices = create_box_vertices(bounds)
            faces = create_box_faces(vertices)
            section_poly = Poly3DCollection(
                faces, alpha=0.5, facecolor=color,
                edgecolor='black', linewidth=1
            )
            ax.add_collection3d(section_poly)

            if show_labels:
                center = bounds['center']
                label = f"L{level_index}"
                ax.text(center['x'], center['y'], center['z'], label,
                       fontsize=5, color='navy', ha='center', va='center')

    # Set axis limits
    padding = 3.0
    ax.set_xlim(floor_bounds['min']['x'] - padding, floor_bounds['max']['x'] + padding)
    ax.set_ylim(floor_bounds['min']['y'] - padding, floor_bounds['max']['y'] + padding)

    # Z limits based on shelf heights
    if shelves:
        max_z = max(s['bounds']['max']['z'] for s in shelves)
        ax.set_zlim(floor_z - 1, max_z + 2)
    else:
        ax.set_zlim(floor_z - 1, floor_z + 5)

    # Labels
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_zlabel('Z (meters)', fontsize=11)
    ax.set_title('3D Warehouse Layout - Shelves & Sections', fontsize=14, weight='bold')

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Add info text
    info_text = (
        f"Floor: {floor.get('area', 0):.1f} m²\n"
        f"Zones: {metadata.get('functional_zone_count', 0)}\n"
        f"Aisles: {metadata.get('aisle_count', 0)}\n"
        f"Shelves: {metadata.get('shelf_count', 0)}\n"
        f"Sections: {metadata.get('shelf_section_count', 0)}"
    )
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def visualize_multi_view(
    layout: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 12)
):
    """
    Create multi-view visualization: 3D view, top view, side views.
    """
    floor = layout['floor']
    aisles, shelves = extract_aisles_and_shelves(layout)
    functional_zones = layout.get('functional_zones', [])
    floor_bounds = floor['bounds']
    floor_z = floor_bounds['min']['z']

    # Zone color mapping
    zone_colors = {
        'receiving': ('sandybrown', 'saddlebrown'),
        'packing': ('plum', 'purple'),
        'pallet_truck': ('lightyellow', 'goldenrod'),
        'hub_robot': ('lightcyan', 'darkcyan'),
        'forklift': ('peachpuff', 'sienna'),
        'storage': ('lightskyblue', 'steelblue'),
        'general': ('palegreen', 'darkgreen')
    }

    fig = plt.figure(figsize=figsize)

    # Color schemes for shelf sections
    colors_horizontal = plt.cm.Blues(np.linspace(0.3, 0.8, 10))
    colors_vertical = plt.cm.Reds(np.linspace(0.3, 0.8, 10))

    # Collect all sections for 2D views
    all_sections = []
    for shelf in shelves:
        for section in get_sections_list(shelf):
            all_sections.append({
                'bounds': section['bounds'],
                'orientation': shelf['orientation'],
                'level_index': section['level_index'],
                'level_type': section['level_type']
            })

    # 3D view (top-left)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('3D View', fontsize=12, weight='bold')

    # Draw floor
    floor_vertices = floor['vertices']
    floor_3d = [[v[0], v[1], floor_z] for v in floor_vertices]
    floor_poly = Poly3DCollection([floor_3d], alpha=0.15, facecolor='lightgray', edgecolor='black', linewidth=2)
    ax1.add_collection3d(floor_poly)

    # Draw functional zones in 3D view
    for zone in functional_zones:
        zone_verts = zone.get('vertices', [])
        if not zone_verts:
            continue
        facecolor, _ = zone_colors.get(zone['type'], ('lightyellow', 'gray'))
        zone_3d = [[v[0], v[1], floor_z + 0.01] for v in zone_verts]
        zone_poly = Poly3DCollection(
            [zone_3d], alpha=0.2, facecolor=facecolor,
            edgecolor='gray', linewidth=1
        )
        ax1.add_collection3d(zone_poly)

    # Draw shelves (wireframe) and sections (filled)
    for shelf in shelves:
        orientation = shelf['orientation']

        # Shelf wireframe
        shelf_verts = create_box_vertices(shelf['bounds'])
        shelf_faces = create_box_faces(shelf_verts)
        shelf_wire = Poly3DCollection(
            shelf_faces, alpha=0.03, facecolor='gray',
            edgecolor='darkgray', linewidth=0.8, linestyle='--'
        )
        ax1.add_collection3d(shelf_wire)

        # Section boxes
        for section in get_sections_list(shelf):
            bounds = section['bounds']
            level_index = section['level_index']

            if orientation == 'horizontal':
                color = colors_horizontal[min(level_index, 9)]
            else:
                color = colors_vertical[min(level_index, 9)]

            vertices = create_box_vertices(bounds)
            faces = create_box_faces(vertices)
            sec_poly = Poly3DCollection(
                faces, alpha=0.5, facecolor=color,
                edgecolor='black', linewidth=0.5
            )
            ax1.add_collection3d(sec_poly)

    padding = 2.0
    ax1.set_xlim(floor_bounds['min']['x'] - padding, floor_bounds['max']['x'] + padding)
    ax1.set_ylim(floor_bounds['min']['y'] - padding, floor_bounds['max']['y'] + padding)
    if shelves:
        max_z = max(s['bounds']['max']['z'] for s in shelves)
        ax1.set_zlim(floor_z - 1, max_z + 2)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.view_init(elev=25, azim=-60)

    # Top view (XY plane) - top-right
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Top View (XY)', fontsize=12, weight='bold')
    ax2.set_aspect('equal')

    # Draw floor
    floor_xs = [v[0] for v in floor_vertices]
    floor_ys = [v[1] for v in floor_vertices]
    floor_patch = patches.Polygon(list(zip(floor_xs, floor_ys)), closed=True,
                                  edgecolor='black', facecolor='lightgray', alpha=0.15, linewidth=2)
    ax2.add_patch(floor_patch)

    # Draw functional zones in top view
    for zone in functional_zones:
        zone_verts = zone.get('vertices', [])
        if not zone_verts:
            continue
        facecolor, edgecolor = zone_colors.get(zone['type'], ('lightyellow', 'gray'))
        zxs = [v[0] for v in zone_verts]
        zys = [v[1] for v in zone_verts]
        zone_patch = patches.Polygon(
            list(zip(zxs, zys)), closed=True,
            edgecolor=edgecolor, facecolor=facecolor, alpha=0.3, linewidth=1.5
        )
        ax2.add_patch(zone_patch)
        zb = zone['bounds']
        ax2.text(zb['center']['x'], zb['center']['y'], zone['name'],
                fontsize=7, ha='center', va='center', color=edgecolor, weight='bold')

    # Draw aisles
    for aisle in aisles:
        bounds = aisle['bounds']
        aisle_rect = patches.Rectangle(
            (bounds['min']['x'], bounds['min']['y']),
            bounds['size']['x'], bounds['size']['y'],
            linewidth=1.5, edgecolor='darkgreen', facecolor='lightyellow', alpha=0.4, linestyle='--'
        )
        ax2.add_patch(aisle_rect)

    # Draw shelf outlines
    for shelf in shelves:
        sb = shelf['bounds']
        shelf_rect = patches.Rectangle(
            (sb['min']['x'], sb['min']['y']),
            sb['size']['x'], sb['size']['y'],
            linewidth=1.5, edgecolor='darkgray', facecolor='none', linestyle='--'
        )
        ax2.add_patch(shelf_rect)

    # Draw section footprints (color by level)
    for sec in all_sections:
        bounds = sec['bounds']
        color = 'lightblue' if sec['orientation'] == 'horizontal' else 'lightcoral'
        sec_rect = patches.Rectangle(
            (bounds['min']['x'], bounds['min']['y']),
            bounds['size']['x'], bounds['size']['y'],
            linewidth=1, edgecolor='navy', facecolor=color, alpha=0.3
        )
        ax2.add_patch(sec_rect)

        center = bounds['center']
        ax2.text(center['x'], center['y'], f"L{sec['level_index']}",
                fontsize=5, ha='center', va='center', color='navy', weight='bold')

    ax2.set_xlim(floor_bounds['min']['x'] - padding, floor_bounds['max']['x'] + padding)
    ax2.set_ylim(floor_bounds['min']['y'] - padding, floor_bounds['max']['y'] + padding)
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Side view XZ (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Side View (XZ)', fontsize=12, weight='bold')

    # Draw floor line
    ax3.plot([floor_bounds['min']['x'], floor_bounds['max']['x']], [floor_z, floor_z],
            'k-', linewidth=2, label='Floor')

    # Draw shelf outlines projected onto XZ
    for shelf in shelves:
        sb = shelf['bounds']
        shelf_rect = patches.Rectangle(
            (sb['min']['x'], sb['min']['z']),
            sb['size']['x'], sb['size']['z'],
            linewidth=1, edgecolor='darkgray', facecolor='none', linestyle='--'
        )
        ax3.add_patch(shelf_rect)

    # Draw sections projected onto XZ plane
    for sec in all_sections:
        bounds = sec['bounds']
        level_index = sec['level_index']

        if sec['orientation'] == 'horizontal':
            color = colors_horizontal[min(level_index, 9)]
        else:
            color = colors_vertical[min(level_index, 9)]

        rect = patches.Rectangle(
            (bounds['min']['x'], bounds['min']['z']),
            bounds['size']['x'], bounds['size']['z'],
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.6
        )
        ax3.add_patch(rect)

    ax3.set_xlim(floor_bounds['min']['x'] - padding, floor_bounds['max']['x'] + padding)
    if shelves:
        max_z = max(s['bounds']['max']['z'] for s in shelves)
        ax3.set_ylim(floor_z - 1, max_z + 2)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Z (m)', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Side view YZ (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Side View (YZ)', fontsize=12, weight='bold')

    # Draw floor line
    ax4.plot([floor_bounds['min']['y'], floor_bounds['max']['y']], [floor_z, floor_z],
            'k-', linewidth=2, label='Floor')

    # Draw shelf outlines projected onto YZ
    for shelf in shelves:
        sb = shelf['bounds']
        shelf_rect = patches.Rectangle(
            (sb['min']['y'], sb['min']['z']),
            sb['size']['y'], sb['size']['z'],
            linewidth=1, edgecolor='darkgray', facecolor='none', linestyle='--'
        )
        ax4.add_patch(shelf_rect)

    # Draw sections projected onto YZ plane
    for sec in all_sections:
        bounds = sec['bounds']
        level_index = sec['level_index']

        if sec['orientation'] == 'horizontal':
            color = colors_horizontal[min(level_index, 9)]
        else:
            color = colors_vertical[min(level_index, 9)]

        rect = patches.Rectangle(
            (bounds['min']['y'], bounds['min']['z']),
            bounds['size']['y'], bounds['size']['z'],
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.6
        )
        ax4.add_patch(rect)

    ax4.set_xlim(floor_bounds['min']['y'] - padding, floor_bounds['max']['y'] + padding)
    if shelves:
        max_z = max(s['bounds']['max']['z'] for s in shelves)
        ax4.set_ylim(floor_z - 1, max_z + 2)
    ax4.set_xlabel('Y (m)', fontsize=10)
    ax4.set_ylabel('Z (m)', fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Warehouse Layout - Multi-View', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Multi-view visualization saved to: {output_path}")
    else:
        plt.show()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize 3D warehouse layout from JSON'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'warehouse_layout' / 'warehouse_layout.json',
        help='Path to warehouse_layout.json'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'warehouse_layout' / 'warehouse_layout_3d.png',
        help='Path to save visualization'
    )
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Create multi-view visualization (3D + top + side views)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show interactive plot instead of saving'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide shelf labels'
    )
    parser.add_argument(
        '--elev',
        type=float,
        default=30,
        help='Elevation angle for 3D view (degrees)'
    )
    parser.add_argument(
        '--azim',
        type=float,
        default=-60,
        help='Azimuth angle for 3D view (degrees)'
    )

    args = parser.parse_args()

    if not args.json.exists():
        print(f"Error: JSON file not found: {args.json}")
        exit(1)

    layout = load_warehouse_layout(args.json)

    output = None if args.show else args.output

    if args.multi_view:
        visualize_multi_view(layout, output)
    else:
        visualize_3d_layout(
            layout,
            output,
            show_labels=not args.no_labels,
            elev=args.elev,
            azim=args.azim
        )
