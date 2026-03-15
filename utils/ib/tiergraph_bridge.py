"""
TierGraph ↔ IB bridge.

Converts TierGraph hierarchy into an IB-compatible NetworkX graph, and
maps IB cluster assignments back into a compressed TierGraph JSON.

The key design: instead of building a spatial proximity graph (bbox dilation
+ connected components as in CLIO/opennav_mem), the adjacency graph is
derived directly from the TierGraph hierarchy.

Adjacency rules (who can merge):
  1. Objects sharing the same shelf (storage zone)  → connected
     (physically co-located on the same shelving unit)
  2. Objects in the same aisle (storage zone)        → connected
  3. Objects in the same non-storage zone             → connected
     (open floor — zone_receiving, zone_forklift, etc.)
  4. Objects in different zones / different shelves   → NOT connected
     (never merge across warehouse structural boundaries)

This is semantically stronger than spatial proximity: two objects on the
same shelf are guaranteed to be co-located, and the hierarchy already
encodes the structural constraint we want to preserve.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_clip_features(objects) -> Tuple[np.ndarray, List[int]]:
    """
    Extract CLIP/SBERT feature vectors from a MapObjectList.

    OpenGraph stores per-object features under the key 'ft' (SBERT-encoded
    caption embedding, shape (D,) as a torch.Tensor or numpy array).

    Returns:
        features:      (N, D) float64 array of valid features
        valid_indices: list[int] mapping row i → object index in `objects`
    """
    features = []
    valid_indices = []

    for i, obj in enumerate(objects):
        feat = None
        for key in ('ft', 'clip_ft', 'features', 'semantic_feature'):
            if key in obj:
                feat = obj[key]
                break

        if feat is None:
            continue

        # Convert torch.Tensor → numpy if needed
        if hasattr(feat, 'detach'):
            feat = feat.detach().cpu().numpy()
        feat = np.array(feat, dtype=np.float64).flatten()

        if feat.size == 0:
            continue

        features.append(feat)
        valid_indices.append(i)

    if not features:
        return np.zeros((0, 0)), []

    return np.stack(features, axis=0), valid_indices


# ---------------------------------------------------------------------------
# Hierarchy traversal helpers
# ---------------------------------------------------------------------------

def _build_node_lookup(tiergraph: dict) -> dict:
    """Return {node_id: node_dict} for fast lookup."""
    return {n['id']: n for n in tiergraph.get('nodes', [])}


def _find_ancestor_of_type(node_id: str, target_type: str,
                            node_lookup: dict) -> Optional[str]:
    """Walk up the hierarchy until we find a node of target_type."""
    current_id = node_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = node_lookup.get(current_id)
        if node is None:
            break
        if node['type'] == target_type:
            return current_id
        current_id = node.get('parent_id')
    return None


def _get_merge_group(obj_node: dict, node_lookup: dict) -> str:
    """
    Return the 'merge group' key for an object node.

    Objects in the same merge group can be merged by IB.

    Rules:
    - Storage zone, object in a section      → group by section_id
      (only objects in the same shelf section may merge)
    - Storage zone, object in aisle walkway  → group by aisle_id
      (objects in the same aisle walkway may merge)
    - Storage zone, directly under shelf     → own group (no merging)
    - Non-storage zone                       → group by zone_id
      (all objects on the open floor of a zone may merge)
    """
    obj_id = obj_node['id']
    if obj_node.get('parent_id') is None:
        return obj_id  # isolated, own group

    # Walk up to find zone
    zone_id = _find_ancestor_of_type(obj_id, 'zone', node_lookup)

    if zone_id != 'zone_storage':
        # Non-storage zone: group by zone
        return zone_id or obj_id

    # Storage zone — check finest structural unit first
    section_id = _find_ancestor_of_type(obj_id, 'section', node_lookup)
    if section_id:
        return section_id  # on a shelf section: only merge within same section

    aisle_id = _find_ancestor_of_type(obj_id, 'aisle', node_lookup)
    if aisle_id:
        return aisle_id   # in aisle walkway: merge within same aisle

    # Directly under shelf (no section match) or direct zone child: no merging
    return obj_id


# ---------------------------------------------------------------------------
# IB graph construction
# ---------------------------------------------------------------------------

def build_ib_graph(tiergraph: dict,
                   objects,
                   valid_obj_indices: List[int]) -> nx.Graph:
    """
    Build an IB-compatible NetworkX adjacency graph from the TierGraph
    hierarchy.

    Nodes: object IDs present in the TierGraph AND in valid_obj_indices.
    Edges: objects that share the same merge group (shelf, aisle, or zone).
    Node attributes: 'position' (3D centroid from TierGraph bounds.center).

    Args:
        tiergraph:         loaded TierGraph JSON dict
        objects:           MapObjectList (for centroid positions)
        valid_obj_indices: list of object indices that have valid features
                           (from extract_clip_features)

    Returns:
        G:            nx.Graph ready for ClusterIB.initialize_nx_graph()
        group_map:    {obj_id: group_key} for diagnostic printing
    """
    node_lookup = _build_node_lookup(tiergraph)
    valid_set   = set(valid_obj_indices)

    # ---- collect object nodes that exist in the TierGraph ----
    obj_nodes = [
        n for n in tiergraph.get('nodes', [])
        if n['type'] == 'object'
    ]

    # Filter to those with valid features
    # Object IDs are "object_{i}" — extract index and check valid_set
    def obj_id_to_idx(obj_id: str) -> Optional[int]:
        try:
            return int(obj_id.split('_')[1])
        except (IndexError, ValueError):
            return None

    filtered_nodes = []
    for n in obj_nodes:
        idx = obj_id_to_idx(n['id'])
        if idx is not None and idx in valid_set:
            filtered_nodes.append(n)

    # ---- build merge groups ----
    group_map: Dict[str, str] = {}
    for n in filtered_nodes:
        group_map[n['id']] = _get_merge_group(n, node_lookup)

    # ---- invert: group_key → [obj_id, ...] ----
    groups: Dict[str, List[str]] = {}
    for obj_id, group_key in group_map.items():
        groups.setdefault(group_key, []).append(obj_id)

    # ---- build graph ----
    G = nx.Graph()

    for n in filtered_nodes:
        obj_id = n['id']

        # Position from TierGraph bounds.center
        center = n.get('bounds', {}).get('center', {})
        pos = [
            center.get('x', 0.0),
            center.get('y', 0.0),
            center.get('z', 0.0),
        ]
        G.add_node(obj_id, position=pos)

    # Add edges within each merge group (fully connected subgraph)
    for group_key, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j])

    print(f"  IB graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges across {len(groups)} merge groups")

    group_sizes = sorted([len(v) for v in groups.values()], reverse=True)
    print(f"  Merge group sizes: {group_sizes[:10]}"
          + ("..." if len(group_sizes) > 10 else ""))

    return G, group_map


# ---------------------------------------------------------------------------
# Apply IB results back to TierGraph
# ---------------------------------------------------------------------------

def apply_clusters_to_tiergraph(
        tiergraph: dict,
        clusters: List[List[str]],
        objects,
        valid_obj_indices: List[int],
        task_names: List[str]) -> dict:
    """
    Insert IB cluster nodes into the TierGraph JSON.

    For single-member clusters: object node is kept as-is.
    For multi-member clusters: a new 'cluster' node replaces the
    individual object nodes. The cluster's parent is the lowest common
    ancestor of its members.

    Args:
        tiergraph:         original TierGraph JSON dict
        clusters:          output of ClusterIB.find_clusters()
                           list[list[obj_id]]
        objects:           MapObjectList (for feature averaging)
        valid_obj_indices: feature-valid object indices
        task_names:        list of task strings (for metadata)

    Returns:
        compressed TierGraph JSON dict
    """
    import copy

    compressed  = copy.deepcopy(tiergraph)
    node_lookup = _build_node_lookup(compressed)  # must point into compressed, not original

    # Index of valid features (same order as ClusterIB was called with)
    idx_to_feat = {}
    for row_i, obj_idx in enumerate(valid_obj_indices):
        obj_id = f"object_{obj_idx}"
        idx_to_feat[obj_id] = row_i

    # Track which object nodes to remove and which cluster nodes to add
    nodes_to_remove = set()
    nodes_to_add    = []
    edges_to_remove = set()
    edges_to_add    = []

    cluster_counter = 0

    for cluster_members in clusters:
        if len(cluster_members) == 1:
            continue   # single-member cluster: no change

        cluster_id = f"cluster_{cluster_counter}"
        cluster_counter += 1

        # ---- find lowest common ancestor ----
        lca_id = _find_lca(cluster_members, node_lookup)
        if lca_id is None:
            # Fallback: skip compression for this cluster
            continue

        # ---- merged bounds (union) ----
        xs, ys, zs = [], [], []
        captions = []
        features = []
        member_indices = []

        for obj_id in cluster_members:
            obj_node = node_lookup.get(obj_id, {})
            center = obj_node.get('bounds', {}).get('center', {})
            xs.append(center.get('x', 0.0))
            ys.append(center.get('y', 0.0))
            zs.append(center.get('z', 0.0))

            name = obj_node.get('name', obj_id)
            if name and not name.startswith('Object '):
                captions.append(name)

            # feature for mean
            obj_idx = obj_id_to_idx(obj_id)
            if obj_idx is not None and obj_idx < len(objects):
                feat = None
                for key in ('ft', 'clip_ft', 'features', 'semantic_feature'):
                    if key in objects[obj_idx]:
                        feat = objects[obj_idx][key]
                        break
                if feat is not None:
                    if hasattr(feat, 'detach'):
                        feat = feat.detach().cpu().numpy()
                    features.append(np.array(feat, dtype=np.float64).flatten())

            member_indices.append(obj_id)

        merged_center = {
            'x': float(np.mean(xs)),
            'y': float(np.mean(ys)),
            'z': float(np.mean(zs)),
        }
        merged_bounds = {'center': merged_center}

        # Representative caption: most common non-generic one
        representative_caption = (
            max(set(captions), key=captions.count) if captions
            else f"Cluster {cluster_counter - 1}"
        )

        # Mean feature (stored as list for JSON serialisation)
        mean_feature = (
            np.mean(features, axis=0).tolist() if features else []
        )

        # ---- cluster node ----
        cluster_node = {
            'id':        cluster_id,
            'type':      'cluster',
            'name':      representative_caption,
            'bounds':    merged_bounds,
            'parent_id': lca_id,
            'children':  list(cluster_members),
            'metadata': {
                'member_ids':    list(cluster_members),
                'member_count':  len(cluster_members),
                'mean_feature':  mean_feature,
                'tasks':         task_names,
            }
        }
        nodes_to_add.append(cluster_node)
        edges_to_add.append({
            'source': lca_id,
            'target': cluster_id,
            'type':   'contains'
        })

        # Mark member objects + their original parent edges for removal
        for obj_id in cluster_members:
            nodes_to_remove.add(obj_id)
            edges_to_remove.add((obj_id,))   # any edge involving obj_id

        # Update LCA's children list: remove members, add cluster_id
        lca_node = node_lookup.get(lca_id)
        if lca_node:
            children = lca_node.get('children', [])
            for obj_id in cluster_members:
                if obj_id in children:
                    children.remove(obj_id)
            if cluster_id not in children:
                children.append(cluster_id)

    # ---- rebuild nodes list ----
    compressed['nodes'] = [
        n for n in compressed['nodes']
        if n['id'] not in nodes_to_remove
    ]
    compressed['nodes'].extend(nodes_to_add)

    # ---- purge deleted object IDs from all children lists ----
    # (intermediate nodes like sections may still reference merged objects)
    for n in compressed['nodes']:
        if n.get('type') == 'cluster':
            # Cluster children are the merged member list — keep in metadata only
            n['children'] = []
        elif 'children' in n:
            n['children'] = [c for c in n['children'] if c not in nodes_to_remove]

    # ---- rebuild edges list ----
    removed_obj_ids = nodes_to_remove
    compressed['edges'] = [
        e for e in compressed['edges']
        if e['source'] not in removed_obj_ids
        and e['target'] not in removed_obj_ids
    ]
    compressed['edges'].extend(edges_to_add)

    # ---- update statistics ----
    stats = compressed.get('statistics', {})
    node_types = {}
    for n in compressed['nodes']:
        t = n['type']
        node_types[t] = node_types.get(t, 0) + 1
    stats['total_nodes'] = len(compressed['nodes'])
    stats['total_edges'] = len(compressed['edges'])
    stats['nodes_by_type'] = node_types
    stats['ib_clusters_created'] = cluster_counter
    stats['objects_merged'] = len(nodes_to_remove)
    compressed['statistics'] = stats

    # ---- metadata ----
    compressed.setdefault('metadata', {})
    compressed['metadata']['compression'] = {
        'method':     'Agglomerative Information Bottleneck (TierGraph-aware)',
        'tasks':      task_names,
        'original_object_count': (
            stats.get('nodes_by_type', {}).get('object', 0) + len(nodes_to_remove)
        ),
        'clusters_created': cluster_counter,
        'objects_merged': len(nodes_to_remove),
    }

    return compressed


# ---------------------------------------------------------------------------
# LCA helper
# ---------------------------------------------------------------------------

def _find_lca(obj_ids: List[str], node_lookup: dict) -> Optional[str]:
    """
    Find the lowest common ancestor of a set of object nodes.

    Builds the ancestor chain for each object and returns the deepest
    node that appears in all chains.
    """
    def ancestor_chain(node_id):
        chain = []
        visited = set()
        current = node_id
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            node = node_lookup.get(current, {})
            current = node.get('parent_id')
        return chain

    chains = [ancestor_chain(obj_id) for obj_id in obj_ids]
    if not chains:
        return None

    # Convert each chain to a set for quick lookup
    ancestor_sets = [set(c) for c in chains]
    # Walk down the first chain, find deepest node in all sets
    for node_id in chains[0]:
        if all(node_id in s for s in ancestor_sets):
            return node_id
    return None


def obj_id_to_idx(obj_id: str) -> Optional[int]:
    """Extract integer index from 'object_{i}' string."""
    try:
        return int(obj_id.split('_')[1])
    except (IndexError, ValueError):
        return None
