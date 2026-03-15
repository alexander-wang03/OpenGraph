"""
Agglomerative Information Bottleneck clustering.

Adapted from opennav_mem/star/clio_batch/ib_cluster.py.
Changes vs. original:
  - Import paths fixed for TierGraph project structure.
  - Debug plotting removed (matplotlib / distinctipy dependencies dropped).
  - ClusterIBConfig accepts a dict in addition to a YAML file path.
  - compute_sim_to_tasks() inlined (no separate helpers.py dependency).
"""

import numpy as np
import networkx as nx
from utils.ib import information_metrics as metrics


# ---------------------------------------------------------------------------
# Similarity helper (inlined from clio_batch/helpers.py)
# ---------------------------------------------------------------------------

def compute_sim_to_tasks(task_features, region_features):
    """
    Compute cosine similarities between task and region (object) embeddings.

    Args:
        task_features:   (T, D) array of task embeddings
        region_features: (N, D) array of object embeddings

    Returns:
        (T, N) cosine similarity matrix
    """
    task_features   = np.array(task_features,   dtype=np.float64)
    region_features = np.array(region_features, dtype=np.float64)

    # Reshape to (T, D) and (N, D)
    D = task_features.shape[-1]
    task_features   = task_features.reshape(-1, D)
    region_features = region_features.reshape(-1, D)

    # L2 normalise
    task_norms   = np.linalg.norm(task_features,   axis=1, keepdims=True) + 1e-12
    region_norms = np.linalg.norm(region_features, axis=1, keepdims=True) + 1e-12
    task_features   /= task_norms
    region_features /= region_norms

    return task_features @ region_features.T   # (T, N)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ClusterIBConfig:
    """
    Configuration for ClusterIB.

    Accepts either a YAML file path (str/Path) or a plain dict.
    """

    # Defaults
    _DEFAULTS = dict(
        debug=False,
        sims_thres=0.20,
        delta=0.15,
        top_k_tasks=1,
        cumulative=False,
        use_lerf_loss=False,
        lerf_loss_cannonical_phrases=[],
    )

    def __init__(self, source):
        if isinstance(source, dict):
            config = source
        else:
            import yaml
            with open(source, 'r') as f:
                config = yaml.safe_load(f)

        cfg = {**self._DEFAULTS, **config}
        self.debug            = cfg['debug']
        self.sims_thres       = cfg['sims_thres']
        self.delta            = cfg['delta']
        self.top_k            = cfg['top_k_tasks']
        self.cumulative       = cfg['cumulative']
        self.use_lerf_loss    = cfg['use_lerf_loss']
        self.lerf_loss_cannonical_phrases = cfg['lerf_loss_cannonical_phrases']


# ---------------------------------------------------------------------------
# Clustering algorithm
# ---------------------------------------------------------------------------

class ClusterIB:
    """
    Agglomerative Information Bottleneck clustering.

    Usage:
        ib = ClusterIB(config)
        ib.setup_py_x(region_features, task_features)   # (N, D), (T, D)
        ib.initialize_nx_graph(nx_graph)                 # spatial/hierarchy graph
        clusters = ib.find_clusters()                    # list[list[node_id]]
    """

    def __init__(self, config: ClusterIBConfig):
        self.config     = config
        self.n          = None
        self.px         = None
        self.py         = None
        self.py_x       = None
        self.nx_graph   = None
        self.pos        = None
        self.Ixy        = None
        self.dIcy_weight = 1.0
        self.idx_mapping = {}   # internal int index → original node id

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_py_x(self, region_features, task_features):
        """Compute initial probability distributions from feature matrices."""
        self.px, self.py_x, self.py = self._compute_initial_probabilities(
            region_features, task_features)
        self.n = len(self.px)
        self.Ixy = metrics.mutual_information(self.px, self.py, self.py_x)

    def _compute_initial_probabilities(self, region_features, task_features):
        num_tasks = task_features.shape[0]
        m = num_tasks + 1        # +1 for null task
        n = len(region_features)
        k = min(num_tasks + 1, self.config.top_k)

        py_x_tmp = np.zeros((m, n))
        py_x_tmp[0, :] = self.config.sims_thres   # null-task floor

        cos_sim = compute_sim_to_tasks(task_features, region_features)  # (T, N)
        cos_sim = np.clip(cos_sim, 0, 1000)
        py_x_tmp[1:, :] = cos_sim

        py_x = 1e-12 * np.ones((m, n))
        l = 1 if self.config.cumulative else k

        # find which objects have no strong task signal (null-task wins)
        top_inds  = np.argpartition(py_x_tmp, -1, axis=0)[-1:]
        null_tasks = np.where(top_inds == 0)[1]

        while l <= k:
            top_inds = np.argpartition(py_x_tmp, -l, axis=0)[-l:]
            py_x[top_inds, np.arange(n)] += py_x_tmp[top_inds, np.arange(n)]
            l += 1

        # null-task objects: suppress task rows, set null-task = 1.0
        py_x[:, null_tasks] = 1e-12
        py_x[0, null_tasks] = 1.0

        px  = np.ones(n) / n
        py  = np.ones(m) / m
        py_x = py_x / np.sum(py_x, axis=0)
        return px, py_x, py

    def update_delta_as_part(self, full_region_features, task_features):
        """Factorized clustering: scale delta by subset/full ratio."""
        px_full, py_x_full, _ = self._compute_initial_probabilities(
            full_region_features, task_features)
        self.Ixy = metrics.mutual_information(px_full, self.py, py_x_full)
        self.dIcy_weight = self.n / len(full_region_features)

    def initialize_nx_graph(self, nx_graph):
        """
        Translate an arbitrary nx.Graph into the internal indexed representation.

        Each node in nx_graph must have a 'position' attribute (list/array len≥2)
        used to order nodes consistently. Node IDs can be any hashable type.
        """
        node_idx = {}
        idx = 0
        self.nx_graph = nx.Graph()
        self.pos = {}

        for node in nx_graph.nodes:
            self.idx_mapping[idx] = node
            self.nx_graph.add_node(idx)
            pos = nx_graph.nodes[node].get('position', [0.0, 0.0, 0.0])
            self.pos[idx] = pos[:2]
            node_idx[node] = idx
            idx += 1

        for edge in nx_graph.edges:
            self.nx_graph.add_edge(node_idx[edge[0]], node_idx[edge[1]])

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def find_clusters(self):
        """
        Run agglomerative IB until information loss exceeds delta threshold.

        Returns:
            list[list[node_id]]: cluster assignments using original node IDs.
        """
        # Initialise cluster assignments: each object is its own cluster
        pc_x     = np.eye(self.n)
        pc       = pc_x @ self.px
        py_c     = self.py_x * self.px @ np.transpose(pc_x) / pc
        prev_Icy = metrics.mutual_information(pc, self.py, py_c)

        last_merged_node = None
        t = 0
        delta = 0.0

        while True:
            if self.nx_graph.number_of_edges() == 0:
                break

            # Only recompute weights for edges touching the last merged node
            if last_merged_node is None:
                update_edges = list(self.nx_graph.edges())
            else:
                update_edges = [(last_merged_node, nb)
                                for nb in self.nx_graph.neighbors(last_merged_node)]

            for e in update_edges:
                w = self._compute_edge_weight(pc, py_c, e[0], e[1])
                self.nx_graph[e[0]][e[1]]['weight'] = w

            min_edge = min(self.nx_graph.edges(),
                           key=lambda x: self.nx_graph[x[0]][x[1]]['weight'])

            # Trial merge: compute information loss
            py_c_tmp = np.copy(py_c)
            pc_tmp   = np.copy(pc)
            pc_x_tmp = np.copy(pc_x)

            merged_pc = pc[min_edge[0]] + pc[min_edge[1]]
            py_c_tmp[:, min_edge[0]] = (
                py_c[:, min_edge[0]] * pc[min_edge[0]] +
                py_c[:, min_edge[1]] * pc[min_edge[1]]
            ) / merged_pc
            py_c_tmp[:, min_edge[1]] = 0
            pc_tmp[min_edge[0]]      = merged_pc
            pc_tmp[min_edge[1]]      = 0
            pc_x_tmp[min_edge[0], :] = pc_x[min_edge[0], :] + pc_x[min_edge[1], :]
            pc_x_tmp[min_edge[1], :] = 0

            Icy   = metrics.mutual_information(pc_tmp, self.py, py_c_tmp)
            dIcy  = prev_Icy - Icy
            delta = self.dIcy_weight * dIcy / (self.Ixy + 1e-12)

            if delta > self.config.delta:
                break   # merge would lose too much task information

            # Commit merge
            py_c      = py_c_tmp
            pc        = pc_tmp
            pc_x      = pc_x_tmp
            prev_Icy  = Icy

            last_merged_node = min_edge[0]
            self.nx_graph = nx.contracted_edge(
                self.nx_graph, min_edge, self_loops=False)
            t += 1

        internal_clusters = self._get_clusters_from_pc_x(pc_x)
        return self._to_node_clusters(internal_clusters)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_edge_weight(self, pc, py_c, i, j):
        prior  = pc[[i, j]] / np.sum(pc[[i, j]])
        weight = (pc[i] + pc[j]) * metrics.js_divergence(py_c[:, [i, j]], prior)
        return weight

    def _get_clusters_from_pc_x(self, pc_x):
        cluster_mapping = np.argmax(pc_x, axis=0)
        clusters = {}
        for i in range(pc_x.shape[1]):
            c = cluster_mapping[i]
            clusters.setdefault(c, []).append(i)
        return list(clusters.values())

    def _to_node_clusters(self, idx_clusters):
        return [[self.idx_mapping[idx] for idx in cluster]
                for cluster in idx_clusters]
