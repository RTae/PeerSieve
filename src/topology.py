from __future__ import annotations

import networkx as nx
import numpy as np


def build_topology(
    name: str,
    n_nodes: int = 20,
    er_p: float = 0.3,
    ws_k: int = 4,
    ws_p: float = 0.3,
    seed: int = 42,
) -> nx.Graph:
    """Build a connected undirected communication graph on n_nodes peers.

    Args:
        name: Graph type ('ring', 'random', or 'small_world').
        n_nodes: Number of peers in the graph.
        er_p: Edge probability for Erdos-Renyi random graphs.
        ws_k: Nearest-neighbor degree for Watts-Strogatz small-world graphs.
        ws_p: Rewiring probability for Watts-Strogatz small-world graphs.
        seed: Random seed used when sampling connected random or small-world graphs.

    Returns:
        Connected NetworkX undirected graph with integer nodes 0..n_nodes-1.
    """
    if name == "ring":
        return nx.cycle_graph(n_nodes)

    if name == "random":
        for attempt in range(256):
            graph = nx.erdos_renyi_graph(n_nodes, er_p, seed=seed + attempt)
            if nx.is_connected(graph):
                return graph
        raise RuntimeError(f"Failed to sample a connected Erdos-Renyi graph (n={n_nodes}, p={er_p}).")

    if name == "small_world":
        if ws_k >= n_nodes or ws_k < 2 or ws_k % 2 != 0:
            raise ValueError(f"ws_k must be an even integer in [2, n_nodes-1], got {ws_k}.")
        for attempt in range(256):
            graph = nx.watts_strogatz_graph(n_nodes, ws_k, ws_p, seed=seed + attempt)
            if nx.is_connected(graph):
                return graph
        raise RuntimeError(
            f"Failed to sample a connected Watts-Strogatz graph (n={n_nodes}, k={ws_k}, p={ws_p})."
        )

    raise ValueError(f"Unknown topology name: {name!r}")


def select_byzantine_nodes(
    n_nodes: int,
    byzantine_frac: float,
    seed: int = 42,
) -> set[int]:
    """Sample Byzantine peer IDs without replacement using seed.

    Args:
        n_nodes: Total number of peers.
        byzantine_frac: Fraction of peers to mark Byzantine in [0, 1].
        seed: Random seed for reproducible selection.

    Returns:
        Set of selected Byzantine peer IDs of size round(byzantine_frac * n_nodes).
    """
    if not (0.0 <= byzantine_frac <= 1.0):
        raise ValueError(f"byzantine_frac must be in [0, 1], got {byzantine_frac}.")
    n_byz = int(round(byzantine_frac * n_nodes))
    if n_byz == 0:
        return set()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_nodes, size=n_byz, replace=False)
    return {int(x) for x in chosen}


def topology_summary(graph: nx.Graph, byzantine_ids: set[int]) -> dict[str, object]:
    """Compute graph connectivity and benign-neighborhood degree statistics.

    Args:
        graph: Undirected peer communication graph.
        byzantine_ids: Set of Byzantine peer IDs.

    Returns:
        Dictionary of node and edge counts, degree statistics, and benign subgraph connectivity.
    """
    degrees = [d for _, d in graph.degree()]
    benign_ids = [n for n in graph.nodes() if n not in byzantine_ids]
    benign_sub = graph.subgraph(benign_ids)

    benign_neighbors_counts = []
    byz_neighbors_counts = []
    for node in benign_ids:
        nbrs = list(graph.neighbors(node))
        benign_neighbors_counts.append(sum(1 for j in nbrs if j not in byzantine_ids))
        byz_neighbors_counts.append(sum(1 for j in nbrs if j in byzantine_ids))

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_byzantine": len(byzantine_ids),
        "byzantine_ids": sorted(byzantine_ids),
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "graph_connected": nx.is_connected(graph),
        "benign_subgraph_connected": nx.is_connected(benign_sub) if len(benign_ids) > 0 else False,
        "benign_node_benign_deg_mean": float(np.mean(benign_neighbors_counts)) if benign_neighbors_counts else 0.0,
        "benign_node_byz_deg_mean": float(np.mean(byz_neighbors_counts)) if byz_neighbors_counts else 0.0,
    }
