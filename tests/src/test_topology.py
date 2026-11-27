from __future__ import annotations

import pytest

from src.topology import build_topology, select_byzantine_nodes, topology_summary


def test_topologies_connected_and_degree() -> None:
    for name in ("ring", "random", "small_world"):
        g = build_topology(name, n_nodes=20, seed=7)
        byz = select_byzantine_nodes(20, 0.3, seed=7)
        info = topology_summary(g, byz)
        assert info["graph_connected"] is True
        assert info["n_byzantine"] == 6
        if name == "ring":
            assert info["degree_min"] == 2 and info["degree_max"] == 2


def test_topology_and_byzantine_edge_cases() -> None:
    assert select_byzantine_nodes(10, 0.0, seed=1) == set()
    assert select_byzantine_nodes(4, 1.0, seed=1) == {0, 1, 2, 3}
    assert select_byzantine_nodes(10, 0.3, seed=5) == select_byzantine_nodes(10, 0.3, seed=5)

    with pytest.raises(ValueError, match="byzantine_frac"):
        select_byzantine_nodes(10, 1.2)
    with pytest.raises(ValueError, match="ws_k"):
        build_topology("small_world", n_nodes=6, ws_k=3)
    with pytest.raises(ValueError, match="Unknown topology"):
        build_topology("mesh", n_nodes=6)

    g = build_topology("ring", n_nodes=4, seed=0)
    info = topology_summary(g, {0})
    assert info["benign_subgraph_connected"] is True
    assert info["benign_node_byz_deg_mean"] == pytest.approx(2.0 / 3.0)
    assert info["benign_node_benign_deg_mean"] == pytest.approx(4.0 / 3.0)
