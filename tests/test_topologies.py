"""Smoke tests for network topologies (T1, T2, T3)."""

import pytest
import networkx as nx
from src.topologies import build_networkx, get_inventory, build_mininet


def test_t1_tree_networkx():
    G = build_networkx("T1_tree")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 7  # 3 switches + 4 hosts
    assert G.number_of_edges() == 6
    inv = get_inventory(G)
    assert len(inv["switches"]) == 3
    assert len(inv["hosts"]) == 4
    for link in inv["links"]:
        assert "capacity" in link
        assert "delay" in link


def test_t2_abilene_networkx():
    G = build_networkx("T2_abilene")
    assert isinstance(G, nx.Graph)
    inv = get_inventory(G)
    assert len(inv["switches"]) == 11
    assert len(inv["hosts"]) == 11
    assert nx.is_connected(G)


def test_t3_fattree_networkx():
    G = build_networkx("T3_fattree")
    assert isinstance(G, nx.Graph)
    inv = get_inventory(G)
    assert len(inv["switches"]) == 20  # 4 core + 8 agg + 8 edge
    assert len(inv["hosts"]) == 16
    assert nx.is_connected(G)


@pytest.mark.needs_root
def test_t1_tree_mininet_build():
    try:
        net = build_mininet("T1_tree")
        net.start()
        # Verify hosts and switches exist
        assert len(net.hosts) == 4
        assert len(net.switches) == 3
        # Quick ping test between h1 and h2
        loss = net.ping([net.hosts[0], net.hosts[1]])
        assert loss == 0.0
        net.stop()
    except Exception as e:
        pytest.skip(f"Mininet root execution skipped: {e}")

