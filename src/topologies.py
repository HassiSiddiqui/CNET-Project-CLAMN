"""CL-ANM Network Topologies (T1 Tree, T2 Abilene, T3 Fat-Tree).

Exposes build_networkx() and build_mininet() with rich node/edge attributes:
role, dialect, interfaces, capacity (Mbps), delay (ms), reliability.
"""

from __future__ import annotations
import argparse
import sys
from typing import Any, Dict, List, Tuple
import networkx as nx


def build_networkx(topo_name: str = "T1_tree") -> nx.Graph:
    """Build a NetworkX graph with node and edge attributes for verification and simulation."""
    G = nx.Graph(name=topo_name)

    if topo_name == "T1_tree":
        # T1: 3 switches (s1 root, s2-s3 agg), 4 hosts (h1-h2 on s2, h3-h4 on s3)
        switches = [
            ("s1", {"role": "root", "dialect": "ovs-ofctl", "dpid": "0000000000000001"}),
            ("s2", {"role": "agg", "dialect": "ovs-ofctl", "dpid": "0000000000000002"}),
            ("s3", {"role": "agg", "dialect": "ovs-ofctl", "dpid": "0000000000000003"}),
        ]
        hosts = [
            ("h1", {"role": "host", "ip": "10.0.0.1", "mac": "00:00:00:00:00:01"}),
            ("h2", {"role": "host", "ip": "10.0.0.2", "mac": "00:00:00:00:00:02"}),
            ("h3", {"role": "host", "ip": "10.0.0.3", "mac": "00:00:00:00:00:03"}),
            ("h4", {"role": "host", "ip": "10.0.0.4", "mac": "00:00:00:00:00:04"}),
        ]
        for node, attr in switches + hosts:
            attr["interfaces"] = []
            G.add_node(node, **attr)

        links: List[Tuple[str, str, float, float, float, str, str]] = [
            ("s1", "s2", 100.0, 2.0, 0.999, "s1-eth1", "s2-eth1"),
            ("s1", "s3", 100.0, 2.0, 0.999, "s1-eth2", "s3-eth1"),
            ("s2", "h1", 100.0, 1.0, 0.999, "s2-eth2", "h1-eth0"),
            ("s2", "h2", 100.0, 1.0, 0.999, "s2-eth3", "h2-eth0"),
            ("s3", "h3", 100.0, 1.0, 0.999, "s3-eth2", "h3-eth0"),
            ("s3", "h4", 100.0, 1.0, 0.999, "s3-eth3", "h4-eth0"),
        ]
        for u, v, cap, delay, rel, intf_u, intf_v in links:
            G.add_edge(u, v, capacity=cap, delay=delay, reliability=rel, intf_u=intf_u, intf_v=intf_v)
            G.nodes[u]["interfaces"].append(intf_u)
            G.nodes[v]["interfaces"].append(intf_v)

    elif topo_name == "T2_abilene":
        # T2 Abilene: 11 nodes (Internet2 topology zoo), 14 links, 11 hosts
        abilene_nodes = [
            ("s1", "NewYork", "0000000000000001"),
            ("s2", "Chicago", "0000000000000002"),
            ("s3", "WashingtonDC", "0000000000000003"),
            ("s4", "Seattle", "0000000000000004"),
            ("s5", "Atlanta", "0000000000000005"),
            ("s6", "Sunnyvale", "0000000000000006"),
            ("s7", "Indianapolis", "0000000000000007"),
            ("s8", "Houston", "0000000000000008"),
            ("s9", "LosAngeles", "0000000000000009"),
            ("s10", "KansasCity", "000000000000000a"),
            ("s11", "Denver", "000000000000000b"),
        ]
        for s, city, dpid in abilene_nodes:
            G.add_node(s, role="backbone", city=city, dialect="ovs-ofctl", dpid=dpid, interfaces=[])
            h = f"h{s[1:]}"
            idx = int(s[1:])
            G.add_node(h, role="host", ip=f"10.0.0.{idx}", mac=f"00:00:00:00:00:{idx:02x}", interfaces=[])

        abilene_links = [
            ("s1", "s2", 1000.0, 5.0, 0.999),
            ("s1", "s3", 1000.0, 4.0, 0.999),
            ("s2", "s7", 1000.0, 3.0, 0.999),
            ("s3", "s5", 1000.0, 6.0, 0.999),
            ("s4", "s6", 1000.0, 5.0, 0.999),
            ("s4", "s11", 1000.0, 7.0, 0.999),
            ("s5", "s7", 1000.0, 5.0, 0.999),
            ("s5", "s8", 1000.0, 8.0, 0.999),
            ("s6", "s9", 1000.0, 4.0, 0.999),
            ("s6", "s11", 1000.0, 6.0, 0.999),
            ("s7", "s10", 1000.0, 4.0, 0.999),
            ("s8", "s9", 1000.0, 9.0, 0.999),
            ("s8", "s10", 1000.0, 7.0, 0.999),
            ("s10", "s11", 1000.0, 5.0, 0.999),
        ]
        port_counts: Dict[str, int] = {f"s{i}": 0 for i in range(1, 12)}
        for u, v, cap, delay, rel in abilene_links:
            port_counts[u] += 1
            port_counts[v] += 1
            intf_u, intf_v = f"{u}-eth{port_counts[u]}", f"{v}-eth{port_counts[v]}"
            G.add_edge(u, v, capacity=cap, delay=delay, reliability=rel, intf_u=intf_u, intf_v=intf_v)
            G.nodes[u]["interfaces"].append(intf_u)
            G.nodes[v]["interfaces"].append(intf_v)

        for i in range(1, 12):
            s = f"s{i}"
            h = f"h{i}"
            port_counts[s] += 1
            intf_s, intf_h = f"{s}-eth{port_counts[s]}", f"{h}-eth0"
            G.add_edge(s, h, capacity=1000.0, delay=1.0, reliability=0.999, intf_u=intf_s, intf_v=intf_h)
            G.nodes[s]["interfaces"].append(intf_s)
            G.nodes[h]["interfaces"].append(intf_h)

    elif topo_name == "T3_fattree":
        # T3: k=4 Fat-Tree (4 core, 8 agg, 8 edge = 20 switches; 16 hosts)
        # Pods: 4 pods (0..3), each pod has 2 agg and 2 edge
        k = 4
        num_core = (k // 2) ** 2  # 4
        core_switches = [f"c{i+1}" for i in range(num_core)]
        for i, c in enumerate(core_switches):
            G.add_node(c, role="core", dialect="ovs-ofctl", dpid=f"00000000000001{i+1:02x}", interfaces=[])

        agg_switches: List[str] = []
        edge_switches: List[str] = []
        hosts: List[str] = []

        for p in range(k):
            p_agg = [f"a_{p}_{i}" for i in range(k // 2)]
            p_edge = [f"e_{p}_{i}" for i in range(k // 2)]
            for a in p_agg:
                G.add_node(a, role="agg", pod=p, dialect="ovs-ofctl", interfaces=[])
                agg_switches.append(a)
            for e in p_edge:
                G.add_node(e, role="edge", pod=p, dialect="ovs-ofctl", interfaces=[])
                edge_switches.append(e)

            # Connect agg to edge within pod
            for a in p_agg:
                for e in p_edge:
                    intf_a = f"{a}-to-{e}"
                    intf_e = f"{e}-to-{a}"
                    G.add_edge(a, e, capacity=1000.0, delay=1.0, reliability=0.999, intf_u=intf_a, intf_v=intf_e)
                    G.nodes[a]["interfaces"].append(intf_a)
                    G.nodes[e]["interfaces"].append(intf_e)

            # Connect hosts to edge
            for e_idx, e in enumerate(p_edge):
                for h_idx in range(k // 2):
                    host_id = p * 4 + e_idx * 2 + h_idx + 1
                    h_name = f"h{host_id}"
                    ip = f"10.{p}.{e_idx}.{h_idx+2}"
                    mac = f"00:00:00:{p:02x}:{e_idx:02x}:{h_idx+2:02x}"
                    G.add_node(h_name, role="host", ip=ip, mac=mac, interfaces=[])
                    hosts.append(h_name)
                    intf_e = f"{e}-to-{h_name}"
                    intf_h = f"{h_name}-eth0"
                    G.add_edge(e, h_name, capacity=1000.0, delay=1.0, reliability=0.999, intf_u=intf_e, intf_v=intf_h)
                    G.nodes[e]["interfaces"].append(intf_e)
                    G.nodes[h_name]["interfaces"].append(intf_h)

        # Connect core to agg: each core switch connects to (k) agg switches (one per pod)
        for c_idx, c in enumerate(core_switches):
            agg_offset = c_idx // (k // 2)
            for p in range(k):
                target_agg = f"a_{p}_{agg_offset}"
                intf_c = f"{c}-to-{target_agg}"
                intf_a = f"{target_agg}-to-{c}"
                G.add_edge(c, target_agg, capacity=1000.0, delay=1.0, reliability=0.999, intf_u=intf_c, intf_v=intf_a)
                G.nodes[c]["interfaces"].append(intf_c)
                G.nodes[target_agg]["interfaces"].append(intf_a)

    else:
        raise ValueError(f"Unknown topology name: {topo_name}")

    return G


def get_inventory(G: nx.Graph) -> Dict[str, Any]:
    """Extract canonical device and interface inventory dictionary for verifier checking."""
    inventory = {
        "switches": [n for n, d in G.nodes(data=True) if d.get("role") in ("root", "agg", "edge", "core", "backbone")],
        "hosts": [n for n, d in G.nodes(data=True) if d.get("role") == "host"],
        "interfaces": {n: G.nodes[n].get("interfaces", []) for n in G.nodes},
        "links": [
            {
                "u": u,
                "v": v,
                "capacity": d.get("capacity", 100.0),
                "delay": d.get("delay", 2.0),
                "reliability": d.get("reliability", 0.999),
                "intf_u": d.get("intf_u", ""),
                "intf_v": d.get("intf_v", ""),
            }
            for u, v, d in G.edges(data=True)
        ],
    }
    return inventory


def build_mininet(topo_name: str = "T1_tree") -> Any:
    """Build a Mininet network instance for the given topology.

    Requires Linux / WSL with root privileges.
    """
    try:
        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController
        from mininet.link import TCLink
    except ImportError as e:
        raise RuntimeError(f"Mininet is not installed or unavailable in this environment: {e}")

    G = build_networkx(topo_name)
    net = Mininet(switch=OVSSwitch, link=TCLink, controller=RemoteController, autoSetMacs=True, autoStaticArp=True)

    # Add controller placeholder (or OVS standalone)
    c0 = net.addController("c0", ip="127.0.0.1", port=6653)

    mn_nodes: Dict[str, Any] = {}
    for node, data in G.nodes(data=True):
        if data.get("role") == "host":
            mn_nodes[node] = net.addHost(node, ip=data.get("ip", "10.0.0.1"), mac=data.get("mac", None))
        else:
            mn_nodes[node] = net.addSwitch(node, protocols="OpenFlow13", failMode="standalone")

    for u, v, data in G.edges(data=True):
        cap = int(data.get("capacity", 100))
        del_ms = f"{int(data.get('delay', 2))}ms"
        net.addLink(mn_nodes[u], mn_nodes[v], bw=cap, delay=del_ms)

    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM Topology Builder & Inventory Inspector")
    parser.add_argument("--topology", type=str, default="T1_tree", choices=["T1_tree", "T2_abilene", "T3_fattree"])
    parser.add_argument("--inspect", action="store_true", help="Print summary inventory")
    args = parser.parse_args()

    G = build_networkx(args.topology)
    inv = get_inventory(G)
    print(f"Topology: {args.topology}")
    print(f"  Nodes: {G.number_of_nodes()} (Switches: {len(inv['switches'])}, Hosts: {len(inv['hosts'])})")
    print(f"  Edges: {G.number_of_edges()}")
    if args.inspect:
        print(f"  Switches: {inv['switches']}")
        print(f"  Hosts: {inv['hosts']}")


if __name__ == "__main__":
    main()

