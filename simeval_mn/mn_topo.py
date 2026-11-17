"""simeval_mn/mn_topo.py — Mininet topology builder (T1 tree, T2 Abilene).

Uses OVS userspace datapath to work without kernel module in WSL2.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch, Host, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel

T1_LINKS = [
    ("s1", "s2", 100, 2), ("s1", "s3", 100, 2),
    ("s2", "h1", 100, 1), ("s2", "h2", 100, 1),
    ("s3", "h3", 100, 1), ("s3", "h4", 100, 1),
]

T2_LINKS = [
    ("s1","s2",1000,5),("s1","s3",1000,4),("s2","s7",1000,3),
    ("s3","s5",1000,6),("s4","s6",1000,5),("s4","s11",1000,7),
    ("s5","s7",1000,5),("s5","s8",1000,8),("s6","s9",1000,4),
    ("s6","s11",1000,6),("s7","s10",1000,4),("s8","s10",1000,6),
    ("s9","s11",1000,3),("s10","s11",1000,5),
]
T2_HOSTS = [f"h{i}" for i in range(1, 12)]
T2_SWITCHES = [f"s{i}" for i in range(1, 12)]


class CLANMTopo(Topo):
    def __init__(self, name="T1_tree"):
        super().__init__()
        if name == "T1_tree":
            for sw in ["s1","s2","s3"]:
                self.addSwitch(sw)
            for h, ip in [("h1","10.0.0.1/8"),("h2","10.0.0.2/8"),
                          ("h3","10.0.0.3/8"),("h4","10.0.0.4/8")]:
                self.addHost(h, ip=ip)
            for u, v, bw, delay in T1_LINKS:
                self.addLink(u, v, bw=bw, delay=f"{delay}ms")
        elif name == "T2_abilene":
            for sw in T2_SWITCHES:
                self.addSwitch(sw)
            for i, h in enumerate(T2_HOSTS, 1):
                self.addHost(h, ip=f"10.0.0.{i}/8")
            for sw, h in zip(T2_SWITCHES, T2_HOSTS):
                self.addLink(sw, h, bw=1000, delay="1ms")
            for u, v, bw, delay in T2_LINKS:
                self.addLink(u, v, bw=bw, delay=f"{delay}ms")


def build_net(topo_name="T1_tree", controller=None):
    """Build and return a Mininet instance (not started). Uses OVS userspace datapath."""
    topo = CLANMTopo(topo_name)
    sw_cls = lambda name, **kw: OVSSwitch(name, datapath="user", **kw)
    if controller:
        net = Mininet(topo=topo, switch=sw_cls, link=TCLink,
                      controller=controller, autoSetMacs=True)
    else:
        net = Mininet(topo=topo, switch=sw_cls, link=TCLink,
                      controller=None, autoSetMacs=True)
    return net
