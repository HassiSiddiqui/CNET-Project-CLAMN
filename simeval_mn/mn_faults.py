"""simeval_mn/mn_faults.py — Fault injection for Mininet episodes (S1–S5).

Each inject_* function takes a running Mininet `net` and returns a dict
describing what was injected (for the controller's ground truth).
"""
import subprocess, time, random


def _tc_rate(iface, rate_kbps):
    """Shape iface to rate_kbps using tc (netem + tbf)."""
    subprocess.call(["tc", "qdisc", "del", "dev", iface, "root"],
                    stderr=subprocess.DEVNULL)
    if rate_kbps <= 0:
        subprocess.call(["tc", "qdisc", "add", "dev", iface, "root",
                         "netem", "loss", "100%"])
    else:
        subprocess.call(["tc", "qdisc", "add", "dev", iface, "root",
                         "tbf", "rate", f"{rate_kbps}kbit",
                         "burst", "32kbit", "latency", "400ms"])


def inject_link_failure(net, link_pair=None, mode="hard", rng=None):
    """S1: take a switch-switch link down (hard / degraded / flapping)."""
    rng = rng or random.Random()
    sw_links = [(u, v, d) for u, v, d in net.links
                if hasattr(u, 'defaultIntf') and hasattr(v, 'defaultIntf')
                and str(u).startswith("s") and str(v).startswith("s")]
    if not sw_links:
        return {"scenario": "S1_link_failure", "link": None, "mode": mode}
    u, v, d = rng.choice(sw_links) if link_pair is None else next(
        (l for l in sw_links if set([str(l[0]), str(l[1])]) == set(link_pair)), sw_links[0])
    intf_u = net.get(str(u)).connectionsTo(net.get(str(v)))[0][0]
    intf_v = net.get(str(u)).connectionsTo(net.get(str(v)))[0][1]
    if mode == "hard":
        intf_u.config(bw=0)
        subprocess.call(["ip", "link", "set", intf_u.name, "down"])
    elif mode == "degraded":
        _tc_rate(intf_u.name, 2000)   # degrade to 2 Mbps
    return {"scenario": "S1_link_failure", "link": (str(u), str(v)), "mode": mode,
            "intf_u": intf_u.name, "intf_v": intf_v.name}


def inject_ddos(net, victim=None, intensity="medium", rng=None):
    """S2: start hping3 flood from random hosts to victim."""
    rng = rng or random.Random()
    hosts = net.hosts
    victim_h = net.get(victim) if victim else rng.choice(hosts)
    victim_ip = victim_h.IP()
    attackers = [h for h in hosts if h != victim_h]
    n_att = min(3, len(attackers))
    att_hosts = rng.sample(attackers, n_att)
    rate = {"low": 100, "medium": 500, "high": 2000}[intensity]
    pids = []
    for h in att_hosts:
        cmd = f"hping3 --flood -p 80 --rand-source -d 1400 --rate {rate} {victim_ip} &"
        h.cmd(cmd)
        pids.append(h.name)
    return {"scenario": "S2_ddos", "victim": victim_h.name, "victim_ip": victim_ip,
            "attackers": [h.name for h in att_hosts], "intensity": intensity}


def inject_misconfig(net, switch=None, rng=None):
    """S3: install a drop rule on a mid-topology switch (simulate ACL misconfiguration)."""
    rng = rng or random.Random()
    switches = net.switches
    sw = net.get(switch) if switch else rng.choice(switches)
    # Install a low-priority drop rule (OpenFlow: match all -> drop)
    sw.cmd(f"ovs-ofctl add-flow {sw.name} priority=5,actions=drop")
    return {"scenario": "S3_misconfig", "switch": sw.name}


def inject_cascading(net, rng=None):
    """S4: fail busiest link then introduce a DDoS to trigger cascading overload."""
    fault = inject_link_failure(net, mode="hard", rng=rng)
    time.sleep(2)
    ddos = inject_ddos(net, intensity="medium", rng=rng)
    fault["cascading_ddos"] = ddos
    fault["scenario"] = "S4_cascading"
    return fault


def inject_combined(net, rng=None):
    """S5: link failure + DDoS (sequential, 5s apart)."""
    fault = inject_link_failure(net, mode="hard", rng=rng)
    time.sleep(5)
    ddos = inject_ddos(net, intensity="high", rng=rng)
    fault["combined_ddos"] = ddos
    fault["scenario"] = "S5_combined"
    return fault


def clear_faults(net):
    """Remove all injected faults: bring links up, kill hping3, remove drop flows."""
    for h in net.hosts:
        h.cmd("killall hping3 2>/dev/null; killall iperf3 2>/dev/null")
    for sw in net.switches:
        sw.cmd(f"ovs-ofctl del-flows {sw.name} priority=5 2>/dev/null")
    for link in net.links:
        try:
            u_intf = link.intf1
            subprocess.call(["ip", "link", "set", u_intf.name, "up"],
                            stderr=subprocess.DEVNULL)
        except Exception:
            pass


INJECTORS = {
    "S1_link_failure": inject_link_failure,
    "S2_ddos": inject_ddos,
    "S3_misconfig": inject_misconfig,
    "S4_cascading": inject_cascading,
    "S5_combined": inject_combined,
}
