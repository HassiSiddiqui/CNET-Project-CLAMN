"""simeval_mn/mn_telemetry.py — Collect 1-Hz telemetry via ovs-ofctl dump-ports."""
import subprocess, re, time, json, os


def _parse_dump_ports(raw: str) -> dict:
    """Parse ovs-ofctl dump-ports output into {port_no: {rx_bytes, tx_bytes, rx_dropped, tx_dropped}}."""
    ports = {}
    for blk in re.split(r'\s+port\s+', raw)[1:]:
        m = re.match(r'(\S+)', blk)
        if not m:
            continue
        port = m.group(1).rstrip(':')
        rx_b = re.search(r'rx.*?bytes=(\d+)', blk)
        tx_b = re.search(r'tx.*?bytes=(\d+)', blk)
        rx_d = re.search(r'rx.*?dropped=(\d+)', blk)
        tx_d = re.search(r'tx.*?dropped=(\d+)', blk)
        ports[port] = {
            "rx_bytes": int(rx_b.group(1)) if rx_b else 0,
            "tx_bytes": int(tx_b.group(1)) if tx_b else 0,
            "rx_dropped": int(rx_d.group(1)) if rx_d else 0,
            "tx_dropped": int(tx_d.group(1)) if tx_d else 0,
        }
    return ports


def collect_snapshot(switches: list) -> dict:
    """Single snapshot: run dump-ports on each switch, return nested dict."""
    snap = {"t": time.time(), "switches": {}}
    for sw in switches:
        try:
            out = subprocess.check_output(
                ["ovs-ofctl", "dump-ports", sw],
                stderr=subprocess.DEVNULL, timeout=2
            ).decode()
            snap["switches"][sw] = _parse_dump_ports(out)
        except Exception:
            snap["switches"][sw] = {}
    return snap


def measure_util(prev: dict, curr: dict, links: list, dt: float) -> dict:
    """Compute per-link utilisation [0,1] from two snapshots.

    links: list of (sw, port, capacity_mbps) tuples.
    """
    util = {}
    for sw, port, cap_mbps in links:
        try:
            p_prev = prev["switches"][sw][port]
            p_curr = curr["switches"][sw][port]
            bytes_delta = (p_curr["tx_bytes"] - p_prev["tx_bytes"])
            if bytes_delta < 0:
                bytes_delta = 0
            bits_per_sec = (bytes_delta * 8) / max(dt, 1e-6)
            cap_bps = cap_mbps * 1e6
            util[(sw, port)] = min(bits_per_sec / cap_bps, 5.0)
        except (KeyError, TypeError):
            util[(sw, port)] = 0.0
    return util
