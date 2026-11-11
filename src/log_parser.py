"""CL-ANM Log Parser.

Parses network and controller log streams using:
  1. Regex for structured known formats (OVS, Mininet, Linux kernel, Ryu).
  2. Drain3 tree-based log clustering for unstructured / unknown lines.
  3. Optional transformer embeddings (MiniLM / DistilBERT) for cluster templates.
"""

from __future__ import annotations
import argparse
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass
class ParsedLogEvent:
    raw_message: str
    event_type: str  # 'link_down', 'link_up', 'port_drop', 'controller_conn', 'unknown'
    device: Optional[str] = None
    interface: Optional[str] = None
    severity: str = "INFO"
    template: str = ""
    cluster_id: int = -1
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LogParser:
    """Hybrid regex and Drain3 log parser."""

    # Pre-compiled high-priority regex patterns for known network logs
    KNOWN_PATTERNS = [
        (re.compile(r"(?:link|port)\s+([a-zA-Z0-9_-]+)\s+changed\s+state\s+to\s+(DOWN|UP)", re.IGNORECASE), "link_state"),
        (re.compile(r"([a-zA-Z0-9_-]+):\s+link\s+(DOWN|UP)", re.IGNORECASE), "link_state"),
        (re.compile(r"drop(?:ped)?\s+packet[s]?\s+on\s+([a-zA-Z0-9_-]+)", re.IGNORECASE), "port_drop"),
        (re.compile(r"connected\s+to\s+controller\s+([0-9.:]+)", re.IGNORECASE), "controller_conn"),
        (re.compile(r"loss\s+([0-9.]+)%\s+on\s+([a-zA-Z0-9_-]+)", re.IGNORECASE), "loss_alert"),
    ]

    def __init__(self, use_embeddings: bool = False):
        self.use_embeddings = use_embeddings
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.5
        config.drain_depth = 4
        self.miner = TemplateMiner(config=config)
        self._embedder = None

    def _get_embedder(self) -> Any:
        if self._embedder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder = None
        return self._embedder

    def parse_line(self, log_line: str) -> ParsedLogEvent:
        """Parse a single log string into a structured ParsedLogEvent."""
        clean_line = log_line.strip()

        # 1. Check known regex patterns
        for pattern, ev_type in self.KNOWN_PATTERNS:
            match = pattern.search(clean_line)
            if match:
                groups = match.groups()
                intf = groups[0] if len(groups) > 0 else None
                state = groups[1].upper() if len(groups) > 1 else "UNKNOWN"
                event_name = f"link_{state.lower()}" if ev_type == "link_state" else ev_type
                sev = "ERROR" if "DOWN" in state or "drop" in ev_type else "INFO"
                return ParsedLogEvent(
                    raw_message=clean_line,
                    event_type=event_name,
                    device=intf.split("-")[0] if intf and "-" in intf else intf,
                    interface=intf,
                    severity=sev,
                    template=pattern.pattern,
                    cluster_id=0,
                )

        # 2. Fall back to Drain3 clustering
        result = self.miner.add_log_message(clean_line)
        template = result["template_mined"] if result else clean_line
        cluster_id = result["cluster_id"] if result else -1

        embedding = None
        embedder = self._get_embedder()
        if embedder is not None:
            embedding = embedder.encode(template).tolist()

        return ParsedLogEvent(
            raw_message=clean_line,
            event_type="unstructured",
            severity="WARNING" if "fail" in clean_line.lower() or "error" in clean_line.lower() else "INFO",
            template=template,
            cluster_id=cluster_id,
            embedding=embedding,
        )

    def parse_batch(self, lines: List[str]) -> List[ParsedLogEvent]:
        """Parse multiple log lines in batch."""
        return [self.parse_line(l) for l in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM Log Parser CLI")
    parser.add_argument("--logs", nargs="+", default=[
        "ovs|0001|bridge|INFO|link s1-eth1 changed state to DOWN",
        "drop packets on s2-eth2 due to buffer overrun",
        "bgp neighbor 10.0.0.1 is down",
    ])
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()

    lp = LogParser(use_embeddings=args.embed)
    for l in args.logs:
        ev = lp.parse_line(l)
        print(f"[{ev.severity}] {ev.event_type} | Dev: {ev.device} | Tpl: {ev.template}")


if __name__ == "__main__":
    main()

