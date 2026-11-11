"""CL-ANM Retrieval-Augmented Generation (RAG) Engine.

Maintains doc snippets for:
  - ovs-ofctl (OpenFlow 1.3 flow/meter/port manipulation)
  - Ryu REST API (flowentry/add, meterentry/add)
  - FRR vtysh (OSPF, access-list, ip route)

Computes dense embeddings (all-MiniLM-L6-v2) or lexical TF-IDF fallback,
performs top-k similarity retrieval, and flags out-of-distribution (OOD) intents.
"""

from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocSnippet:
    doc_id: str
    dialect: str
    topic: str
    content: str
    commands: List[str]


CORPUS: List[DocSnippet] = [
    DocSnippet(
        doc_id="ovs-01",
        dialect="ovs-ofctl",
        topic="forwarding",
        content="To add a standard OpenFlow 1.3 L3 unicast forwarding rule with priority p on switch s to output port out_port:\n"
                "ovs-ofctl -O OpenFlow13 add-flow <switch> priority=<p>,ip,nw_dst=<dst_ip>,actions=output:<out_port>",
        commands=["ovs-ofctl -O OpenFlow13 add-flow <switch> priority=<p>,ip,nw_dst=<dst_ip>,actions=output:<port>"],
    ),
    DocSnippet(
        doc_id="ovs-02",
        dialect="ovs-ofctl",
        topic="security_acl",
        content="To drop malicious ingress flood packets from a specific source IP on switch s:\n"
                "ovs-ofctl -O OpenFlow13 add-flow <switch> priority=40000,ip,nw_src=<src_ip>,actions=drop",
        commands=["ovs-ofctl -O OpenFlow13 add-flow <switch> priority=40000,ip,nw_src=<src_ip>,actions=drop"],
    ),
    DocSnippet(
        doc_id="ovs-03",
        dialect="ovs-ofctl",
        topic="rate_limiting",
        content="To create an OpenFlow 1.3 rate-limiting meter capping traffic to rate_kbps:\n"
                "ovs-ofctl -O OpenFlow13 add-meter <switch> meter=<meter_id>,kbps,band=type=drop,rate=<rate_kbps>\n"
                "Then bind with flow: ovs-ofctl -O OpenFlow13 add-flow <switch> priority=30000,ip,actions=meter:<meter_id>,output:<out_port>",
        commands=[
            "ovs-ofctl -O OpenFlow13 add-meter <switch> meter=<id>,kbps,band=type=drop,rate=<kbps>",
            "ovs-ofctl -O OpenFlow13 add-flow <switch> priority=30000,ip,actions=meter:<id>,output:<port>",
        ],
    ),
    DocSnippet(
        doc_id="ovs-04",
        dialect="ovs-ofctl",
        topic="port_management",
        content="To administratively shut down or enable an interface for link failure containment:\n"
                "ovs-ofctl -O OpenFlow13 mod-port <switch> <port_no> down\n"
                "ovs-ofctl -O OpenFlow13 mod-port <switch> <port_no> up",
        commands=[
            "ovs-ofctl -O OpenFlow13 mod-port <switch> <port> down",
            "ovs-ofctl -O OpenFlow13 mod-port <switch> <port> up",
        ],
    ),
    DocSnippet(
        doc_id="ryu-01",
        dialect="ryu-rest",
        topic="forwarding",
        content="Ryu REST controller POST /stats/flowentry/add payload structure:\n"
                "{\n  \"dpid\": <dpid_int>,\n  \"priority\": 1000,\n  \"match\": {\"ipv4_dst\": \"<ip>\", \"eth_type\": 2048},\n  \"actions\": [{\"type\": \"OUTPUT\", \"port\": <out_port>}]\n}",
        commands=["POST /stats/flowentry/add"],
    ),
    DocSnippet(
        doc_id="ryu-02",
        dialect="ryu-rest",
        topic="security_acl",
        content="Ryu REST controller drop rule payload structure:\n"
                "{\n  \"dpid\": <dpid_int>,\n  \"priority\": 40000,\n  \"match\": {\"ipv4_src\": \"<bad_ip>\", \"eth_type\": 2048},\n  \"actions\": []\n}",
        commands=["POST /stats/flowentry/add (empty actions for drop)"],
    ),
    DocSnippet(
        doc_id="frr-01",
        dialect="frr-vtysh",
        topic="routing_policy",
        content="FRRouting vtysh dynamic OSPF routing configuration:\n"
                "vtysh -c 'configure terminal' -c 'router ospf' -c 'network <subnet> area 0.0.0.0' -c 'exit'",
        commands=["vtysh -c 'configure terminal' -c 'router ospf' -c 'network <subnet> area 0.0.0.0'"],
    ),
    DocSnippet(
        doc_id="frr-02",
        dialect="frr-vtysh",
        topic="security_acl",
        content="FRRouting vtysh IP access-list packet filter:\n"
                "vtysh -c 'configure terminal' -c 'access-list 100 deny ip host <src_ip> any' -c 'access-list 100 permit ip any any'",
        commands=["access-list 100 deny ip host <src_ip> any"],
    ),
]


class NetworkRAG:
    """Retrieves relevant command syntax snippets and assesses query OOD distance."""

    def __init__(self, ood_threshold: float = 0.25):
        self.ood_threshold = ood_threshold
        self.corpus = CORPUS
        self._embedder = None
        self._corpus_embeddings = None

    def _init_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            corpus_texts = [f"{s.dialect} {s.topic}: {s.content}" for s in self.corpus]
            self._corpus_embeddings = self._embedder.encode(corpus_texts, convert_to_numpy=True)
        except Exception:
            self._embedder = None
            self._corpus_embeddings = None

    @staticmethod
    def _lexical_similarity(query: str, doc: str) -> float:
        """Token-based Jaccard / term-overlap similarity fallback."""
        q_words = set(re.findall(r"\w+", query.lower()))
        d_words = set(re.findall(r"\w+", doc.lower()))
        if not q_words or not d_words:
            return 0.0
        intersection = q_words.intersection(d_words)
        union = q_words.union(d_words)
        return len(intersection) / len(union)

    def retrieve(self, query: str, top_k: int = 2) -> Tuple[List[DocSnippet], List[float], bool]:
        """Returns top_k snippets, similarity scores, and is_ood flag."""
        if self._embedder is None and self._corpus_embeddings is None:
            self._init_embeddings()

        scores: List[float] = []
        if self._embedder is not None and self._corpus_embeddings is not None:
            import numpy as np
            q_emb = self._embedder.encode([query], convert_to_numpy=True)[0]
            # Cosine similarity
            dot = np.dot(self._corpus_embeddings, q_emb)
            norm_c = np.linalg.norm(self._corpus_embeddings, axis=1)
            norm_q = np.linalg.norm(q_emb)
            sims = dot / (norm_c * norm_q + 1e-6)
            scores = [float(s) for s in sims]
        else:
            # Lexical fallback
            for s in self.corpus:
                text = f"{s.dialect} {s.topic} {s.content}"
                scores.append(self._lexical_similarity(query, text))

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        top_snippets = [self.corpus[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        best_score = top_scores[0] if top_scores else 0.0
        is_ood = (best_score < self.ood_threshold)

        return top_snippets, top_scores, is_ood


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM RAG Retriever CLI")
    parser.add_argument("--query", type=str, default="How do I block UDP flood packets on an OVS switch?")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    rag = NetworkRAG()
    snippets, scores, is_ood = rag.retrieve(args.query, top_k=args.top_k)
    print(f"Query: {args.query}")
    print(f"OOD Flag: {is_ood} (Best Score: {scores[0]:.3f})")
    for s, score in zip(snippets, scores):
        print(f"\n[{s.doc_id}] ({s.dialect} / {s.topic}) (Score: {score:.3f}):")
        print(s.content)


if __name__ == "__main__":
    main()

