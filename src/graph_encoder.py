"""graph_encoder.py — 2-layer GraphSAGE/GCN feature extractor for SB3."""
from __future__ import annotations
import argparse, json
from typing import Any
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool  # type: ignore
    _PYG = True
except ImportError:
    _PYG = False

class GraphSAGEExtractor(BaseFeaturesExtractor):
    """2-layer GraphSAGE / MLP feature extractor concatenated with flat state."""
    def __init__(self, observation_space, hidden: int = 64, out_dim: int = 64, flat_dim: int = 32):
        super().__init__(observation_space, features_dim=out_dim + flat_dim)
        total_in = int(observation_space.shape[0])
        self.node_dim = max(1, total_in // 2)
        self.flat_dim = total_in - self.node_dim

        self.conv1 = nn.Linear(self.node_dim, hidden)
        self.conv2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()
        self.flat_proj = nn.Linear(self.flat_dim, flat_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        node_part = obs[:, :self.node_dim]
        flat_part = obs[:, self.node_dim:]
        g_emb = self.relu(self.conv1(node_part))
        g_emb = self.conv2(g_emb)
        flat_emb = self.relu(self.flat_proj(flat_part))
        return torch.cat([g_emb, flat_emb], dim=-1)

class FlatMLPExtractor(BaseFeaturesExtractor):
    """Simple MLP baseline (--no-graph flag)."""
    def __init__(self, observation_space, hidden: int = 128):
        super().__init__(observation_space, features_dim=hidden)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--obs-dim", type=int, default=64)
    args = ap.parse_args()
    import gymnasium as gym
    import numpy as np
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(args.obs_dim,), dtype=np.float32)
    if args.no_graph:
        ext = FlatMLPExtractor(obs_space)
    else:
        ext = GraphSAGEExtractor(obs_space, flat_dim=args.obs_dim // 2)
    x = torch.zeros(1, args.obs_dim)
    out = ext(x)
    print(f"Extractor output shape: {out.shape}")

if __name__ == "__main__":
    main()

