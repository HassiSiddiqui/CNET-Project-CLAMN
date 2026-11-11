"""fluid_env.py — Gymnasium flow-level NetworkX simulator for fast PPO pretraining."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any
import numpy as np
import gymnasium as gym
import networkx as nx
import yaml

class FluidNetEnv(gym.Env):
    """
    Flow-level NetworkX simulator.

    State s_t = [g_t(64), u_t, d_t, l_t, q_t, c_t, m_t, p_t]
    where scalars u_t..p_t are per-link utilisation, delay, loss,
    queue depth, controller CPU, staleness mask, fault posterior.

    Actions: discrete index over k=3 shortest paths for each active flow.
    Reward: R = a*Rrec - b*Rdelay - g*Rloss - d*Rutil - e*Rchurn
    """
    metadata = {"render_modes": []}

    def __init__(self, graph: nx.Graph | None = None, config_path: str = "config/ppo.yaml",
                 k: int = 3, seed: int = 42, episode_len: int = 90, no_graph: bool = False):
        super().__init__()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.episode_len = episode_len
        self.k = k
        self.no_graph = no_graph

        # Load reward weights
        cfg = {}
        if Path(config_path).exists():
            cfg = yaml.safe_load(Path(config_path).read_text()).get("reward", {})
        self.wa = cfg.get("alpha", 1.0)
        self.wb = cfg.get("beta", 0.5)
        self.wg = cfg.get("gamma", 0.5)
        self.wd = cfg.get("delta", 0.3)
        self.we = cfg.get("epsilon", 0.2)

        if graph is None:
            G = nx.balanced_tree(r=2, h=2)
            # label nodes as strings
            self.graph = nx.relabel_nodes(G, {n: f"s{n}" for n in G.nodes()})
        else:
            self.graph = graph

        self.nodes = sorted(self.graph.nodes())
        self.n_nodes = len(self.nodes)
        self.n_links = self.graph.number_of_edges()

        # State dim: graph embed (64 stub) + 7 scalar features per link (truncated to fixed)
        self.scalar_dim = 7
        self.graph_dim = 0 if no_graph else 64
        obs_dim = self.graph_dim + self.scalar_dim * min(self.n_links, 8)
        self.obs_dim = obs_dim if obs_dim > 0 else 1

        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32)
        # Action: choose one of k paths for the primary flow
        self.action_space = gym.spaces.Discrete(k)

        self._step = 0
        self._prev_path_idx = 0
        self._link_util = np.zeros(self.n_links, dtype=np.float32)
        self._fault_active = False

    def _get_obs(self) -> np.ndarray:
        g_emb = np.zeros(self.graph_dim, dtype=np.float32)
        scalars = np.tile(
            np.array([
                np.mean(self._link_util),          # u_t
                0.01 * self._step,                 # d_t (simulated delay growth)
                float(self._fault_active) * 0.1,   # l_t loss
                np.max(self._link_util),            # q_t queue depth proxy
                0.3,                               # c_t controller CPU (stub)
                0.0,                               # m_t staleness mask
                float(self._fault_active) * 0.8,   # p_t fault posterior
            ], dtype=np.float32),
            max(1, self.obs_dim // 7),
        )[:self.obs_dim - self.graph_dim]
        return np.concatenate([g_emb, scalars]).astype(np.float32)[:self.obs_dim]

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._prev_path_idx = 0
        self._link_util = self.np_rng.uniform(0.1, 0.4, size=self.n_links).astype(np.float32)
        self._fault_active = False
        return self._get_obs(), {}

    def step(self, action: int):
        self._step += 1
        # Fault onset at step 20
        if self._step == 20:
            self._fault_active = True
            self._link_util[0] = 1.0  # primary link overloaded

        # Apply path-weight action
        churn = float(action != self._prev_path_idx)
        self._prev_path_idx = action

        # Update utilisation: shift load based on action
        relief = 0.3 * (action / max(self.k - 1, 1))
        if self._fault_active:
            self._link_util[0] = max(0.0, self._link_util[0] - relief)

        # Reward
        recovery = 1.0 - float(self._fault_active) * self._link_util[0]
        delay = 0.01 * self._step
        loss = float(self._fault_active) * self._link_util[0] * 0.1
        util = float(np.mean(self._link_util))
        reward = (self.wa * recovery - self.wb * delay
                  - self.wg * loss - self.wd * util - self.we * churn)

        terminated = self._step >= self.episode_len
        return self._get_obs(), float(reward), terminated, False, {}

    def render(self): pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    env = FluidNetEnv(seed=args.seed, no_graph=args.no_graph)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, done, _, _ = env.step(action)
            total_r += r
        print(f"Episode {ep+1}: total_reward={total_r:.3f}")

if __name__ == "__main__":
    main()

