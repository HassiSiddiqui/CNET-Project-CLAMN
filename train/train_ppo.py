"""train_ppo.py — Fluid env pretraining + optional Mininet fine-tune."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--env", default="fluid", choices=["fluid"])
    args = ap.parse_args()

    from src.fluid_env import FluidNetEnv
    from src.ppo_agent import train_ppo
    env = FluidNetEnv(seed=args.seed, no_graph=args.no_graph)
    train_ppo(env, total_steps=args.steps, seed=args.seed,
              no_graph=args.no_graph, checkpoint_dir=args.checkpoint_dir,
              resume=args.resume)

if __name__ == "__main__":
    main()

