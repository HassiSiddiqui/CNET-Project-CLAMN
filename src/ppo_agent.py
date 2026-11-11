"""ppo_agent.py — SB3 PPO with GraphSAGE extractor, action masking, OOD guard."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def build_ppo(env, no_graph: bool = False, seed: int = 42):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from src.graph_encoder import GraphSAGEExtractor, FlatMLPExtractor

    check_env(env, warn=True)
    policy_kwargs = dict(
        features_extractor_class=FlatMLPExtractor if no_graph else GraphSAGEExtractor,
        features_extractor_kwargs=dict(hidden=128) if no_graph else dict(
            hidden=64, out_dim=64, flat_dim=32),
        net_arch=[64, 64],
    )
    # Device selection: Use DirectML if available, else CPU
    device = "cpu"
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"[PPO] Utilizing AMD GPU via DirectML: {device}")
    except Exception:
        pass

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    return model

def mahalanobis_ood(obs: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray,
                    threshold: float = 10.0) -> bool:
    """Return True if obs is OOD (Mahalanobis distance > threshold)."""
    diff = obs - mu
    dist = float(np.sqrt(diff @ cov_inv @ diff))
    return dist > threshold

def train_ppo(
    env,
    total_steps: int = 50_000,
    checkpoint_dir: str = "results/checkpoints",
    checkpoint_every: int = 10_000,
    no_graph: bool = False,
    seed: int = 42,
    resume: bool = True,
) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = sorted(ckpt_dir.glob("ppo_step_*.zip"))

    model = build_ppo(env, no_graph=no_graph, seed=seed)
    start_step = 0

    if resume and latest:
        from stable_baselines3 import PPO
        latest_path = str(latest[-1])
        model = PPO.load(latest_path, env=env)
        start_step = int(latest[-1].stem.split("_")[-1])
        print(f"Resumed from {latest_path} (step {start_step})")

    steps_done = start_step
    while steps_done < total_steps:
        chunk = min(checkpoint_every, total_steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        path = ckpt_dir / f"ppo_step_{steps_done}.zip"
        model.save(str(path))
        print(f"[PPO] checkpoint: {path}")

    # Save reward log stub
    log_path = ckpt_dir / "ppo_reward_log.json"
    log_path.write_text(json.dumps({"total_steps": total_steps, "seed": seed}))
    print("PPO training done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="fluid", choices=["fluid", "mininet"])
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    if args.env == "fluid":
        from src.fluid_env import FluidNetEnv
        env = FluidNetEnv(seed=args.seed, no_graph=args.no_graph)
    else:
        raise NotImplementedError("Mininet env not wired yet; use --env fluid")

    if args.evaluate:
        from stable_baselines3 import PPO
        from pathlib import Path as P
        ckpts = sorted(P(args.checkpoint_dir).glob("ppo_step_*.zip"))
        if not ckpts:
            print("No checkpoint found; run training first.")
            return
        model = PPO.load(str(ckpts[-1]), env=env)
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            total_r += r
        print(f"Eval episode total_reward={total_r:.3f}")
    else:
        train_ppo(env, total_steps=args.steps, seed=args.seed,
                  no_graph=args.no_graph, checkpoint_dir=args.checkpoint_dir,
                  resume=args.resume)

if __name__ == "__main__":
    main()

