"""tests/test_fluid_env.py — Smoke tests for FluidNetEnv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.fluid_env import FluidNetEnv

def test_env_step():
    env = FluidNetEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    obs2, r, done, trunc, info2 = env.step(0)
    assert obs2.shape == (env.obs_dim,)
    assert isinstance(r, float)

def test_env_full_episode():
    env = FluidNetEnv(seed=42, episode_len=10)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        obs, r, done, trunc, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 10

def test_no_graph_flag():
    env = FluidNetEnv(seed=42, no_graph=True)
    obs, _ = env.reset()
    assert obs.shape[0] > 0

if __name__ == "__main__":
    test_env_step()
    test_env_full_episode()
    test_no_graph_flag()
    print("All fluid_env tests passed.")

