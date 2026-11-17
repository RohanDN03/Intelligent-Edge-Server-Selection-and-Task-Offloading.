import os
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.env.ns3_env import Ns3OffloadingEnv


def main():
    binary = os.environ.get("NS3_UAV_BINARY")
    if not binary or not Path(binary).is_file():
        print("FAIL: NS3_UAV_BINARY is not set to an existing executable.")
        return 2
    if not os.access(binary, os.X_OK):
        print(f"FAIL: ns-3 executable is not executable: {binary}")
        return 2

    with open(ROOT / "configs/harsh.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    smoke_cfg = dict(cfg)
    smoke_cfg["episode_duration_s"] = 0.1
    smoke_cfg["arrival_rate_per_user"] = 0.0  # C++ guarantees one minimum task.

    env = Ns3OffloadingEnv(smoke_cfg, seed=123)
    try:
        obs, mask = env.reset()
        if obs.shape != (60,) or mask.shape != (16,) or not mask.any():
            raise RuntimeError(f"Invalid initial observation/mask: {obs.shape}, {mask.shape}")
        if not np.isfinite(obs).all() or float(obs.min()) < 0.0 or float(obs.max()) > 1.0:
            raise RuntimeError("Observation values must be finite and normalized to [0, 1]")
        action = int(next(i for i, valid in enumerate(mask) if valid))
        _, _, done, _, info, _ = env.step(action)
        if not done or info.get("tasks") != 1:
            raise RuntimeError(f"Smoke episode did not terminate correctly: done={done}, info={info}")
        print(f"PASS: ns-3 executable exists: {binary}")
        print("PASS: Python<->ns-3 protocol: READY, OBS/MASK, ACTION, STEP, SUMMARY")
        print("PASS: observation dimension=60; action mask dimension=16; observations bounded [0,1]")
        print(f"PASS: one-task smoke episode completed; objective={info['mean_objective']:.6f}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
