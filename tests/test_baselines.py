import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.action_space import DROP_ACTION
from src.baselines import greedy, local, nearest

ROOT = Path(__file__).resolve().parents[1]


def test_baselines_return_feasible_actions():
    with open(ROOT / "configs/harsh.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    obs = [0.0] * 60
    # Each candidate has a positive CPU, memory, and link-rate feature.
    for candidate in range(5):
        base = 5 + candidate * 9
        obs[base] = 0.1 if candidate < 4 else 1.0
        obs[base + 3] = 0.6
        obs[base + 5] = 1.0 if candidate else 0.0
        obs[base + 6] = 1.0
    mask = [True] * 16
    assert local(mask) == 0
    assert nearest(obs, mask) in range(16)
    assert greedy(obs, mask, cfg) in range(16)
    assert greedy(obs, [False] * 15 + [True], cfg) == DROP_ACTION


def test_drop_and_cloud_baselines():
    from src.baselines import cloud, drop
    from src.action_space import PAIRS

    mask = [True] * 16
    assert drop(mask) == DROP_ACTION
    assert cloud(mask) == PAIRS.index((4, 4))
    mask[PAIRS.index((4, 4))] = False
    assert cloud(mask) == DROP_ACTION
