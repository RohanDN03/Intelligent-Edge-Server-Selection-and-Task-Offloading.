import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.objective import ObjectiveConfig, objective
from src.action_space import ACTION_DIM, PAIRS, decode_action


def test_weights_sum_to_one_and_no_deadline_term():
    cfg = ObjectiveConfig()
    assert cfg.latency_weight == 0.60
    assert cfg.energy_weight == 0.20
    assert cfg.imbalance_weight == 0.20
    assert objective(2, 10, 1, cfg) == pytest.approx(1.6)


def test_action_space():
    assert len(PAIRS) == 15
    assert ACTION_DIM == 16
    assert decode_action(15) is None
