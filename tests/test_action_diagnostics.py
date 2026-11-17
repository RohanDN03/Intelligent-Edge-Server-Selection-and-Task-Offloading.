import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.action_diagnostics import summarize_actions
from src.action_space import PAIRS, DROP_ACTION


def test_action_diagnostics_include_drop_and_destination_subtask_fractions():
    counts = [0] * 16
    counts[PAIRS.index((0, 1))] = 2  # two local subtasks and two edge subtasks
    counts[PAIRS.index((4, 4))] = 1  # two cloud subtasks
    counts[DROP_ACTION] = 1
    result = summarize_actions(counts)
    assert result["drop_rate"] == 0.25
    assert result["local_subtask_fraction"] == (1.0 / 3.0)
    assert result["edge_subtask_fraction"] == (1.0 / 3.0)
    assert result["cloud_subtask_fraction"] == (1.0 / 3.0)
    assert result["action_15_count"] == 1
