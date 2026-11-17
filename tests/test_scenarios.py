import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name):
    with open(ROOT / "configs" / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_primary_harsh_keeps_local_cpu_overloaded_but_cluster_near_capacity():
    cfg = load("harsh.yaml")
    per_user_demand = cfg["arrival_rate_per_user"] * cfg["cpu_cycles_per_task"]
    local_utilization = per_user_demand / cfg["user_cpu_hz"]
    total_demand = cfg["num_users"] * per_user_demand
    total_capacity = (
        cfg["num_users"] * cfg["user_cpu_hz"]
        + cfg["num_edges"] * cfg["edge_cpu_hz"]
        + cfg["cloud_cpu_hz"]
    )
    assert local_utilization == 6.0
    assert total_demand / total_capacity > 0.95
    assert cfg["objective_weights"]["deadline_penalty"] == 0.0


def test_extreme_profile_preserves_original_overload_stress():
    cfg = load("extreme_overload.yaml")
    demand = cfg["num_users"] * cfg["arrival_rate_per_user"] * cfg["cpu_cycles_per_task"]
    capacity = (
        cfg["num_users"] * cfg["user_cpu_hz"]
        + cfg["num_edges"] * cfg["edge_cpu_hz"]
        + cfg["cloud_cpu_hz"]
    )
    assert cfg["arrival_rate_per_user"] == 2.5
    assert demand / capacity > 1.6
