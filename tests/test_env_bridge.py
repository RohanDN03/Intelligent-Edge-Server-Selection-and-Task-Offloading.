import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env.ns3_env import Ns3OffloadingEnv

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "configs/harsh.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_env_command_passes_harsh_and_objective_settings(monkeypatch):
    env = Ns3OffloadingEnv(load_config(), seed=19)
    monkeypatch.setattr(env, "_binary", lambda: "/mock/ns3")
    command = env._command()
    for expected in (
        "--userBattery=1000",
        "--edgeBattery=1500",
        "--userMemoryMb=512",
        "--edgeMemoryMb=2048",
        "--cloudMemoryMb=32768",
        "--cpuCoeff=1e-29",
        "--pathLossExponent=2.8",
        "--minLinkRateMbps=0.5",
        "--shadowingStdDb=4.0",
        "--latencyWeight=0.6",
        "--energyWeight=0.2",
        "--imbalanceWeight=0.2",
        "--dropLatencyS=10.0",
    ):
        assert expected in command


def test_env_rejects_unsupported_topology():
    cfg = load_config()
    cfg["num_edges"] = 4
    try:
        Ns3OffloadingEnv(cfg)
    except ValueError as exc:
        assert "num_edges" in str(exc)
    else:
        raise AssertionError("Unsupported topology should fail fast")


def test_env_synchronous_protocol_round_trip(tmp_path, monkeypatch):
    fake = tmp_path / "fake_ns3_backend.py"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('READY 1', flush=True)\n"
        "print('OBS ' + ' '.join(['0'] * 60) + ' MASK ' + ' '.join(['1'] + ['0'] * 14 + ['1']), flush=True)\n"
        "action = sys.stdin.readline().strip()\n"
        "assert action == 'ACTION 0', action\n"
        "print('STEP -1.0 1.0 0.1 1 1', flush=True)\n"
        "print('SUMMARY 1 1.0 0.1 1.0 1.0', flush=True)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("NS3_UAV_BINARY", str(fake))
    env = Ns3OffloadingEnv(load_config(), seed=9)
    try:
        obs, mask = env.reset()
        assert obs.shape == (60,)
        assert mask.shape == (16,)
        assert mask[0] and mask[-1]
        _, reward, done, truncated, info, next_mask = env.step(0)
        assert reward == -1.0
        assert done and not truncated
        assert info["tasks"] == 1
        assert info["mean_objective"] == 1.0
        assert next_mask[-1]
    finally:
        env.close()
