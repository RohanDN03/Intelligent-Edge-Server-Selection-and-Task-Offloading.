import os
import subprocess
from pathlib import Path

import numpy as np

from ..observation import normalize_observation


class Ns3OffloadingEnv:
    """Python adapter for the native ns-3 task-level offloading episode runner.

    This deliberately uses a small synchronous stdin/stdout protocol. It is
    Gym-like rather than a full Gymnasium Env: reset returns (obs, mask), and
    step returns (obs, reward, terminated, truncated, info, next_mask).
    """

    EXPECTED_USERS = 10
    EXPECTED_EDGES = 3
    EXPECTED_CLOUDS = 1
    EXPECTED_SUBTASKS = 2

    def __init__(self, config, seed=7):
        self.cfg = config
        self.seed = int(seed)
        self.proc = None
        self.obs_dim = int(config.get("observation_dim", 60))
        self.action_dim = int(config.get("action_dim", 16))
        self._validate_config()

    def _validate_config(self):
        """Fail loudly when YAML describes a topology this C++ runner cannot use."""
        expected = {
            "num_users": self.EXPECTED_USERS,
            "num_edges": self.EXPECTED_EDGES,
            "num_clouds": self.EXPECTED_CLOUDS,
            "num_subtasks": self.EXPECTED_SUBTASKS,
            "observation_dim": 60,
            "action_dim": 16,
        }
        for key, value in expected.items():
            if int(self.cfg.get(key, value)) != value:
                raise ValueError(
                    f"{key}={self.cfg.get(key)!r} is not supported by this ns-3 scenario; "
                    f"expected {value}. Update the C++ topology/action interface before changing it."
                )

        if self.cfg.get("wifi_standard", "802.11g") != "802.11g":
            raise ValueError("This scenario currently supports only wifi_standard: 802.11g")
        allowed_rates = {6, 9, 12, 18, 24, 36, 48, 54}
        rate = int(round(float(self.cfg.get("wifi_phy_rate_mbps", 6.0))))
        if rate not in allowed_rates:
            raise ValueError(
                "wifi_phy_rate_mbps must be one of 6, 9, 12, 18, 24, 36, 48, or 54 for 802.11g"
            )
        if self.cfg.get("objective_weights", {}).get("deadline_penalty", 0.0) != 0.0:
            raise ValueError("deadline_penalty must remain 0.0; deadline is a reporting metric only")

        weights = self.cfg["objective_weights"]
        total_weight = float(weights["latency"]) + float(weights["energy"]) + float(weights["imbalance"])
        if any(float(weights[key]) < 0.0 for key in ("latency", "energy", "imbalance")):
            raise ValueError("objective weights cannot be negative")
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("latency, energy, and imbalance objective weights must sum to 1")

        required_positive = (
            "episode_duration_s", "cpu_cycles_per_task", "deadline_s", "user_cpu_hz",
            "edge_cpu_hz", "cloud_cpu_hz", "user_battery_j", "edge_battery_j",
            "cloud_battery_j", "user_memory_mb", "edge_memory_mb", "cloud_memory_mb",
            "memory_mb_per_task", "cpu_energy_coefficient", "latency_reference_s",
            "energy_reference_j", "imbalance_reference", "drop_latency_s",
            "input_size_mb", "output_size_mb", "radio_tx_w", "radio_rx_w",
            "per_link_bandwidth_mbps", "wifi_phy_rate_mbps", "path_loss_exponent", "min_link_rate_mbps",
        )
        for key in required_positive:
            if float(self.cfg[key]) <= 0.0:
                raise ValueError(f"{key} must be positive")
        if float(self.cfg["arrival_rate_per_user"]) < 0.0:
            raise ValueError("arrival_rate_per_user cannot be negative")
        if float(self.cfg["shadowing_std_db"]) < 0.0:
            raise ValueError("shadowing_std_db cannot be negative")

    def _binary(self):
        path = os.environ.get("NS3_UAV_BINARY")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                "Set NS3_UAV_BINARY to the built ns-3 uav-offloading executable; see README.md"
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(f"NS3_UAV_BINARY is not executable: {path}")
        return path

    def _command(self):
        c = self.cfg
        weights = c["objective_weights"]
        args = {
            "seed": self.seed,
            "duration": c["episode_duration_s"],
            "arrival": c["arrival_rate_per_user"],
            "cycles": c["cpu_cycles_per_task"],
            "deadline": c["deadline_s"],
            "inputMb": c["input_size_mb"],
            "outputMb": c["output_size_mb"],
            "taskMemoryMb": c["memory_mb_per_task"],
            "txW": c["radio_tx_w"],
            "rxW": c["radio_rx_w"],
            "bwMbps": c["per_link_bandwidth_mbps"],
            "minLinkRateMbps": c["min_link_rate_mbps"],
            "wifiRateMbps": c["wifi_phy_rate_mbps"],
            "userCpu": c["user_cpu_hz"],
            "edgeCpu": c["edge_cpu_hz"],
            "cloudCpu": c["cloud_cpu_hz"],
            "userBattery": c["user_battery_j"],
            "edgeBattery": c["edge_battery_j"],
            "cloudBattery": c["cloud_battery_j"],
            "userMemoryMb": c["user_memory_mb"],
            "edgeMemoryMb": c["edge_memory_mb"],
            "cloudMemoryMb": c["cloud_memory_mb"],
            "cpuCoeff": c["cpu_energy_coefficient"],
            "pathLossExponent": c["path_loss_exponent"],
            "shadowingStdDb": c["shadowing_std_db"],
            "dropLatencyS": c["drop_latency_s"],
            "latencyRefS": c["latency_reference_s"],
            "energyRefJ": c["energy_reference_j"],
            "imbalanceRef": c["imbalance_reference"],
            "latencyWeight": weights["latency"],
            "energyWeight": weights["energy"],
            "imbalanceWeight": weights["imbalance"],
        }
        return [self._binary(), *(f"--{key}={value}" for key, value in args.items())]

    def reset(self, seed=None):
        self.close()
        if seed is not None:
            self.seed = int(seed)

        self.proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self.proc.stdout.readline().strip()
        if not line.startswith("READY "):
            stderr = self.proc.stderr.read()
            return_code = self.proc.poll()
            self.close()
            raise RuntimeError(
                f"ns-3 backend failed to start (exit={return_code}): {line}; stderr={stderr}"
            )

        self.total_tasks = int(line.split()[1])
        self.steps = 0
        return self._read_obs()

    def _read_obs(self):
        line = self.proc.stdout.readline().strip()
        if not line.startswith("OBS "):
            stderr = self.proc.stderr.read() if self.proc.poll() is not None else ""
            raise RuntimeError(f"Expected OBS from ns-3 backend, got: {line}; stderr={stderr}")
        left, right = line.split(" MASK ", maxsplit=1)
        raw_obs = np.asarray([float(x) for x in left.split()[1:]], dtype=np.float32)
        obs = normalize_observation(raw_obs, self.cfg)
        mask = np.asarray([bool(int(x)) for x in right.split()], dtype=bool)
        if obs.shape != (self.obs_dim,) or mask.shape != (self.action_dim,):
            raise RuntimeError(f"Bad backend shapes: {obs.shape}, {mask.shape}")
        if not mask.any():
            raise RuntimeError("ns-3 returned an action mask with no feasible action")
        return obs, mask

    def step(self, action):
        if self.proc is None:
            raise RuntimeError("Call reset() before step()")
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"Invalid action index {action}")

        self.proc.stdin.write(f"ACTION {int(action)}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if not line.startswith("STEP "):
            stderr = self.proc.stderr.read() if self.proc.poll() is not None else ""
            raise RuntimeError(f"Expected STEP from ns-3 backend, got: {line}; stderr={stderr}")

        _, reward, latency, energy, success, done = line.split()
        self.steps += 1
        info = {
            "latency_s": float(latency),
            "energy_j": float(energy),
            "success": bool(int(success)),
        }
        terminated = bool(int(done))
        if terminated:
            summary_line = self.proc.stdout.readline().strip()
            summary = summary_line.split()
            if len(summary) != 6 or summary[0] != "SUMMARY":
                raise RuntimeError(f"Expected SUMMARY after final STEP, got: {summary_line}")
            info.update(
                tasks=int(summary[1]),
                mean_latency_s=float(summary[2]),
                mean_energy_j=float(summary[3]),
                success_rate=float(summary[4]),
                mean_objective=float(summary[5]),
            )
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            mask = np.zeros(self.action_dim, dtype=bool)
            mask[-1] = True
        else:
            obs, mask = self._read_obs()
        return obs, float(reward), terminated, False, info, mask

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    self.proc.kill()
            self.proc = None
