import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action_diagnostics import summarize_actions
from src.action_space import DROP_ACTION
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.baselines import cloud, drop, greedy, local, nearest
from src.env.ns3_env import Ns3OffloadingEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate learned policies and baselines on common ns-3 seeds")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--config", default=str(ROOT / "configs/harsh.yaml"))
    parser.add_argument("--dqn")
    parser.add_argument("--ppo")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--run-name", default="", help="Optional suffix for output files")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    agents = {}
    if not args.baselines_only:
        if args.dqn:
            dqn = DQNAgent(cfg["observation_dim"], cfg["action_dim"])
            dqn.load(args.dqn)
            agents["DQN"] = lambda obs, mask, ag=dqn: ag.act(obs, mask, epsilon=0.0)
        if args.ppo:
            ppo = PPOAgent(cfg["observation_dim"], cfg["action_dim"])
            ppo.load(args.ppo)
            agents["PPO"] = lambda obs, mask, ag=ppo: ag.act(obs, mask, deterministic=True)[0]

    policies = {
        "DROP": lambda obs, mask: drop(mask),
        "LOCAL": lambda obs, mask: local(mask),
        "CLOUD": lambda obs, mask: cloud(mask),
        "NEAREST": lambda obs, mask: nearest(obs, mask),
        "GREEDY": lambda obs, mask: greedy(obs, mask, cfg),
        **agents,
    }

    rows = []
    for episode in range(args.episodes):
        evaluation_seed = args.seed + episode
        for name, policy in policies.items():
            env = Ns3OffloadingEnv(cfg, evaluation_seed)
            obs, mask = env.reset()
            done = False
            action_counts = np.zeros(cfg["action_dim"], dtype=np.int64)
            final_info = {}
            try:
                while not done:
                    action = int(policy(obs, mask))
                    action_counts[action] += 1
                    obs, _, done, _, final_info, mask = env.step(action)
            finally:
                env.close()
            diagnostics = summarize_actions(action_counts)
            rows.append({
                "episode": episode + 1,
                "seed": evaluation_seed,
                "policy": name,
                "latency_s": final_info.get("mean_latency_s", np.nan),
                "energy_j": final_info.get("mean_energy_j", np.nan),
                "success_rate": final_info.get("success_rate", np.nan),
                "objective": final_info.get("mean_objective", np.nan),
                **diagnostics,
            })
        print(f"Finished evaluation episode {episode + 1}/{args.episodes}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = f"_{args.run_name}" if args.run_name else ""
    episode_path = out / f"evaluation_episodes{suffix}.csv"
    summary_path = out / f"evaluation_summary{suffix}.csv"
    plot_path = out / f"policy_comparison{suffix}.png"
    df.to_csv(episode_path, index=False)

    metrics = [
        "latency_s", "energy_j", "success_rate", "objective", "drop_rate",
        "local_subtask_fraction", "edge_subtask_fraction", "cloud_subtask_fraction",
    ]
    summary = df.groupby("policy")[metrics].agg(["mean", "std"])
    summary.to_csv(summary_path)
    print(summary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric, title in zip(
        axes.flat,
        ("objective", "latency_s", "energy_j", "success_rate"),
        ("Mean objective (lower is better)", "Mean latency (s)", "Mean system energy (J)", "Deadline success rate"),
    ):
        df.groupby("policy")[metric].mean().sort_values().plot(kind="bar", ax=axis, rot=25)
        axis.set_title(title)
        axis.set_xlabel("")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {episode_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
