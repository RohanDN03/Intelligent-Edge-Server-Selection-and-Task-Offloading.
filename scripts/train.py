import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action_diagnostics import summarize_actions
from src.env.ns3_env import Ns3OffloadingEnv
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def linear_epsilon(step, start, end, decay_steps):
    fraction = min(1.0, max(0.0, float(step) / max(1.0, float(decay_steps))))
    return float(start + fraction * (end - start))


def main():
    parser = argparse.ArgumentParser(description="Train DQN or PPO through the native ns-3 bridge")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default=str(ROOT / "configs/harsh.yaml"))
    parser.add_argument("--run-name", default="", help="Optional suffix to keep experiment outputs separate")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    seed_everything(args.seed)
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    training_cfg = cfg["training"]
    algo_cfg = training_cfg[args.algo]
    reward_scale = float(training_cfg.get("reward_scale", 0.01))
    if reward_scale <= 0.0:
        raise ValueError("training.reward_scale must be positive")

    if args.algo == "dqn":
        agent = DQNAgent(
            obs_dim=cfg["observation_dim"],
            action_dim=cfg["action_dim"],
            lr=algo_cfg["learning_rate"],
            gamma=training_cfg["gamma"],
            capacity=algo_cfg["replay_capacity"],
            batch_size=algo_cfg["batch_size"],
            warmup_steps=algo_cfg["warmup_steps"],
            target_update_interval=algo_cfg["target_update_interval"],
            n_step=algo_cfg["n_step"],
            train_frequency=algo_cfg["train_frequency"],
            max_grad_norm=algo_cfg["max_grad_norm"],
        )
    else:
        agent = PPOAgent(
            obs_dim=cfg["observation_dim"],
            action_dim=cfg["action_dim"],
            lr=algo_cfg["learning_rate"],
            gamma=training_cfg["gamma"],
            gae_lambda=algo_cfg["gae_lambda"],
            clip=algo_cfg["clip"],
            entropy_coef=algo_cfg["entropy_coef"],
            value_coef=algo_cfg["value_coef"],
            max_grad_norm=algo_cfg["max_grad_norm"],
            target_kl=algo_cfg["target_kl"],
            value_clip=algo_cfg["value_clip"],
        )

    checkpoint_dir = ROOT / "checkpoints"
    result_dir = ROOT / "results"
    checkpoint_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    stem = args.algo + (f"_{args.run_name}" if args.run_name else "")
    rows = []
    total_env_steps = 0

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        env = Ns3OffloadingEnv(cfg, episode_seed)
        obs, mask = env.reset()
        raw_rewards = []
        dqn_stats = []
        ppo_stats = {}
        action_counts = np.zeros(cfg["action_dim"], dtype=np.int64)
        epsilon_values = []
        done = False
        final_info = {}

        try:
            while not done:
                if args.algo == "dqn":
                    epsilon = linear_epsilon(
                        total_env_steps,
                        algo_cfg["epsilon_start"],
                        algo_cfg["epsilon_end"],
                        algo_cfg["epsilon_decay_steps"],
                    )
                    epsilon_values.append(epsilon)
                    action = agent.act(obs, mask, epsilon=epsilon)
                    logp = value = None
                else:
                    action, logp, value = agent.act(obs, mask)

                next_obs, reward, done, _, info, next_mask = env.step(action)
                total_env_steps += 1
                action_counts[action] += 1
                raw_rewards.append(reward)
                scaled_reward = reward * reward_scale

                if args.algo == "dqn":
                    agent.remember(obs, mask, action, scaled_reward, next_obs, done, next_mask)
                    update_stats = agent.update()
                    if update_stats is not None:
                        dqn_stats.append(update_stats)
                else:
                    agent.store(obs, mask, action, logp, value, scaled_reward, done)
                obs, mask = next_obs, next_mask
                final_info = info

            if args.algo == "ppo":
                ppo_stats = agent.update(
                    epochs=algo_cfg["update_epochs"],
                    minibatch=algo_cfg["minibatch_size"],
                )
        finally:
            env.close()

        action_summary = summarize_actions(action_counts)
        dqn_mean = {
            key: float(np.mean([item[key] for item in dqn_stats])) if dqn_stats else 0.0
            for key in ("loss", "mean_q", "mean_target_q", "grad_norm")
        }
        epsilon_end = epsilon_values[-1] if epsilon_values else 0.0
        row = {
            "episode": episode_index + 1,
            "seed": episode_seed,
            "tasks": final_info.get("tasks", len(raw_rewards)),
            "reward_per_task_raw": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            "success_rate": final_info.get("success_rate", 0.0),
            "mean_latency_s": final_info.get("mean_latency_s", 0.0),
            "mean_energy_j": final_info.get("mean_energy_j", 0.0),
            "objective": final_info.get("mean_objective", 0.0),
            "epsilon_start": epsilon_values[0] if epsilon_values else 0.0,
            "epsilon_end": epsilon_end,
            "epsilon_mean": float(np.mean(epsilon_values)) if epsilon_values else 0.0,
            "loss": dqn_mean["loss"] if args.algo == "dqn" else ppo_stats.get("loss", 0.0),
            "mean_q": dqn_mean["mean_q"],
            "mean_target_q": dqn_mean["mean_target_q"],
            "grad_norm": dqn_mean["grad_norm"] if args.algo == "dqn" else 0.0,
            "policy_loss": ppo_stats.get("policy_loss", 0.0),
            "value_loss": ppo_stats.get("value_loss", 0.0),
            "entropy": ppo_stats.get("entropy", 0.0),
            "approx_kl": ppo_stats.get("approx_kl", 0.0),
            "clip_fraction": ppo_stats.get("clip_fraction", 0.0),
            "explained_variance": ppo_stats.get("explained_variance", 0.0),
            "ppo_updates": ppo_stats.get("updates", 0.0),
            "ppo_early_stop": ppo_stats.get("early_stop", 0.0),
            **action_summary,
        }
        rows.append(row)
        print(
            " | ".join(
                f"{key}={value:.6f}" if isinstance(value, (float, np.floating)) else f"{key}={value}"
                for key, value in row.items()
            ),
            flush=True,
        )

        agent.save(checkpoint_dir / f"{stem}_latest.pt")
        with open(result_dir / f"{stem}_training.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "algorithm": args.algo,
        "seed": args.seed,
        "episodes": args.episodes,
        "total_env_steps": total_env_steps,
        "observation_version": 3,
        "reward_scale_for_learning_only": reward_scale,
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
    }
    with open(result_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
