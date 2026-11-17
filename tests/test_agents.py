import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent


def test_dqn_masked_action_and_update():
    agent = DQNAgent(batch_size=2, warmup_steps=2, target_update_interval=1, train_frequency=1, n_step=1, device="cpu")
    obs = np.zeros(60, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[3] = True
    assert agent.act(obs, mask) == 3
    for _ in range(2):
        agent.remember(obs, mask, 3, -0.1, obs, True, mask)
    assert agent.update() is not None


def test_dqn_honours_replay_warmup():
    agent = DQNAgent(batch_size=1, warmup_steps=2, train_frequency=1, n_step=1, device="cpu")
    obs = np.zeros(60, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[15] = True
    agent.remember(obs, mask, 15, -0.01, obs, True, mask)
    assert agent.update() is None
    agent.remember(obs, mask, 15, -0.01, obs, True, mask)
    assert agent.update() is not None


def test_ppo_masked_action_and_diagnostics_are_finite():
    agent = PPOAgent(device="cpu")
    obs = np.zeros(60, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[4] = True
    action, logp, value = agent.act(obs, mask)
    assert action == 4
    for _ in range(4):
        agent.store(obs, mask, action, logp, value, -0.01, True)
    metrics = agent.update(epochs=1, minibatch=2)
    assert {"loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())


def test_dqn_n_step_returns_flush_on_terminal():
    agent = DQNAgent(batch_size=1, warmup_steps=99, n_step=3, train_frequency=1, device="cpu")
    obs = np.zeros(60, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[0] = True
    for reward, done in ((-1.0, False), (-2.0, True)):
        agent.remember(obs, mask, 0, reward, obs, done, mask)
    assert len(agent.memory) == 2
    first = agent.memory[0]
    assert first[3] == -1.0 + agent.gamma * -2.0
    assert first[5] is True
    assert first[7] == 2
