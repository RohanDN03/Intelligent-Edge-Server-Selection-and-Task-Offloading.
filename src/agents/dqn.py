from collections import deque
import random

import numpy as np
import torch
from torch import nn


class DuelingQNet(nn.Module):
    """Dueling architecture for more stable value estimation across actions."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
        self.apply(self._initialize)
        nn.init.uniform_(self.advantage[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.advantage[-1].bias)
        nn.init.uniform_(self.value[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value[-1].bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNAgent:
    """Masked n-step Double-Dueling DQN with uniform replay."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        obs_dim=60,
        action_dim=16,
        lr=2e-4,
        gamma=0.995,
        capacity=100000,
        batch_size=128,
        warmup_steps=1200,
        target_update_interval=1000,
        n_step=3,
        train_frequency=1,
        max_grad_norm=1.0,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q = DuelingQNet(obs_dim, action_dim).to(self.device)
        self.target = DuelingQNet(obs_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(lr))
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.memory = deque(maxlen=int(capacity))
        self.n_step_queue = deque()
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        self.n_step = max(1, int(n_step))
        self.train_frequency = max(1, int(train_frequency))
        self.max_grad_norm = float(max_grad_norm)
        self.transitions_seen = 0
        self.optimization_steps = 0

    def act(self, obs, mask, epsilon=0.0):
        mask = np.asarray(mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("DQN received an observation with no feasible actions")
        if random.random() < float(epsilon):
            return int(np.random.choice(valid))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q(x).squeeze(0).cpu().numpy()
        q_values[~mask] = -1e9
        return int(q_values.argmax())

    def _store_n_step_prefix(self, horizon: int):
        prefix = list(self.n_step_queue)[:horizon]
        first = prefix[0]
        reward_sum = 0.0
        discount = 1.0
        for transition in prefix:
            reward_sum += discount * float(transition[3])
            discount *= self.gamma
        last = prefix[-1]
        self.memory.append((
            first[0], first[1], first[2], reward_sum,
            last[4], last[5], last[6], horizon,
        ))
        self.n_step_queue.popleft()

    def remember(self, obs, mask, action, reward, next_obs, done, next_mask):
        transition = (
            np.asarray(obs, dtype=np.float32).copy(),
            np.asarray(mask, dtype=bool).copy(),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32).copy(),
            bool(done),
            np.asarray(next_mask, dtype=bool).copy(),
        )
        self.n_step_queue.append(transition)
        self.transitions_seen += 1

        if len(self.n_step_queue) >= self.n_step:
            self._store_n_step_prefix(self.n_step)
        if done:
            while self.n_step_queue:
                self._store_n_step_prefix(len(self.n_step_queue))

    def update(self):
        if self.transitions_seen % self.train_frequency != 0:
            return None
        if len(self.memory) < self.batch_size or self.transitions_seen < self.warmup_steps:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, _, actions, rewards, next_states, dones, next_masks, horizons = zip(*batch)
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks_t = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool, device=self.device)
        horizons_t = torch.as_tensor(horizons, dtype=torch.float32, device=self.device)

        current_q = self.q(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double-DQN: online network selects; target network evaluates.
            next_online = self.q(next_states_t).masked_fill(~next_masks_t, -1e9).argmax(1, keepdim=True)
            next_q = self.target(next_states_t).gather(1, next_online).squeeze(1)
            discounts = torch.pow(torch.full_like(horizons_t, self.gamma), horizons_t)
            target_q = rewards_t + discounts * (1.0 - dones_t) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.opt.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.q.state_dict())
        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.detach().mean().item()),
            "mean_target_q": float(target_q.detach().mean().item()),
            "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
        }

    def save(self, path):
        torch.save({
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "observation_version": 3,
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "transitions_seen": self.transitions_seen,
            "optimization_steps": self.optimization_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("Incompatible DQN checkpoint; train a new model with this project version")
        if checkpoint.get("observation_version") != 3:
            raise ValueError("Checkpoint observation semantics do not match this environment")
        self.q.load_state_dict(checkpoint["q"])
        self.target.load_state_dict(checkpoint["target"])
