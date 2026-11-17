import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Separate actor and critic encoders reduce gradient interference."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.actor(self.actor_body(x)), self.critic(self.critic_body(x)).squeeze(-1)


class PPOAgent:
    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        obs_dim=60,
        action_dim=16,
        lr=1.5e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        value_clip=0.2,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(obs_dim, action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr), eps=1e-5)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip = float(clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)
        self.value_clip = float(value_clip)
        self.rollout = []

    def act(self, obs, mask, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        if not bool(m.any()):
            raise ValueError("PPO received an observation with no feasible actions")
        with torch.no_grad():
            logits, value = self.net(x)
            logits = logits.masked_fill(~m, -1e9)
            distribution = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def store(self, obs, mask, action, logp, value, reward, done):
        self.rollout.append((
            np.asarray(obs, dtype=np.float32).copy(),
            np.asarray(mask, dtype=bool).copy(),
            int(action), float(logp), float(value), float(reward), bool(done),
        ))

    def update(self, epochs=8, minibatch=128):
        if not self.rollout:
            return {key: 0.0 for key in (
                "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "explained_variance", "updates", "early_stop",
            )}

        obs, masks, actions, old_logp, values, rewards, dones = map(list, zip(*self.rollout))
        self.rollout = []
        rewards = np.asarray(rewards, dtype=np.float32)
        values_array = np.asarray(values + [0.0], dtype=np.float32)
        dones_array = np.asarray(dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_array[t + 1] * (1.0 - dones_array[t]) - values_array[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_array[t]) * gae
            advantages[t] = gae
        returns = advantages + values_array[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(values_array[:-1], dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        totals = {key: [] for key in (
            "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        )}
        early_stop = False
        update_count = 0
        n = len(actions_t)
        for _ in range(int(epochs)):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, int(minibatch)):
                batch = indices[start:start + int(minibatch)]
                logits, value_pred = self.net(obs_t[batch])
                logits = logits.masked_fill(~masks_t[batch], -1e9)
                distribution = Categorical(logits=logits)
                logp = distribution.log_prob(actions_t[batch])
                entropy = distribution.entropy().mean()

                log_ratio = torch.clamp(logp - old_logp_t[batch], -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages_t[batch]
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_t[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                value_delta = value_pred - old_values_t[batch]
                clipped_value = old_values_t[batch] + torch.clamp(
                    value_delta, -self.value_clip, self.value_clip
                )
                value_loss_unclipped = nn.functional.smooth_l1_loss(
                    value_pred, returns_t[batch], reduction="none"
                )
                value_loss_clipped = nn.functional.smooth_l1_loss(
                    clipped_value, returns_t[batch], reduction="none"
                )
                value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip).float().mean()
                for key, value in (
                    ("loss", loss), ("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("entropy", entropy), ("approx_kl", approx_kl), ("clip_fraction", clip_fraction),
                ):
                    totals[key].append(float(value.item()))
                update_count += 1
                if self.target_kl > 0.0 and float(approx_kl.item()) > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        variance = float(np.var(returns))
        explained_variance = 0.0 if variance < 1e-12 else 1.0 - float(np.var(returns - values_array[:-1])) / variance
        result = {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}
        result.update({
            "explained_variance": float(explained_variance),
            "updates": float(update_count),
            "early_stop": float(early_stop),
        })
        return result

    def save(self, path):
        torch.save({
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "observation_version": 3,
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("Incompatible PPO checkpoint; train a new model with this project version")
        if checkpoint.get("observation_version") != 3:
            raise ValueError("Checkpoint observation semantics do not match this environment")
        self.net.load_state_dict(checkpoint["net"])
