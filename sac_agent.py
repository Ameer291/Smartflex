"""
Soft Actor-Critic (SAC) agent for continuous control.

Implements the maximum-entropy actor-critic algorithm from
Haarnoja et al. (2018), with automatic entropy coefficient tuning
from the follow-up Haarnoja et al. (2019) paper.

Components:
  - Tanh-squashed Gaussian policy (stochastic, outputs in [-1, 1])
  - Twin Q-networks with Polyak-averaged targets (clipped double-Q)
  - Automatic alpha tuning so entropy targets -|A|
  - Uniform replay buffer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """
    Standard MLP: in -> 256 -> 256 -> 128 -> out, ReLU activations.
    Orthogonal init with gain sqrt(2) per Andrychowicz et al. (2020).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of (s, a, r, s', done) transitions.
    Uses pre-allocated NumPy arrays for memory efficiency.
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.pos = 0
        self.full = False

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos] = s2
        self.dones[self.pos] = d
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.tensor(self.states[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device).unsqueeze(-1),
            torch.tensor(self.next_states[idx], device=device),
            torch.tensor(self.dones[idx], device=device).unsqueeze(-1),
        )

    def __len__(self):
        return self.capacity if self.full else self.pos


class SACAgent:
    """
    SAC agent with automatic entropy tuning.

    Defaults follow the original paper: lr=3e-4, gamma=0.99, tau=0.005.
    Actions are in [-1, 1] after tanh squashing.
    """

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, tau=0.005,
                 init_alpha=0.2, auto_alpha=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # Actor outputs mean + log_std for each action dim
        self.actor = MLP(state_dim, 2 * action_dim).to(device)

        # Twin Q-networks with targets (clipped double-Q)
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Small weight decay helps stability in long training runs
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=lr, weight_decay=1e-5
        )
        self.q1_opt = torch.optim.Adam(
            self.q1.parameters(), lr=lr, weight_decay=1e-5
        )
        self.q2_opt = torch.optim.Adam(
            self.q2.parameters(), lr=lr, weight_decay=1e-5
        )

        # Auto-tuned alpha. Target entropy = -|A| per SAC paper heuristic.
        # Optimising log(alpha) instead of alpha for numerical stability.
        if self.auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                device=device, requires_grad=True
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha

        self.update_count = 0

    @torch.no_grad()
    def get_action(self, state, eval_mode=False):
        """
        Pick an action for a single state. In eval_mode returns the mean
        action (deterministic); otherwise samples from the policy.
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mean_logstd = self.actor(state_t)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)

        if eval_mode:
            z = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()

        action = torch.tanh(z)
        return action.cpu().numpy()[0]

    def _sample_action_and_log_prob(self, state_batch):
        """
        Sample actions for a batch and return their log-probabilities,
        including the tanh squashing correction:
            log pi(a|s) = log mu(u|s) - sum log(1 - tanh(u)^2)
        """
        mean_logstd = self.actor(state_batch)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def update(self, buffer: ReplayBuffer, batch_size: int):
        """One SAC update: Q-nets, policy, alpha, then soft target update."""
        if len(buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Symmetric reward clipping — stops occasional extreme rewards from
        # destabilising the value function
        rewards = torch.clamp(rewards, -500, 500)

        # --- Q-network update ---
        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action_and_log_prob(next_states)
            next_sa = torch.cat([next_states, next_actions], dim=-1)
            q1_next = self.q1_target(next_sa)
            q2_next = self.q2_target(next_sa)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs

            # Continuing task: don't mask by (1 - done). Day boundaries
            # are truncations not terminations, so value keeps propagating.
            target_q = rewards + self.gamma * q_next

        sa = torch.cat([states, actions], dim=-1)
        q1_pred = self.q1(sa)
        q2_pred = self.q2(sa)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # --- Policy update ---
        new_actions, log_probs = self._sample_action_and_log_prob(states)
        sa_new = torch.cat([states, new_actions], dim=-1)
        q1_new = self.q1(sa_new)
        q2_new = self.q2(sa_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # --- Alpha update (auto-tuning) ---
        alpha_loss_val = 0.0
        if self.auto_alpha:
            # Push alpha up when entropy is too low, down when too high,
            # until policy entropy matches target_entropy.
            alpha_loss = -(self.log_alpha.exp() * (
                log_probs.detach() + self.target_entropy
            )).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # --- Soft target update (Polyak averaging) ---
        for target_param, param in zip(
            self.q1_target.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.q2_target.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.update_count += 1

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val,
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
            "log_prob_mean": log_probs.mean().item(),
        }
