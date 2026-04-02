#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) algorithm implementation with CUDA support
Optimized for Spark DGX
"""

import math
import random
from typing import List, Tuple, Optional
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ['s', 'a', 's2', 'r', 'done'])


class RNG:
    def __init__(self, seed=1234):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def uniform(self, a=0.0, b=1.0):
        return float(np.random.uniform(a, b))

    def normal(self, mean=0.0, std=1.0):
        return float(np.random.normal(mean, std))

    def randint(self, lo, hi_inclusive):
        return int(np.random.randint(lo, hi_inclusive + 1))


class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim, lr=3e-4):
        super().__init__()
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h1, device=device)
        self.fc2 = nn.Linear(h1, h2, device=device)
        self.fc3 = nn.Linear(h2, out_dim, device=device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.FloatTensor(x).to(device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(0) if x.shape[0] == 1 else x


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_min = -5.0
        self.log_std_max = 2.0

        self.net = MLP(obs_dim, 128, 128, 2 * act_dim, lr=3e-4)

    def sample_action(self, obs: List[float], rng: RNG, deterministic: bool = False) -> Tuple[List[float], float]:
        with torch.no_grad():
            out = self.net(obs)

            if isinstance(out, torch.Tensor):
                out_np = out.cpu().numpy()
            else:
                out_np = np.array(out)

            mu = out_np[:self.act_dim] if out_np.ndim == 1 else out_np[0, :self.act_dim]
            logstd = out_np[self.act_dim:] if out_np.ndim == 1 else out_np[0, self.act_dim:]

            logstd = np.clip(logstd, self.log_std_min, self.log_std_max)
            std = np.exp(logstd)

            if deterministic:
                u = mu
            else:
                eps = np.array([rng.normal(0, 1) for _ in range(self.act_dim)])
                u = mu + std * eps

            a = np.tanh(u)

            logp = self._compute_logp(u, mu, std, a)

        return a.tolist(), float(logp)

    def _compute_logp(self, u, mu, std, a):
        var = std ** 2
        logp = -0.5 * np.log(2.0 * math.pi * var) - ((u - mu) ** 2) / (2.0 * var)
        logp = np.sum(logp)
        logp -= np.sum(np.log(np.maximum(1.0 - a * a, 1e-12)))
        return logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q = MLP(obs_dim + act_dim, 256, 256, 1, lr=3e-4)

    def forward_q(self, obs: List[float], act: List[float]) -> float:
        x = torch.FloatTensor(obs + act).to(device).unsqueeze(0)
        q_val = self.q.fc1(x)
        q_val = F.relu(q_val)
        q_val = self.q.fc2(q_val)
        q_val = F.relu(q_val)
        q_val = self.q.fc3(q_val)
        return float(q_val.squeeze().item())


class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
        self.full = False

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
            self.idx = (self.idx + 1) % self.capacity
            if self.idx == 0:
                self.full = True

    def size(self):
        return len(self.buffer)

    def sample(self, rng: RNG, batch_size: int) -> List[Transition]:
        n = self.size()
        indices = [rng.randint(0, n - 1) for _ in range(batch_size)]
        return [self.buffer[i] for i in indices]


def polyak_update(target_params, src_params, tau):
    for t, s in zip(target_params, src_params):
        t.data.copy_((1.0 - tau) * t.data + tau * s.data)


def clampd(v, lo, hi):
    return max(lo, min(hi, v))


class SAC:
    def __init__(self, obs_dim: int, act_dim: int = 2, seed: int = 1234):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = RNG(seed)

        self.actor = Actor(obs_dim, act_dim)
        self.critic_q1 = Critic(obs_dim, act_dim)
        self.critic_q2 = Critic(obs_dim, act_dim)
        self.critic_q1_target = Critic(obs_dim, act_dim)
        self.critic_q2_target = Critic(obs_dim, act_dim)

        self._sync_target_networks()

        self.replay_buffer = ReplayBuffer(200000)

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.batch_size = 128
        self.learn_start = 1000
        self.updates_per_step = 1

    def _sync_target_networks(self):
        with torch.no_grad():
            for src, tgt in [(self.critic_q1, self.critic_q1_target),
                           (self.critic_q2, self.critic_q2_target)]:
                for s_param, t_param in zip(src.parameters(), tgt.parameters()):
                    t_param.data.copy_(s_param.data)

    def act(self, obs: List[float], deterministic: bool = False) -> List[float]:
        action, _ = self.actor.sample_action(obs, self.rng, deterministic)
        return action

    def store(self, s: List[float], a: List[float], r: float, s2: List[float], done: bool):
        transition = Transition(s, a, s2, r, done)
        self.replay_buffer.push(transition)

    def update_once(self):
        if self.replay_buffer.size() < self.learn_start:
            return

        batch = self.replay_buffer.sample(self.rng, self.batch_size)

        # Critic update
        for tr in batch:
            # Target Q value
            a2, logp2 = self.actor.sample_action(tr.s2, self.rng, deterministic=False)

            qt1 = self.critic_q1_target.forward_q(tr.s2, a2)
            qt2 = self.critic_q2_target.forward_q(tr.s2, a2)
            min_qt = min(qt1, qt2)

            y = tr.r + (0.0 if tr.done else self.gamma * (min_qt - self.alpha * logp2))

            # Q1 loss
            s_tensor = torch.FloatTensor(tr.s).to(device).unsqueeze(0)
            a_tensor = torch.FloatTensor(tr.a).to(device).unsqueeze(0)
            y_tensor = torch.FloatTensor([y]).to(device)

            q1_pred = self.critic_q1.q(torch.cat([s_tensor, a_tensor], dim=1))
            loss_q1 = F.mse_loss(q1_pred.unsqueeze(0), y_tensor.unsqueeze(0))

            self.critic_q1.q.optimizer.zero_grad()
            loss_q1.backward()
            self.critic_q1.q.optimizer.step()

            # Q2 loss
            q2_pred = self.critic_q2.q(torch.cat([s_tensor, a_tensor], dim=1))
            loss_q2 = F.mse_loss(q2_pred.unsqueeze(0), y_tensor.unsqueeze(0))

            self.critic_q2.q.optimizer.zero_grad()
            loss_q2.backward()
            self.critic_q2.q.optimizer.step()

        # Target network update
        polyak_update(self.critic_q1_target.parameters(), self.critic_q1.parameters(), self.tau)
        polyak_update(self.critic_q2_target.parameters(), self.critic_q2.parameters(), self.tau)

    def update_many(self, n: int):
        for _ in range(n):
            self.update_once()

    def replay_size(self) -> int:
        return self.replay_buffer.size()
