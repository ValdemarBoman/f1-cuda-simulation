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

    def _distribution(self, obs_t: torch.Tensor):
        out = self.net(obs_t)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        mu, logstd = torch.chunk(out, 2, dim=-1)
        logstd = torch.clamp(logstd, self.log_std_min, self.log_std_max)
        std = torch.exp(logstd)
        return mu, std

    def sample_action_tensor(self, obs_t: torch.Tensor, deterministic: bool = False):
        mu, std = self._distribution(obs_t)
        if deterministic:
            u = mu
        else:
            eps = torch.randn_like(std)
            u = mu + std * eps
        a = torch.tanh(u)
        logp = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * torch.log(std + 1e-8) + math.log(2.0 * math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        logp = logp - torch.log(torch.clamp(1.0 - a * a, min=1e-6)).sum(dim=-1, keepdim=True)
        return a, logp

    def sample_action(self, obs: List[float], rng: RNG, deterministic: bool = False) -> Tuple[List[float], float]:
        del rng  # Sampling uses torch RNG on selected device.
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a, logp = self.sample_action_tensor(obs_t, deterministic=deterministic)
        return a.squeeze(0).detach().cpu().tolist(), float(logp.item())

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
        x = torch.as_tensor(obs + act, dtype=torch.float32, device=device).unsqueeze(0)
        return float(self.q(x).squeeze().item())


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

        s = torch.as_tensor(np.asarray([tr.s for tr in batch], dtype=np.float32), device=device)
        a = torch.as_tensor(np.asarray([tr.a for tr in batch], dtype=np.float32), device=device)
        s2 = torch.as_tensor(np.asarray([tr.s2 for tr in batch], dtype=np.float32), device=device)
        r = torch.as_tensor(np.asarray([tr.r for tr in batch], dtype=np.float32), device=device).unsqueeze(1)
        done = torch.as_tensor(np.asarray([tr.done for tr in batch], dtype=np.float32), device=device).unsqueeze(1)

        with torch.no_grad():
            a2, logp2 = self.actor.sample_action_tensor(s2, deterministic=False)
            q1_t = self.critic_q1_target.q(torch.cat([s2, a2], dim=1))
            q2_t = self.critic_q2_target.q(torch.cat([s2, a2], dim=1))
            min_q_t = torch.minimum(q1_t, q2_t)
            y = r + (1.0 - done) * self.gamma * (min_q_t - self.alpha * logp2)

        q1_pred = self.critic_q1.q(torch.cat([s, a], dim=1))
        q2_pred = self.critic_q2.q(torch.cat([s, a], dim=1))
        loss_q1 = F.mse_loss(q1_pred, y)
        loss_q2 = F.mse_loss(q2_pred, y)

        self.critic_q1.q.optimizer.zero_grad(set_to_none=True)
        loss_q1.backward()
        self.critic_q1.q.optimizer.step()

        self.critic_q2.q.optimizer.zero_grad(set_to_none=True)
        loss_q2.backward()
        self.critic_q2.q.optimizer.step()

        # Policy update
        a_pi, logp_pi = self.actor.sample_action_tensor(s, deterministic=False)
        q1_pi = self.critic_q1.q(torch.cat([s, a_pi], dim=1))
        q2_pi = self.critic_q2.q(torch.cat([s, a_pi], dim=1))
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - min_q_pi).mean()

        self.actor.net.optimizer.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.actor.net.optimizer.step()

        # Target network update
        polyak_update(self.critic_q1_target.parameters(), self.critic_q1.parameters(), self.tau)
        polyak_update(self.critic_q2_target.parameters(), self.critic_q2.parameters(), self.tau)

    def update_many(self, n: int):
        for _ in range(n):
            self.update_once()

    def replay_size(self) -> int:
        return self.replay_buffer.size()
