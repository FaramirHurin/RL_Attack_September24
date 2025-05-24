from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import torch
from marlenv import Episode

from .batch import Batch


class EpisodeBatch(Batch):
    def __init__(self, episodes: Sequence[Episode], device: Optional[torch.device] = None, pad_episodes: bool = True):
        super().__init__(len(episodes), episodes[0].n_agents, device)
        self._max_episode_len = max(len(e) for e in episodes)
        self._base_episodes = episodes
        if pad_episodes:
            episodes = [e.padded(self._max_episode_len) for e in episodes]
        self.episodes = episodes

    def for_individual_learners(self):
        self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape, self.n_agents)
        return super().for_individual_learners()

    def compute_returns(self, gamma: float) -> torch.Tensor:
        result = torch.empty_like(self.rewards, dtype=torch.float32)
        next_step_returns = self.rewards[-1]
        result[-1] = next_step_returns
        for step in range(self._max_episode_len - 2, -1, -1):
            reward = self.rewards[step]
            next_step_returns = reward + gamma * next_step_returns
            result[step] = next_step_returns
        return result

    def get_minibatch(self, indices_or_size) -> Batch:
        match indices_or_size:
            case int(minibatch_size):
                indices = np.random.choice(self.size, minibatch_size, replace=False)
            case tuple() | list() | np.ndarray() as indices:
                pass
            case _:
                raise ValueError(f"Invalid minibatch size {indices_or_size}")
        return EpisodeBatch([self.episodes[i] for i in indices], self.device, pad_episodes=False)

    def multi_objective(self):
        raise NotImplementedError()
        self.actions = self.actions.unsqueeze(-1).repeat(*(1 for _ in self.actions.shape), self.reward_size)

    def __getitem__(self, key: str) -> torch.Tensor:
        res = np.array([e[key] for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(res).to(self.device)

    def compute_gae_(
        self,
        gamma: float,
        values: torch.Tensor,
        next_values: torch.Tensor,
        trace_decay: float = 0.95,
        normalize: bool = False,
    ):
        deltas = self.rewards + gamma * next_values - values
        if self.rewards.dim() > 1:
            gae = torch.zeros(self.size, dtype=torch.float32).to(device=self.device)
        else:
            gae = torch.zeros(self.reward_size, dtype=torch.float32).to(device=self.device)
        advantages = torch.empty_like(self.rewards, dtype=torch.float32).to(device=self.device)
        # Note: we want to discount the reward by the actual time between two observations
        dt = self.dt
        for t in range(self._max_episode_len - 1, -1, -1):
            gae = deltas[t] + gamma ** dt[t] * trace_decay * gae
            advantages[t] = gae
        if normalize:
            advantages = self._normalize(advantages)
        return advantages

    def _initialize_gae(self):
        return torch.zeros(self.size, dtype=torch.float32).to(device=self.device), self._max_episode_len

    @cached_property
    def dt(self):
        """
        Delta time (in days) between two consecutile observations.
        """
        delay_days = self.actions[:, :, -2]
        delay_hours = self.actions[:, :, -1]
        return delay_days + delay_hours / 24.0

    @cached_property
    def reward_size(self) -> int:
        """Number of rewards, i.e. the number of objectives"""
        if self.rewards.dim() == 2:
            return 1
        return self.rewards.shape[-1]

    @cached_property
    def probs(self):
        raise NotImplementedError()

    @cached_property
    def obs(self):
        obs = np.array([e.obs for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(obs).to(self.device).transpose(0, 1)

    @cached_property
    def next_obs(self):
        obs = np.array([e.next_obs for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(obs).to(self.device).transpose(0, 1)

    @cached_property
    def all_obs(self):
        all_obs_ = np.array([e.all_observations for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(all_obs_).to(self.device).transpose(0, 1)

    @cached_property
    def extras(self):
        extras = np.array([e.extras for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(extras).to(self.device).transpose(0, 1)

    @cached_property
    def next_extras(self):
        extras_ = np.array([e.next_extras for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(extras_).to(self.device).transpose(0, 1)

    @cached_property
    def states_extras(self):
        extras_ = np.array([e.states_extras for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(extras_).to(self.device).transpose(0, 1)

    @cached_property
    def next_states_extras(self):
        extras_ = np.array([e.next_states_extras for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(extras_).to(self.device).transpose(0, 1)

    @cached_property
    def all_extras(self):
        all_extras_ = np.array([e.all_extras for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(all_extras_).to(self.device).transpose(0, 1)

    @cached_property
    def available_actions(self):
        available_actions = np.array([e.available_actions for e in self.episodes], dtype=np.bool)
        return torch.from_numpy(available_actions).to(self.device).transpose(0, 1)

    @cached_property
    def next_available_actions(self):
        available_actions_ = np.array([e.next_available_actions for e in self.episodes], dtype=np.bool)
        return torch.from_numpy(available_actions_).to(self.device).transpose(0, 1)

    @cached_property
    def states(self):
        states = np.array([e.states for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(states).to(self.device).transpose(0, 1)

    @cached_property
    def next_states(self):
        states_ = np.array([e.next_states for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(states_).to(self.device).transpose(0, 1)

    @cached_property
    def actions(self):
        dtype = self.episodes[0].actions[0].dtype
        actions = torch.from_numpy(np.array([e.actions for e in self.episodes], dtype=dtype)).to(self.device)
        return actions.transpose(0, 1)

    @cached_property
    def rewards(self):
        rewards = torch.from_numpy(np.array([e.rewards for e in self.episodes], dtype=np.float32)).to(self.device)
        return rewards.squeeze(-1).transpose(0, 1)

    @cached_property
    def dones(self):
        dones = torch.from_numpy(np.array([e.dones for e in self.episodes], dtype=np.bool)).to(self.device)
        return dones.squeeze(-1).transpose(0, 1)

    @cached_property
    def masks(self):
        masks = torch.from_numpy(np.array([e.mask for e in self.episodes], dtype=np.float32)).to(self.device)
        return masks.squeeze(-1).transpose(0, 1)
