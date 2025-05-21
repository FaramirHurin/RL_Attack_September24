from typing import Literal, Optional

import numpy as np
import torch
from marlenv import Transition
from marlenv.utils import Schedule

from .batch import Batch, TransitionBatch
from agents import Agent


class RPPO_2(Agent):
    batch_size: int
    c1: Schedule
    c2: Schedule
    eps_clip: float
    gae_lambda: float
    gamma: float
    lr: float
    minibatch_size: int
    n_epochs: int
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        actor_critic,
        gamma: float,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 20,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        entropy_c2: Schedule | float = 0.01,
        train_interval: int = 128,
        minibatch_size: int = 10,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._device = device
        self.batch_size = train_interval
        if minibatch_size is None:
            minibatch_size = train_interval
        self.minibatch_size = minibatch_size
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self._ratio_min = 1 - eps_clip
        self._ratio_max = 1 + eps_clip
        param_groups, self._parameters = self._compute_param_groups(lr_actor, lr_critic)
        self.optimizer = torch.optim.Adam(param_groups)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.c2 = entropy_c2
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping
        self._memory = list[Transition]()

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        all_parameters = list(self.actor_critic.parameters())
        params = [
            {"params": self.actor_critic.actor.parameters(), "lr": lr_actor, "name": "actor parameters"},
            {"params": self.actor_critic.critic.parameters(), "lr": lr_critic, "name": "critic parameters"},
        ]
        return params, all_parameters

    def choose_action(self, observation: np.ndarray):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation).unsqueeze(0).to(self.device, non_blocking=True)
            distribution, _ = self.actor_critic.policy(obs_data)
        action = distribution.sample().squeeze(0)
        return action.numpy(force=True)

    def _compute_training_data(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the returns, advantages and action log_probs according to the current policy"""
        batch.to(self.device)
        policy, _ = self.actor_critic.policy(batch.obs)
        log_probs = policy.log_prob(batch.actions)
        all_values, _ = self.actor_critic.value(batch.all_obs)
        values = all_values[:-1]
        next_values = all_values[1:]
        advantages = batch.compute_gae(self.gamma, values, next_values, trace_decay=self.gae_lambda)
        returns = advantages + values
        return returns, advantages, log_probs

    def train(self, batch: Batch, step: int):
        self.c1.update(step)
        self.c2.update(step)
        with torch.no_grad():
            returns, advantages, log_probs = self._compute_training_data(batch)

        for _ in range(self.n_epochs):
            indices = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            mini_log_probs, mini_returns, mini_advantages = log_probs[indices], returns[indices], advantages[indices]

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values, _ = self.actor_critic.value(minibatch.obs)
            critic_loss = torch.nn.functional.mse_loss(mini_values, mini_returns)

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy, _ = self.actor_critic.policy(minibatch.obs)
            new_log_probs = mini_policy.log_prob(minibatch.actions)

            ratios = torch.exp(new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratios
            surrogate2 = torch.clamp(ratios, self._ratio_min, self._ratio_max) * mini_advantages
            # Minus because we want to maximize the objective
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy_loss = torch.mean(mini_policy.entropy())

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            self.optimizer.step()

    def update(self, t: Transition, step: int):
        self._memory.append(t)
        if len(self._memory) >= self.batch_size:
            batch = TransitionBatch(self._memory).to(self.device)
            self.train(batch, step)
            self._memory.clear()

    @property
    def networks(self):
        """Dynamic list of neural networks attributes in the trainer"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, torch.nn.Module)]

    @property
    def device(self):
        return self._device

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the state of the trainer."""
        match method:
            case "xavier":
                rinit = torch.nn.init.xavier_uniform_
            case "orthogonal":
                rinit = torch.nn.init.orthogonal_
            case _:
                raise ValueError(f"Unknown randomization method: {method}")
        for nn in self.networks:
            randomize(rinit, nn)

    def to(self, device: torch.device):
        """Send the networks to the given device."""
        self._device = device
        for nn in self.networks:
            nn.to(device)
        return self

    def seed(self, seed: int):
        """
        Seed the algorithm for reproducibility (e.g. during testing).

        Seed `random`, `numpy`, and `torch` libraries by default.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def randomize(init_fn, nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)
