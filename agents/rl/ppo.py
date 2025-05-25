from typing import Literal, Optional
import numpy as np
import torch
from marlenv import Transition, Episode
from marlenv.utils import Schedule

from agents import Agent

from .batch import Batch, TransitionBatch, EpisodeBatch
from .replay_memory import ReplayMemory
from .networks import ActorCritic


class PPO(Agent):
    actor_critic: ActorCritic
    memory: ReplayMemory
    batch_size: int
    minibatch_size: int
    c1: Schedule
    c2: Schedule
    eps_clip: float
    gae_lambda: float
    gamma: float
    lr: float
    n_epochs: int
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        actor_critic: ActorCritic,
        memory: ReplayMemory,
        gamma: float = 0.99,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 20,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        entropy_c2: Schedule | float = 0.01,
        train_interval: int = 64,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
        minibatch_size: int = 32,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__()
        self._device = device
        self.batch_size = train_interval
        self.actor_critic = actor_critic.to(device)
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self.minibatch_size = minibatch_size
        self.memory = memory
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

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        all_parameters = list(self.actor_critic.parameters())
        params = [
            {"params": self.actor_critic.actor_parameters(), "lr": lr_actor, "name": "actor parameters"},
            {"params": self.actor_critic.critic_parameters(), "lr": lr_critic, "name": "critic parameters"},
        ]
        return params, all_parameters

    def choose_action(self, observation: np.ndarray, hx: torch.Tensor | None):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation).unsqueeze(0).to(self.device, non_blocking=True)
            distribution, hx = self.actor_critic.policy(obs_data, hx)
        np_action = distribution.sample().squeeze(0).numpy(force=True)
        return np_action, hx

    def _compute_training_data(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the returns, advantages and action log_probs according to the current policy"""
        policy, _ = self.actor_critic.policy(batch.obs)
        log_probs = policy.log_prob(batch.actions) * batch.masks
        all_values, _ = self.actor_critic.value(batch.all_obs)
        values = all_values[:-1] * batch.masks
        next_values = all_values[1:] * batch.not_dones
        advantages = batch.compute_gae(self.gamma, values, next_values, self.gae_lambda, normalize=True)
        returns = advantages + values
        return returns, advantages, log_probs

    def train(self, batch: Batch, step_num: int, episode_num: int):
        batch.normalize_rewards()
        self.c1.update(episode_num)
        self.c2.update(episode_num)
        with torch.no_grad():
            returns, advantages, log_probs = self._compute_training_data(batch)

        for _ in range(self.n_epochs):
            indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            match batch:
                case TransitionBatch():
                    mini_log_probs, mini_returns, mini_advantages = log_probs[indices], returns[indices], advantages[indices]
                case EpisodeBatch():
                    mini_log_probs, mini_returns, mini_advantages = log_probs[:, indices], returns[:, indices], advantages[:, indices]
            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values, _ = self.actor_critic.value(minibatch.obs)
            mini_values = mini_values * minibatch.masks
            td_error = mini_values - mini_returns
            critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy, _ = self.actor_critic.policy(minibatch.obs)
            new_log_probs = mini_policy.log_prob(minibatch.actions)

            ratios = torch.exp(new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratios
            surrogate2 = torch.clamp(ratios, self._ratio_min, self._ratio_max) * mini_advantages
            # Minus because we want to maximize the objective
            actor_loss = torch.sum(-torch.min(surrogate1, surrogate2)) / minibatch.masks_sum

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.entropy()
            masked_entropy = entropy * minibatch.masks
            entropy_loss = torch.sum(masked_entropy) / minibatch.masks_sum

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
            self.optimizer.step()

    def update_transition(self, t: Transition, step: int, episode_num: int):
        if self.memory.update_on_transitions:
            self.memory.add(t)
            if self.memory.is_full:
                batch = self.memory.as_batch(self.device)
                self.train(batch, step, episode_num)
                self.memory.clear()

    def update_episode(self, episode: Episode, step_num: int, episode_num: int):
        if self.memory.update_on_episodes:
            self.memory.add(episode)
            if self.memory.is_full:
                batch = self.memory.as_batch(self.device)
                self.train(batch, step_num, episode_num)
                self.memory.clear()

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

        Seed `ranom`, `numpy`, and `torch` libraries by default.
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
