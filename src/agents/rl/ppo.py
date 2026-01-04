import logging
from typing import Optional
import numpy as np
import torch
import os
from marlenv import Transition, Episode
from marlenv.utils import Schedule
from agents import Agent
from utils import tb_log
from copy import deepcopy

from .batch import Batch, EpisodeBatch
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
        normalize_advantages: bool = True,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__()
        if len(kwargs) > 0:
            logging.warning(f"Unexpected PPO arguments ignored: {kwargs}")
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
        self.normalize_advantages = normalize_advantages
        param_groups, self._parameters = self._compute_param_groups(lr_actor, lr_critic)
        self.optimizer = torch.optim.AdamW(param_groups, eps=1e-5)
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
        torch_action: torch.Tensor = distribution.sample()  # type: ignore
        np_action = torch_action.squeeze(0).numpy(force=True)
        return np_action, hx

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        values = self.actor_critic.value(batch.obs)
        next_values = self.actor_critic.value(batch.next_obs)
        values[batch.masked_indices] = 0.0
        next_values[batch.dones == 1] = 0.0
        assert torch.all(next_values[batch.masked_indices] == 0.0)
        advantages = batch.compute_gae(self.gamma, values, next_values, self.gae_lambda, normalize=self.normalize_advantages)
        returns = batch.compute_mc_returns(self.gamma, 0.0)
        advantages[batch.masked_indices] = 0.0
        assert torch.all(advantages[batch.masked_indices] == 0)
        assert torch.all(returns[batch.dones == 1] == 0.0)
        assert torch.all(returns[batch.masked_indices] == 0.0)
        return returns, advantages

    def save(self, path: str):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            torch.save(self.actor_critic.state_dict(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.actor_critic.load_state_dict(torch.load(f))

    def train(self, batch: Batch, step_num: int, episode_num: int, simulation_t: int):
        self.c1.update(episode_num)
        self.c2.update(episode_num)

        old_ac = deepcopy(self.actor_critic)
        with torch.no_grad():
            returns, advantages = self._compute_training_data(batch)
        critic_losses, actor_losses, entropy_losses, losses, ratios, entropies, norms = [], [], [], [], [], [], []
        for e in range(self.n_epochs):
            if self.minibatch_size == batch.size:
                minibatch = batch
                indices = slice(None)
            else:
                indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
                minibatch = batch.get_minibatch(indices)
            if isinstance(minibatch, EpisodeBatch):
                indices = (slice(None), indices)  # The episode dimension come second in episode batches: (time, episode, ...)
            mini_returns, mini_advantages = returns[indices], advantages[indices]
            with torch.no_grad():
                mini_log_probs = old_ac.policy(minibatch.obs)[0].log_prob(minibatch.actions)
                mini_log_probs[minibatch.masked_indices] = 0.0

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.actor_critic.value(minibatch.obs)
            mini_values[minibatch.masked_indices] = 0.0
            td_error = mini_values - mini_returns
            critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy = self.actor_critic.policy(minibatch.obs)[0]
            mini_new_log_probs: torch.Tensor = mini_policy.log_prob(minibatch.actions)
            mini_new_log_probs[minibatch.masked_indices] = 0.0
            ratio = torch.exp(mini_new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratio
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min) / minibatch.masks_sum  # Minus sign to maximize the objective

            if e == 0:
                assert torch.equal(ratio, torch.ones_like(ratio)), f"Ratio max diff = {(ratio - 1).abs().max()}"

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.base_dist.entropy()
            masked_entropy = entropy * minibatch.masks
            entropy_loss = -torch.sum(masked_entropy) / minibatch.masks_sum  # Minus sign to maximize the entropy

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                norms.append(norm.cpu().item())
            self.optimizer.step()
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            losses.append(loss.item())
            ratios.append(ratio.numpy(force=True))
            entropies.append(entropy.numpy(force=True))

        tb_log("ppo/min_new_log_prob", mini_new_log_probs.min().item(), simulation_t)
        tb_log("ppo/max_new_log_prob", mini_new_log_probs.max().item(), simulation_t)
        tb_log("ppo/mean_new_log_prob", mini_new_log_probs.mean().item(), simulation_t)
        tb_log("ppo/min_critic_loss", min(critic_losses), simulation_t)
        tb_log("ppo/max_critic_loss", max(critic_losses), simulation_t)
        tb_log("ppo/mean_critic_loss", np.mean(critic_losses), simulation_t)
        tb_log("ppo/min_actor_loss", min(actor_losses), simulation_t)
        tb_log("ppo/max_actor_loss", max(actor_losses), simulation_t)
        tb_log("ppo/mean_actor_loss", np.mean(actor_losses), simulation_t)
        tb_log("ppo/min_entropy_loss", min(entropy_losses), simulation_t)
        tb_log("ppo/max_entropy_loss", max(entropy_losses), simulation_t)
        tb_log("ppo/mean_entropy_loss", np.mean(entropy_losses), simulation_t)
        tb_log("ppo/min_loss", min(losses), simulation_t)
        tb_log("ppo/max_loss", max(losses), simulation_t)
        tb_log("ppo/mean_loss", np.mean(losses), simulation_t)
        tb_log("ppo/min_ratio", np.min(ratios), simulation_t)
        tb_log("ppo/max_ratio", np.max(ratios), simulation_t)
        tb_log("ppo/mean_ratio", np.mean(ratios), simulation_t)
        tb_log("ppo/mean_entropy", np.mean(entropies), simulation_t)
        tb_log("ppo/min_entropy", np.min(entropies), simulation_t)
        tb_log("ppo/max_entropy", np.max(entropies), simulation_t)
        if len(norms) > 0:
            tb_log("ppo/min_grad_norm", min(norms), simulation_t)
            tb_log("ppo/max_grad_norm", max(norms), simulation_t)
            tb_log("ppo/mean_grad_norm", np.mean(norms), simulation_t)

    def update_transition(self, transition: Transition, step: int, episode_num: int, simulation_t: int):
        if self.memory.update_on_transitions:
            self.memory.add(transition)
            if self.memory.is_full:
                batch = self.memory.as_batch(self.device)
                self.train(batch, step, episode_num, simulation_t)
                self.memory.clear()

    def update_episode(self, episode: Episode, step_num: int, episode_num: int, simulation_t: int):
        if self.memory.update_on_episodes:
            self.memory.add(episode)
            if self.memory.is_full:
                batch = self.memory.as_batch(self.device)
                self.train(batch, step_num, episode_num, simulation_t)
                self.memory.clear()

    @property
    def device(self):
        return self._device
