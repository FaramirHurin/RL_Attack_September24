import logging
from typing import Optional
import numpy as np
import torch
import os
from marlenv import Transition, Episode
from marlenv.utils import Schedule
from agents import Agent
from utils import tb_log


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
        normalize_rewards: bool = True,
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
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        param_groups, self._parameters = self._compute_param_groups(lr_actor, lr_critic)
        self.optimizer = torch.optim.Adam(param_groups, eps=1e-5)
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

    def _compute_training_data(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the returns, advantages and action log_probs according to the current policy"""
        # NOTE: Mask the log probs to prevent numerical instability when passed in torch.exp. If the value
        # present without masking is large enough (e.g. >=10^3), torch.exp yields +inf which causes issues
        # in the optimization process because it can not be masked properly (0 x inf = NaN).
        policy, _ = self.actor_critic.policy(batch.obs)
        log_probs = policy.log_prob(batch.actions)
        log_probs[batch.masked_indices] = 0.0
        all_values, _ = self.actor_critic.value(batch.all_obs)
        values = all_values[:-1] * batch.masks
        next_values = all_values[1:] * batch.not_dones
        advantages = batch.compute_gae(self.gamma, values, next_values, self.gae_lambda, normalize=False)
        returns = batch.compute_mc_returns(self.gamma, normalize=False)
        assert torch.all(advantages[batch.masked_indices] == 0)
        assert torch.all(returns[batch.masked_indices] == 0)
        return returns, advantages, log_probs

    def save(self, path: str):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            torch.save(self.actor_critic.state_dict(), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.actor_critic.load_state_dict(torch.load(f))

    def train(self, batch: Batch, step_num: int, episode_num: int, simulation_t: int):
        if self.normalize_rewards:
            batch.normalize_rewards()
        self.c1.update(episode_num)
        self.c2.update(episode_num)
        with torch.no_grad():
            returns, advantages, log_probs = self._compute_training_data(batch)

        critic_losses, actor_losses, entropy_losses, losses, ratios, entropies = [], [], [], [], [], []
        for e in range(self.n_epochs):
            if step_num >= 242:
                debug = True
            indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            if isinstance(minibatch, EpisodeBatch):
                indices = (..., indices)  # The episode dimension come second in episode batches: (time, episode, ...)
            mini_log_probs, mini_returns, mini_advantages = log_probs[indices], returns[indices], advantages[indices]

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values, _ = self.actor_critic.value(minibatch.obs)
            td_error = mini_values - mini_returns
            td_error[minibatch.masked_indices] = 0.0
            critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy, _ = self.actor_critic.policy(minibatch.obs)
            mini_new_log_probs = mini_policy.log_prob(minibatch.actions)
            mini_new_log_probs[minibatch.masked_indices] = 0.0
            ratio = torch.exp(mini_new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratio
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min) / minibatch.masks_sum  # Minus sign to maximize the objective

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.base_dist.entropy()
            masked_entropy = entropy * minibatch.masks
            entropy_loss = -torch.sum(masked_entropy) / minibatch.masks_sum  # Minus sign to maximize the entropy

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
            self.optimizer.step()
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            losses.append(loss.item())
            ratios.append(ratio.numpy(force=True))
            entropies.append(entropy.numpy(force=True))

        tb_log("ppo/mean_log_probs", log_probs.mean().item(), simulation_t)
        tb_log("ppo/min_log_probs", log_probs.min().item(), simulation_t)
        tb_log("ppo/max_log_probs", log_probs.max().item(), simulation_t)
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
