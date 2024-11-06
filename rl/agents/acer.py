import torch
import numpy as np
from marlenv import Transition, EpisodeBuilder
from rl.replay_memory import EpisodeMemory
from .networks import ActorCritic
from .agent import Agent
import logging


class ACER(Agent):
    """Actor-Critic with Experience Replay algorithm (ACER)."""

    def __init__(self, gamma: float, lr: float, state_size: int, n_actions: int, device: torch.device):
        self.device = device
        self.gamma = gamma
        self.policy = ActorCritic(state_size, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss()
        self.memory = EpisodeMemory(20_000)
        self.batch_size = 32
        self.is_training = True
        self.action_probs = []
        self.current_episode = EpisodeBuilder()

    def select_action(self, obs_data: np.ndarray):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_data).to(self.device, non_blocking=True)
            action, action_logprob = self.policy.act(obs_tensor)

        return action.numpy(force=True), action_logprob.numpy(force=True)

    def store(self, transition: Transition):
        self.current_episode.add(transition)
        if transition.is_terminal:
            self.memory.add(self.current_episode.build())
            self.current_episode = EpisodeBuilder()

    def update(self):
        if not self.memory.can_sample(self.batch_size):
            return

        batch = self.memory.sample(self.batch_size).to(self.device)
        returns = torch.squeeze(batch.compute_returns(self.gamma))
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        states = batch.obs.squeeze(0)
        actions = batch.actions
        actions = actions.squeeze()
        predicted_values = self.policy.value(states)
        advantages = returns - torch.squeeze(predicted_values)

        # Computation of the importance sampling weights 'rho' (equation 3 in the paper)
        log_probs, _ = self.policy.evaluate(states, actions)
        old_probs = batch.probs.squeeze()
        rho = torch.exp(log_probs - old_probs)

        # Multiply by -1 because of gradient ascent
        weighted_log_probs = log_probs * rho * advantages
        actor_loss = -torch.sum(weighted_log_probs) / self.batch_size
        critic_loss = torch.sum(advantages**2) / self.batch_size

        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        # logging.debug(loss.item())
        self.optimizer.step()

    def to(self, device: torch.device):
        self.device = device
        self.policy.to(device)
        return self
