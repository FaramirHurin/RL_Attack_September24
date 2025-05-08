import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from environment import Action
import numpy as np
from marlenv import Transition
from copy import deepcopy

from rl.batch import TransitionBatch


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super().__init__()
        # self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values.squeeze(-1), dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        action_std_init=0.6,
    ):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.memory = list[Transition]()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def choose_action(self, state: np.ndarray):
        with torch.no_grad():
            tensor_state = torch.from_numpy(state)
            action, action_logprob, state_val = self.policy.act(tensor_state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.numpy(force=True).flatten()

    def update(self, transition: Transition):
        # Monte Carlo estimate of returns
        self.memory.append(transition)
        self.buffer.rewards.append(transition.reward.item())
        self.buffer.is_terminals.append(transition.is_terminal)
        # assert np.array_equal(self.memory[-1].obs.data, self.buffer.states[-1])
        # assert np.array_equal(self.memory[-1].action, self.buffer.actions[-1])
        # assert self.memory[-1].reward.item() == self.buffer.rewards[-1]
        if len(self.buffer.states) < 32:
            return
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # batch = TransitionBatch(deepcopy(self.memory))
        # returns = batch.compute_mc_returns(self.gamma, torch.zeros(1))

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # convert list to tensor
        old_states = torch.from_numpy(np.array(self.buffer.states))
        old_actions = torch.from_numpy(np.array(self.buffer.actions)).squeeze(1)
        with torch.no_grad():
            old_logprobs, old_state_values, _ = self.policy.evaluate(old_states, old_actions)
            # old_logprobs2, old_state_values2, _ = self.policy.evaluate(batch.obs, batch.actions)

        # calculate advantages
        advantages = rewards.detach() - old_state_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()
