from rlenv import Transition
import torch
import numpy as np
import torch.utils
from .agent import Agent
from .networks import ActorCritic

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []


class PPO(Agent):
    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        lr_actor: float,
        lr_critic: float,
        action_min: np.ndarray,  # New parameter for minimum action bounds
        action_max: np.ndarray,  # New parameter for maximum action bounds
        k_epochs: int,
        eps_clip: float,
        update_interval=16,
    ):
        self.device = torch.device("cuda")
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_min = torch.tensor(action_min).to(self.device)  # Convert to tensor for efficiency
        self.action_max = torch.tensor(action_max).to(self.device)  # Convert to tensor for efficiency
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(obs_size, n_actions)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actions_mean_std.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(obs_size, n_actions)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = torch.nn.MSELoss()
        self.time_step = 0
        self.update_interval = update_interval
        self.parameters = self.policy.parameters()

    def select_action(self, obs_data: np.ndarray):
        with torch.no_grad():
            tensor_data = torch.from_numpy(obs_data).to(self.device)
            action, action_logprob = self.policy.act(tensor_data)
            # Convert the action to a numpy array, perform the clamping, and then convert back to a tensor
            clamped_action_numpy = np.clip(action.cpu().numpy(), self.action_min.cpu().numpy(),
                                           self.action_max.cpu().numpy())

            # Convert the clamped numpy array back to a PyTorch tensor and move it to the same device as the original action
            clamped_action = torch.tensor(clamped_action_numpy, dtype=action.dtype, device=action.device)

            state_val = self.policy.value(tensor_data).item()

        self.buffer.state_values.append(state_val)
        return clamped_action, action_logprob.numpy(force=True)

    def store(self, transition: Transition):
        self.buffer.states.append(transition.obs.data)
        self.buffer.actions.append(transition.action)
        self.buffer.rewards.append(transition.reward.item())
        self.buffer.is_terminals.append(transition.done)
        self.buffer.logprobs.append(transition.probs)  # type: ignore

    def update(self):
        self.time_step += 1
        if self.time_step % self.update_interval != 0:
            return

        # Use immediate rewards
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        # Normalize rewards
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert lists to tensors
        old_states = torch.squeeze(torch.from_numpy(np.stack(self.buffer.states))).to(self.device)
        if old_states.dim() <= 1:
            old_states = old_states.unsqueeze(-1)

        old_actions = torch.squeeze(
            torch.from_numpy(
                np.stack([action.cpu().numpy() for action in self.buffer.actions])
            )
        ).to(self.device)
        if old_actions.dim() <= 1:
            pass

        old_logprobs = torch.squeeze(torch.from_numpy(np.stack(self.buffer.logprobs))).to(self.device)
        old_state_values = torch.tensor(self.buffer.state_values, device=self.device)

        # Calculate advantages as reward - state_value
        advantages = rewards - old_state_values


        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = self.policy.value(old_states).squeeze()

            # Calculate the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def to(self, device: torch.device):
        self.policy.to(device)
        self.policy_old.to(device)
        self.device = device
        return self
