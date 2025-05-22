from typing import Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import distributions


class PositiveDefiniteMatrixGenerator(nn.Module):
    def __init__(self, input_size, matrix_size):
        super().__init__()
        # Output size is for the lower triangular part of the matrix
        self.fc = nn.Linear(input_size, matrix_size * (matrix_size + 1) // 2)

    def forward(self, x):
        # Get the output from the linear layer
        chol_params = self.fc(x)  # Cholesky parameters
        # Initialize a lower triangular matrix
        L = torch.zeros((x.size(0), chol_params.size(1), chol_params.size(1)), device=x.device)
        # Fill the lower triangular part
        idx = 0
        for i in range(L.size(1)):
            for j in range(i + 1):
                if i == j:
                    # Diagonal entries should be positive
                    L[:, i, j] = torch.relu(chol_params[:, idx])  # Use ReLU to ensure positivity
                else:
                    L[:, i, j] = chol_params[:, idx]
                idx += 1
        # Generate the positive definite matrix
        positive_definite_matrix = torch.bmm(L, L.transpose(1, 2))  # L * L^T
        return positive_definite_matrix


class ActorCritic(torch.nn.Module, ABC):
    @abstractmethod
    def policy(
        self,
        states: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.distributions.Distribution, Optional[torch.Tensor]]: ...

    @abstractmethod
    def value(
        self,
        states: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]: ...

    @abstractmethod
    def actor_parameters(self) -> list[torch.nn.Parameter]: ...

    @abstractmethod
    def critic_parameters(self) -> list[torch.nn.Parameter]: ...


class LinearActorCritic(ActorCritic):
    def __init__(self, state_size: int, n_actions: int, device: torch.device):
        super().__init__()
        self.n_actions = n_actions
        self.device = device
        # Because we output one mean per action and a covariance matrix, we have an output of size n_actions + n_actions**2
        n_action_outputs = n_actions + n_actions**2
        INNER_SIZE_ACTIONS = 64
        INNER_SIZE_SEQUNTIAL = 64
        self.actor = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_ACTIONS),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, INNER_SIZE_ACTIONS),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, n_action_outputs),
        ).to(self.device)

        self.critic = torch.nn.Sequential(
            torch.nn.LayerNorm(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_SEQUNTIAL),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, INNER_SIZE_SEQUNTIAL),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, 1),
        ).to(self.device)

    def _action_distribution(self, state: torch.Tensor):
        action_mean_std = self.actor.forward(state.to(self.device))
        *dims, action_outputs = action_mean_std.shape
        action_mean_std = action_mean_std.view(-1, action_outputs)
        means = action_mean_std[:, : self.n_actions]
        std = torch.exp(action_mean_std[:, self.n_actions :])
        std = std.reshape(-1, self.n_actions, self.n_actions)
        std = torch.clamp(std, min=-1e3, max=1e3)

        # Calculate the Frobenius norm of the original matrix
        norm = torch.norm(std, p="fro", dim=(1, 2), keepdim=True)
        # Normalize the matrix by dividing by its Frobenius norm
        std_normalized_local = std / (norm + 1e-8)
        # Perform the matrix multiplication of the normalized matrix and its transpose
        result_normalized = std_normalized_local @ std_normalized_local.mT + torch.eye(self.n_actions).unsqueeze(0).to(self.device)
        # Scale the result by the original Frobenius norm
        normalized_cov_mat = result_normalized * norm

        # Reshape
        means = means.reshape(*dims, self.n_actions)
        normalized_cov_mat = normalized_cov_mat.reshape(*dims, self.n_actions, self.n_actions)
        # print(torch.linalg.eigvals(cov_mat))
        dist = distributions.MultivariateNormal(means, normalized_cov_mat)
        # dist = distributions.Normal(means, std)
        return dist

    def actor_parameters(self):
        return list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())

    def policy(self, state: torch.Tensor, *args, **kwargs):
        dist = self._action_distribution(state)
        return dist, None

    def value(self, state: torch.Tensor, *args, **kwargs):
        value = self.critic.forward(state)
        return torch.squeeze(value, -1), None

    def to(self, device: torch.device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self


class RNN(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int):
        super().__init__()
        self.n_outputs = n_outputs
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(n_inputs, n_hidden), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=n_hidden, hidden_size=n_hidden, batch_first=True)
        self.fc2 = torch.nn.Linear(n_hidden, n_outputs)

    def forward(self, obs: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.fc1.forward(obs)
        x, hidden_states = self.gru.forward(x, hidden_states)
        x = self.fc2.forward(x)
        return x, hidden_states


class RecurrentActorCritic(ActorCritic):
    def __init__(self, state_size: int, n_actions: int, device: torch.device):
        super().__init__()
        self.hidden_states_actor = None
        self.hidden_states_critic = None
        self.saved_hidden_states = None, None
        self.n_actions = n_actions
        self.device = device
        # Because we output one mean per action and a covariance matrix, we have an output of size n_actions + n_actions**2
        self.n_action_outputs = n_actions + n_actions**2
        self.actor = RNN(n_inputs=state_size, n_outputs=self.n_action_outputs, n_hidden=64).to(self.device)
        self.critic = RNN(n_inputs=state_size, n_outputs=1, n_hidden=64).to(self.device)

    def _action_distribution(self, state: torch.Tensor, hidden_states: Optional[torch.Tensor] = None):
        *dims, _ = state.shape
        action_mean_std, hidden_states = self.actor.forward(state.to(self.device), hidden_states)
        action_mean_std = torch.reshape(action_mean_std, (-1, self.n_action_outputs))
        means = action_mean_std[:, : self.n_actions]
        std = torch.exp(action_mean_std[:, self.n_actions :])
        std = std.reshape(-1, self.n_actions, self.n_actions)
        std = torch.clamp(std, min=-1e3, max=1e3)

        # Calculate the Frobenius norm of the original matrix
        norm = torch.norm(std, p="fro", dim=(1, 2), keepdim=True)
        # Normalize the matrix by dividing by its Frobenius norm
        std_normalized_local = std / (norm + 1e-8)
        # Perform the matrix multiplication of the normalized matrix and its transpose
        result_normalized = std_normalized_local @ std_normalized_local.mT + torch.eye(self.n_actions).unsqueeze(0).to(self.device)
        # Scale the result by the original Frobenius norm
        normalized_cov_mat = result_normalized * norm

        means = means.reshape(*dims, self.n_actions)
        normalized_cov_mat = normalized_cov_mat.reshape(*dims, self.n_actions, self.n_actions)
        dist = distributions.MultivariateNormal(means, normalized_cov_mat)
        return dist, hidden_states

    def save_hidden_states(self):
        self.saved_hidden_states = self.hidden_states_actor, self.hidden_states_critic

    def restore_hidden_states(self):
        self.hidden_states_actor, self.hidden_states_critic = self.saved_hidden_states

    def reset(self):
        self.hidden_states_actor = None
        self.hidden_states_critic = None

    def policy(self, state: torch.Tensor, hx: Optional[torch.Tensor] = None):
        dist, hx = self._action_distribution(state, hx)
        return dist, hx

    def value(self, state: torch.Tensor, hx: Optional[torch.Tensor] = None):
        value, hx = self.critic.forward(state, hx)
        return value.squeeze(-1), hx

    def to(self, device: torch.device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self

    def actor_parameters(self):
        return list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())
