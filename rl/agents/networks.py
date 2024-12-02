import torch
from torch import distributions
from datetime import datetime

import torch.nn as nn


class PositiveDefiniteMatrixGenerator(nn.Module):
    def __init__(self, input_size, matrix_size):
        super(PositiveDefiniteMatrixGenerator, self).__init__()
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


class ActorCritic(torch.nn.Module):
    def __init__(self, state_size: int, n_actions: int):
        super(ActorCritic, self).__init__()
        self.n_actions = n_actions
        self.device = torch.device("cuda")
        # Because we output one mean per action and a covariance matrix, we have an output of size n_actions + n_actions**2
        n_action_outputs = n_actions + n_actions**2
        INNER_SIZE_ACTIONS = 32
        INNER_SIZE_SEQUNTIAL = 32
        self.actions_mean_std = torch.nn.Sequential(
            # torch.nn.LayerNorm(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_ACTIONS),
            torch.nn.ReLU(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, INNER_SIZE_ACTIONS),
            torch.nn.ReLU(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, n_action_outputs),
        ).to(self.device)

        self.critic = torch.nn.Sequential(
            torch.nn.LayerNorm(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_SEQUNTIAL),
            torch.nn.ReLU(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, INNER_SIZE_SEQUNTIAL),
            torch.nn.ReLU(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, 1),
        ).to(self.device)

    def forward(self):
        raise NotImplementedError()

    def _action_distribution(self, state: torch.Tensor):
        action_mean_std = self.actions_mean_std(state.to(self.device))
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

        if torch.any(normalized_cov_mat.isnan()):
            print(f"[{datetime.now()}]Nan encountered for input: {state}")
            print(f"\tmeans: {means}")
            print(f"\tstd: {std}")
            print(f"\tnorm: {norm}")
            print(f"\tstd_normalized_local: {std_normalized_local}")
            print(f"\tresult_normalized: {result_normalized}")
            print(f"\tnormalized_cov_mat: {normalized_cov_mat}")
            # normalized_cov_mat = torch.ones_like(normalized_cov_mat)
            raise Exception("Nan encountered")

        """
        cov_mat = std @ std.mT
        cov_mat = cov_mat +  torch.eye(self.n_actions).unsqueeze(0)

        # Calculate the determinant for each matrix in the batch
        det = torch.det(cov_mat)
        # normalizer = cov_mat.min()
        #    Normalize by dividing each matrix by its determinant, with epsilon to avoid division by zero
        epsilon = 1e-8
        normalized_cov_mat = cov_mat / (det.unsqueeze(-1).unsqueeze(-1) + epsilon)
        """

        # print(torch.linalg.eigvals(cov_mat))
        dist = distributions.MultivariateNormal(means, normalized_cov_mat)
        # dist = distributions.Normal(means, std)
        return dist

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.unsqueeze(0).to(self.device)
        dist = self._action_distribution(state)
        action = dist.sample().squeeze(0)
        action_logprob = dist.log_prob(action).squeeze(0)
        return action, action_logprob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._action_distribution(state.to(self.device))
        if len(action.size()) == 1:
            action = action.unsqueeze(-1)
        action_logprob = dist.log_prob(action.to(self.device))
        entropy = dist.entropy()
        return action_logprob, entropy

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x.to(self.device))

    def q_value(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get the q-value estimated for the given state and action"""
        raise NotImplementedError()

    def to(self, device: torch.device):
        self.device = device
        self.actions_mean_std.to(device)
        self.critic.to(device)
        return self
