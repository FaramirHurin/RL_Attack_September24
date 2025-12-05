from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import distributions
from torch.distributions import transforms
from marlenv import ContinuousSpace
from utils import tb_log


class ActorCritic(torch.nn.Module, ABC):
    def __init__(self, action_space: ContinuousSpace, use_covariance_matrix: bool, device: torch.device):
        super().__init__()
        self.action_space = action_space
        # Transforms to rescale the actions from [0, 1] to [low, high]
        self.transforms = [
            transforms.SigmoidTransform(),
            transforms.AffineTransform(
                loc=torch.tensor(self.action_space.low, device=device),
                scale=torch.tensor(self.action_space.high - self.action_space.low, device=device),
            ),
        ]
        self.n_actions = action_space.size
        self.device = device
        self.use_covariance_matrix = use_covariance_matrix
        if use_covariance_matrix:
            # Because we output one mean per action and a covariance matrix, we have an output of size n_actions + n_actions**2
            # n_actions for the means
            # n_actions ** 2 for the covariance matrix
            self.output_size = self.n_actions + self.n_actions**2
        else:
            # Output both means and variances for each action
            self.output_size = self.n_actions * 2

    @abstractmethod
    def policy(
        self,
        states: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[distributions.TransformedDistribution, Optional[torch.Tensor]]: ...

    @abstractmethod
    def value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of a batch of states.

        Note: since this is only used during training, there is no possible hidden states to
        maintain, as only full episodes are fed to this function.
        """

    @abstractmethod
    def actor_parameters(self) -> list[torch.nn.Parameter]: ...

    @abstractmethod
    def critic_parameters(self) -> list[torch.nn.Parameter]: ...

    def make_distribution(self, outputs: torch.Tensor):
        """
        Generate a multivariate normal distribution from the outputs of the actor network.
        Ensures that the covariance matrix is a valid one:
         - A @ A^T is symmetric by construction
         - Adding the identity matrix ensures positive definiteness
        """
        *dims, _ = outputs.shape
        outputs = outputs.view(-1, self.output_size)
        means = outputs[:, : self.n_actions]
        means = means.reshape(*dims, self.n_actions)
        if self.use_covariance_matrix:
            # Generate a Positive Definite covariance matrix
            # https://stackoverflow.com/questions/58176501/how-do-you-generate-positive-definite-matrix-in-pytorch
            raw = outputs[:, self.n_actions :]
            raw = raw.reshape(-1, self.n_actions, self.n_actions)
            positive_semi_definite = raw @ raw.transpose(1, 2)
            positive_definite = positive_semi_definite + torch.eye(self.n_actions, device=outputs.device)
            cov = positive_definite.reshape(*dims, self.n_actions, self.n_actions)
            dist = distributions.MultivariateNormal(means, cov)
            tb_log("distribution/cov_min", cov.min().item())
            tb_log("distribution/cov_max", cov.max().item())
            tb_log("distribution/cov_mean", cov.mean().item())
            tb_log("distribution/means_min", means.min().item())
            tb_log("distribution/means_max", means.max().item())
            tb_log("distribution/means_mean", means.mean().item())
        else:
            stds = outputs[:, self.n_actions :]
            stds = torch.nn.functional.softplus(stds)
            stds = stds.reshape(*dims, self.n_actions)
            dist = torch.distributions.Independent(torch.distributions.Normal(means, stds), 1)
        return distributions.TransformedDistribution(dist, self.transforms)


class LinearActorCritic(ActorCritic):
    def __init__(self, state_size: int, action_space: ContinuousSpace, device: torch.device, use_covariance_matrix: bool):
        super().__init__(action_space, use_covariance_matrix, device)
        self.device = device
        INNER_SIZE_ACTIONS = 64
        INNER_SIZE_SEQUNTIAL = 64
        self.actor = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_ACTIONS),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, INNER_SIZE_ACTIONS),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_ACTIONS, self.output_size),
        ).to(self.device)

        self.critic = torch.nn.Sequential(
            # torch.nn.LayerNorm(state_size),
            torch.nn.Linear(state_size, INNER_SIZE_SEQUNTIAL),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, INNER_SIZE_SEQUNTIAL),
            torch.nn.Tanh(),
            torch.nn.Linear(INNER_SIZE_SEQUNTIAL, 1),
        ).to(self.device)

    def actor_parameters(self):
        return list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())

    def policy(self, state: torch.Tensor, *args, **kwargs):
        outputs = self.actor.forward(state.to(self.device))
        dist = self.make_distribution(outputs)
        return dist, None

    def value(self, state: torch.Tensor):
        value = self.critic.forward(state)
        return torch.squeeze(value, -1)

    def to(self, device: torch.device, *args, **kwargs):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self


class RNN(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int):
        super().__init__()
        self.n_outputs = n_outputs
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(n_inputs, n_hidden), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=n_hidden, hidden_size=n_hidden, batch_first=False)
        self.fc2 = torch.nn.Linear(n_hidden, n_outputs)

    def forward(self, obs: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        *dims, n_features = obs.shape
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add a batch dimension
        else:
            debug = True
        seq_length, batch_size, n_features = obs.shape
        x = self.fc1.forward(obs)
        x, hidden_states = self.gru.forward(x, hidden_states)
        assert hidden_states.shape == (1, batch_size, 64)
        x = self.fc2.forward(x)
        return x.view(*dims, -1), hidden_states


class RecurrentActorCritic(ActorCritic):
    def __init__(self, state_size: int, action_space: ContinuousSpace, device: torch.device, use_covariance_matrix: bool):
        super().__init__(action_space, use_covariance_matrix, device)
        self.device = device
        self.actor = RNN(n_inputs=state_size, n_outputs=self.output_size, n_hidden=64).to(self.device)
        self.critic = RNN(n_inputs=state_size, n_outputs=1, n_hidden=64).to(self.device)

    def policy(self, states: torch.Tensor, hx: Optional[torch.Tensor] = None):
        outputs, hx = self.actor.forward(states, hx)
        dist = self.make_distribution(outputs)
        return dist, hx

    def value(self, states: torch.Tensor):
        value, _ = self.critic.forward(states, None)
        return value.squeeze(-1)

    def to(self, device: torch.device, *args, **kwargs):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self

    def actor_parameters(self):
        return list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())
