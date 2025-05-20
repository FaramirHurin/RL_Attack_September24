import random
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from marlenv.utils import Schedule

from agents import PPO, RPPO, Agent, VaeAgent
from agents.rl import LinearActorCritic, RecurrentActorCritic
from environment import SimpleCardSimEnv


@dataclass(eq=True)
class CardSimParameters:
    n_days: int = 50
    start_date: str = "2023-01-01"
    n_payers: int = 10_000


@dataclass(eq=True)
class RPPOParameters:
    gamma: float
    lr_actor: float
    lr_critic: float
    n_epochs: int
    eps_clip: float
    critic_c1: Schedule
    entropy_c2: Schedule
    train_interval: int
    gae_lambda: float
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
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
    ):
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.critic_c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.entropy_c2 = entropy_c2
        self.train_interval = train_interval
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping

    def as_dict(self):
        kwargs = asdict(self)
        kwargs["critic_c1"] = self.critic_c1
        kwargs["entropy_c2"] = self.entropy_c2
        return kwargs

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device):
        network = RecurrentActorCritic(env.observation_size, env.n_actions, device)
        return RPPO(network, **self.as_dict(), device=device)


@dataclass(eq=True)
class PPOParameters(RPPOParameters):
    minibatch_size: int

    def __init__(
        self,
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
    ):
        super().__init__(
            gamma=gamma,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            n_epochs=n_epochs,
            eps_clip=eps_clip,
            critic_c1=critic_c1,
            entropy_c2=entropy_c2,
            train_interval=train_interval,
            gae_lambda=gae_lambda,
            grad_norm_clipping=grad_norm_clipping,
        )
        self.minibatch_size = minibatch_size

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device):
        network = LinearActorCritic(env.observation_size, env.n_actions, device)
        return PPO(network, **self.as_dict(), device=device)


@dataclass(eq=True)
class VAEParameters:
    latent_dim: int = 10
    hidden_dim: int = 120
    lr: float = 0.0005
    trees: int = 20
    batch_size: int = 8
    num_epochs: int = 4000
    quantile: float = 0.99
    supervised: bool = False

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device, know_client: bool, quantile: float):
        return VaeAgent(
            device=device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            trees=self.trees,
            banksys=env.system,
            terminal_codes=env.system.terminals[-5:],
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            know_client=know_client,
            supervised=self.supervised,
            current_time=env.t,
            quantile=quantile,
        )


@dataclass(eq=True)
class Parameters:
    agent: PPOParameters | RPPOParameters | VAEParameters
    cardsim: CardSimParameters = field(default_factory=CardSimParameters)
    agent_name: Literal["ppo", "vae", "rppo"] = "ppo"
    n_episodes: int = 4000
    know_client: bool = False
    terminal_fract: float = 1.0
    seed: int = 0
    use_anomaly: bool = True
    n_days_training: int = 30
    avg_card_block_delay_days: int = 7
    quantiles_anomaly: list[float] = field(default_factory=lambda: [0.01, 0.99])
    rules: dict[str, float] = field(
        default_factory=lambda: {
            "max_trx_hour": 6,
            "max_trx_week": 40,
            "max_trx_day": 15,
        }
    )

    def __post_init__(self):
        match self.agent:
            case PPOParameters():
                self.agent_name = "ppo"
            case RPPOParameters():
                self.agent_name = "rppo"
            case VAEParameters():
                self.agent_name = "vae"
            case _:
                raise ValueError("Unknown agent type")
        # Seed the experiment
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device) -> Agent:
        match self.agent:
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.quantiles_anomaly[0])
            case PPOParameters() | RPPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")
