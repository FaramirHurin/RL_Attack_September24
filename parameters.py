from dataclasses import dataclass, asdict, field
from typing import Literal, Optional
from marlenv.utils import Schedule
import torch
from agents import PPO, RPPO, Agent
from agents.rl import LinearActorCritic, RecurrentActorCritic
from Baselines.attack_generation import VaeAgent
from environment import SimpleCardSimEnv


@dataclass
class CardSimParameters:
    n_days: int = 50
    start_date: str = "2023-01-01"
    n_payers: int = 10_000


@dataclass
class RPPOParameters:
    gamma: float = 0.99
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    n_epochs: int = 20
    eps_clip: float = 0.2
    critic_c1: Schedule = field(default_factory=lambda: Schedule.constant(0.5))
    entropy_c2: Schedule = field(default_factory=lambda: Schedule.constant(0.01))
    train_interval: int = 64
    gae_lambda: float = 0.95
    grad_norm_clipping: Optional[float] = None

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device):
        network = RecurrentActorCritic(env.observation_size, env.n_actions, device)
        return RPPO(network, **asdict(self), device=device)


@dataclass
class PPOParameters(RPPOParameters):
    minibatch_size: int = 10

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device):
        network = LinearActorCritic(env.observation_size, env.n_actions, device)
        return PPO(network, **asdict(self), device=device)


@dataclass
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
        TERMINALS = env.system.terminals[-5:]
        return VaeAgent(
            device=device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            trees=self.trees,
            banksys=env.system,
            terminal_codes=[t for t in TERMINALS],
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            know_client=know_client,
            supervised=self.supervised,
            current_time=env.t,
            quantile=quantile,
        )


@dataclass
class Parameters:
    agent: PPOParameters | RPPOParameters | VAEParameters
    cardsim: CardSimParameters = field(default_factory=CardSimParameters)
    agent_name: Literal["ppo", "vae", "rppo"] = "ppo"
    n_episodes: int = 4000
    know_client: bool = False
    terminal_fract: int = 1
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

    def get_agent(self, env: SimpleCardSimEnv, device: torch.device) -> Agent:
        match self.agent:
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.quantiles_anomaly[0])
            case PPOParameters() | RPPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")
