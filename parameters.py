import random
from dataclasses import asdict, dataclass, replace
from datetime import timedelta, datetime
import os
import orjson
from typing import Optional

import numpy as np
import torch
from marlenv.utils import Schedule

from agents import Agent
from agents.rl import LinearActorCritic, RecurrentActorCritic
from cardsim import Cardsim
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
        from agents import RPPO

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
        from agents import PPO

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
        from agents import VaeAgent

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
    cardsim: CardSimParameters
    n_episodes: int
    know_client: bool
    terminal_fract: float
    seed_value: int
    use_anomaly: bool
    n_days_training: int
    avg_card_block_delay_days: int
    quantiles_anomaly: list[float]
    rules: dict[str, float]
    logdir: str

    def __init__(
        self,
        agent: PPOParameters | RPPOParameters | VAEParameters,
        cardsim: CardSimParameters = CardSimParameters(),
        n_episodes: int = 4000,
        know_client: bool = False,
        terminal_fract: float = 1.0,
        seed_value: int = 0,
        use_anomaly: bool = True,
        n_days_training: int = 30,
        avg_card_block_delay_days: int = 7,
        quantiles_anomaly: list[float] = [0.01, 0.99],
        rules: dict[str, float] = {
            "max_trx_hour": 6,
            "max_trx_week": 40,
            "max_trx_day": 15,
        },
        logdir: Optional[str] = None,
        save: bool = True,
    ):
        self.agent = agent
        self.cardsim = cardsim
        self.n_episodes = n_episodes
        self.know_client = know_client
        self.terminal_fract = terminal_fract
        self.seed_value = seed_value
        self.use_anomaly = use_anomaly
        self.n_days_training = n_days_training
        self.avg_card_block_delay_days = avg_card_block_delay_days
        self.quantiles_anomaly = quantiles_anomaly
        self.rules = rules
        if logdir is None:
            logdir = self.default_logdir()
        self.logdir = logdir
        if save:
            self.save()

    def seed(self):
        """
        Seed the random number generator.
        """
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)

    def create_agent(self, env: SimpleCardSimEnv) -> Agent:
        self.seed()
        device = self.get_device_by_seed()
        match self.agent:
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.quantiles_anomaly[0])
            case PPOParameters() | RPPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")

    def create_env(self):
        from banksys import Banksys

        try:
            banksys = Banksys.load(self.cardsim, self.banksys_dir)
        except (FileNotFoundError, ValueError):
            print("Banksys not found, creating a new one")
            banksys = self.create_banksys()
            banksys.save(self.cardsim)

        banksys.set_up_run(rules_values=self.rules, use_anomaly=self.use_anomaly)
        env = SimpleCardSimEnv(
            banksys,
            timedelta(days=self.avg_card_block_delay_days),
            customer_location_is_known=self.know_client,
            normalize_location=self.agent_name in ("ppo", "rppo"),
        )
        env.seed(self.seed_value)
        return env

    @property
    def banksys_dir(self):
        return os.path.join(
            "cache",
            "banksys",
            f"{self.cardsim.n_payers}-payers",
            f"{self.cardsim.n_days}-days",
            f"start-{self.cardsim.start_date}",
        )

    def create_banksys(self):
        from banksys import Banksys

        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(
            n_days=self.cardsim.n_days,
            n_payers=self.cardsim.n_payers,
            start_date=self.cardsim.start_date,
        )
        banksys = Banksys(
            cards=cards,
            terminals=terminals,
            training_duration=timedelta(days=self.n_days_training),
            transactions=transactions,
            feature_names=["amount"],
            quantiles=self.quantiles_anomaly,
            attackable_terminal_factor=self.terminal_fract,
        )
        banksys.save(self.cardsim, self.banksys_dir)
        return banksys

    def get_device_by_seed(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        device = f"cuda:{self.seed_value % torch.cuda.device_count()}"
        return torch.device(device)

    def save(self):
        os.makedirs(self.logdir, exist_ok=True)
        file_path = os.path.join(self.logdir, "params.json")
        print(file_path)
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(self))

    @property
    def agent_name(self):
        match self.agent:
            case PPOParameters():
                return "ppo"
            case RPPOParameters():
                return "rppo"
            case VAEParameters():
                return "vae"
            case _:
                raise ValueError("Unknown agent type")

    def default_logdir(self):
        timestamp = datetime.now().isoformat().replace(":", "-")
        return os.path.join("logs", self.agent_name, timestamp)

    def repeat(self, n: int):
        """
        Repeat the parameters n times, with different seeds.
        """
        for i in range(n):
            logdir = os.path.join(self.logdir, f"seed-{self.seed_value + i}")
            os.makedirs(logdir, exist_ok=True)
            yield replace(self, seed_value=self.seed_value + i, save=False, logdir=logdir)
