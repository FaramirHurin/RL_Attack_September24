import random
from dataclasses import asdict, dataclass, replace
from datetime import timedelta, datetime
import os
import orjson
import shutil
from typing import Any, Literal, Optional, Sequence
import logging

import numpy as np
import torch
from marlenv.utils import Schedule

from agents import Agent
from environment import CardSimEnv


@dataclass(eq=True)
class CardSimParameters:
    n_days: int = 365
    start_date: str = "2023-01-01"
    n_payers: int = 10_000

    def get_simulation_data(self):
        from cardsim import Cardsim

        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(
            n_days=self.n_days,
            n_payers=self.n_payers,
            start_date=self.start_date,
        )
        return cards, terminals, transactions

    @staticmethod
    def paper_params():
        return CardSimParameters(
            n_days=365 * 2 + 150 + 30,  # 2 years budget + 150 days training + 30 days warmup
            n_payers=20_000,
            start_date="2023-01-01",
        )


@dataclass(eq=True)
class ClassificationParameters:
    use_anomaly: bool
    n_trees: int
    balance_factor: float
    contamination: float
    training_duration: timedelta
    quantiles_features: Sequence[str]
    quantiles_values: Sequence[float]
    rules: dict[str, float]

    def __init__(
        self,
        use_anomaly: bool = True,
        n_trees: int = 100,
        balance_factor: float = 0.05,
        contamination: float = 0.005,
        training_duration: timedelta | float = timedelta(days=30),
        quantiles_features: Sequence[str] = ("amount",),
        quantiles_values: Sequence[float] = (0.01, 0.99),
        rules: dict[str, float] = {
            "max_trx_hour": 6,
            "max_trx_week": 40,
            "max_trx_day": 15,
        },
    ):
        self.use_anomaly = use_anomaly
        self.n_trees = n_trees
        self.balance_factor = balance_factor
        self.contamination = contamination
        if isinstance(training_duration, (float, int)):
            training_duration = timedelta(seconds=training_duration)
        self.training_duration = training_duration
        self.quantiles_features = quantiles_features
        self.quantiles_values = quantiles_values
        self.rules = rules

    @staticmethod
    def paper_params():
        return ClassificationParameters(
            use_anomaly=False,
            n_trees=100,
            balance_factor=0.05,
            contamination=0.005,
            training_duration=timedelta(days=150),
            quantiles_features=("amount",),
            quantiles_values=(0.01, 1.0),
            rules={
                "max_trx_hour": 6,
                "max_trx_week": 40,
                "max_trx_day": 15,
            },
        )


@dataclass(eq=True)
class PPOParameters:
    gamma: float
    lr_actor: float
    lr_critic: float
    n_epochs: int
    eps_clip: float
    critic_c1: Schedule
    entropy_c2: Schedule
    train_interval: int
    minibatch_size: int
    gae_lambda: float
    grad_norm_clipping: Optional[float]
    train_on: Literal["transition", "episode"]
    is_recurrent: bool

    def __init__(
        self,
        is_recurrent: bool = False,
        train_on: Literal["transition", "episode"] = "transition",
        gamma: float = 0.99,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 20,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        entropy_c2: Schedule | float = 0.01,
        train_interval: int = 64,
        minibatch_size: int = 32,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
    ):
        self.is_recurrent = is_recurrent
        if self.is_recurrent and not train_on == "episode":
            logging.warning("Recurrent PPO is only supported for episode training. Switching to episode training.")
            train_on = "episode"
        self.train_on = train_on
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
        self.minibatch_size = minibatch_size
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping

    def as_dict(self):
        kwargs = asdict(self)
        kwargs["critic_c1"] = self.critic_c1
        kwargs["entropy_c2"] = self.entropy_c2
        return kwargs

    @staticmethod
    def from_json(data: dict[str, Any]):
        """
        Create PPOParameters from a JSON-like dictionary.
        """
        data["critic_c1"] = schedule_from_json(data["critic_c1"])
        data["entropy_c2"] = schedule_from_json(data["entropy_c2"])
        return PPOParameters(**data)

    def get_agent(self, env: CardSimEnv, device: torch.device):
        # from agents import RPPO
        from agents.rl.replay_memory import TransitionMemory, EpisodeMemory
        from agents.rl.ppo import PPO
        from agents.rl.networks import RecurrentActorCritic, LinearActorCritic

        match self.train_on:
            case "transition":
                memory = TransitionMemory(self.train_interval)
            case "episode":
                memory = EpisodeMemory(self.train_interval)
            case _:
                raise ValueError(f"Unknown value for `train_on`: {self.train_on}")
        if self.is_recurrent:
            network = RecurrentActorCritic(env.observation_size, env.n_actions, device)
        else:
            network = LinearActorCritic(env.observation_size, env.n_actions, device)
        return PPO(network, memory, **self.as_dict(), device=device)

    @staticmethod
    def best_rppo():
        """
        The result of the hyperparameter tuning with Optuna for recurrent PPO.
        """
        return PPOParameters(
            True,
            "episode",
            gamma=0.99,
            lr_actor=0.0009169278258635868,
            lr_critic=0.0005058375638259988,
            grad_norm_clipping=2.548454926359372,
            n_epochs=50,
            train_interval=13,
            minibatch_size=4,
            critic_c1=Schedule.linear(
                start_value=0.5640469966895131,
                end_value=0.059606970056594356,
                n_steps=2017,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.05257108712492839,
                end_value=0.032373700129899374,
                n_steps=2602,
            ),
        )

    @staticmethod
    def best_ppo():
        """
        The result of the hyperparameter tuning with Optuna for standard PPO (non-recurrent).
        """
        return PPOParameters(
            is_recurrent=False,
            train_on="transition",
            gamma=0.99,
            lr_actor=0.0009830993791440257,
            lr_critic=0.000995558928022473,
            n_epochs=26,
            eps_clip=0.2,
            critic_c1=Schedule.linear(
                start_value=0.4943612898407241,
                end_value=0.09730954213676407,
                n_steps=2614,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.08127702555893541,
                end_value=0.014990199238406538,
                n_steps=3982,
            ),
            train_interval=11,
            minibatch_size=4,
            gae_lambda=0.95,
            grad_norm_clipping=9.319870685466327,
        )


@dataclass(eq=True)
class VAEParameters:
    latent_dim: int = 10
    hidden_dim: int = 120
    lr: float = 0.0005
    trees: int = 20
    batch_size: int = 8
    num_epochs: int = 4000
    quantile: float = 0.95
    supervised: bool = False

    def get_agent(self, env: CardSimEnv, device: torch.device, know_client: bool, quantile: float):
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

    @staticmethod
    def best_vae():
        # [latent_dim: 55, hidden_dim: 172, lr: 0.00028782207302075277, trees: 90, batch_size: 22, quantile: 0.9953256365516118, num_epochs: 7488]
        return VAEParameters(
            latent_dim=55,
            hidden_dim=172,
            lr=0.00028782207302075277,
            trees=90,
            batch_size=22,
            num_epochs=7488,
            quantile=0.9953256365516118,
            supervised=False,
        )


@dataclass(eq=True)
class Parameters:
    agent: PPOParameters | VAEParameters
    cardsim: CardSimParameters
    clf_params: ClassificationParameters
    n_episodes: int
    know_client: bool
    terminal_fract: float
    seed_value: int
    card_pool_size: int
    avg_card_block_delay_days: int
    logdir: str
    aggregation_windows: Sequence[timedelta]
    agent_name: Literal["ppo", "rppo", "vae"]

    def __init__(
        self,
        agent: PPOParameters | VAEParameters,
        cardsim: CardSimParameters = CardSimParameters(),
        clf_params: ClassificationParameters = ClassificationParameters(),
        n_episodes: int = 4000,
        know_client: bool = False,
        terminal_fract: float = 0.1,
        seed_value: Optional[int] = None,
        card_pool_size: int = 10,
        avg_card_block_delay_days: int = 7,
        logdir: Optional[str] = None,
        save: bool = True,
        aggregation_windows: Sequence[timedelta | float] = (timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        **kwargs,
    ):
        kwargs.pop("agent_name", None)  # agent_name is set automatically with the "repeat" method
        if len(kwargs) > 0:
            logging.warning(f"Unknown parameters: {kwargs}. They will be ignored.")
        self.agent = agent
        self.cardsim = cardsim
        self.n_episodes = n_episodes
        self.know_client = know_client
        self.terminal_fract = terminal_fract
        if seed_value is None:
            seed_value = hash(datetime.now().isoformat()) % 2**32
        self.seed_value = seed_value
        self.avg_card_block_delay_days = avg_card_block_delay_days
        self.clf_params = clf_params
        self.card_pool_size = card_pool_size
        self.aggregation_windows = []
        for window in aggregation_windows:
            if isinstance(window, (float, int)):
                window = timedelta(seconds=window)
            self.aggregation_windows.append(window)
        match self.agent:
            case PPOParameters():
                if self.agent.is_recurrent:
                    self.agent_name = "rppo"
                else:
                    self.agent_name = "ppo"
            case VAEParameters():
                self.agent_name = "vae"
            case _:
                raise ValueError("Unknown agent type")
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

    def create_agent(self, env: CardSimEnv, device: Optional[torch.device] = None) -> Agent:
        self.seed()
        if device is None:
            device = self.get_device_by_seed()
        match self.agent:
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.clf_params.quantiles_values[0])
            case PPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")

    def create_pooled_env(self):
        from banksys import Banksys

        try:
            banksys = Banksys.load(self.cardsim, self.banksys_dir)
        except (FileNotFoundError, ValueError):
            print("Banksys not found, creating a new one")
            banksys = self.create_banksys()

        banksys.set_up_run(rules_values=self.clf_params.rules, use_anomaly=self.clf_params.use_anomaly)
        env = CardSimEnv(
            banksys,
            timedelta(days=self.avg_card_block_delay_days),
            customer_location_is_known=self.know_client,
            normalize_location=self.agent_name in ("ppo", "rppo"),
        )
        env.seed(self.seed_value)
        return env

    def banksys_is_in_cache(self):
        """
        Check if the Banksys directory exists.
        """
        return os.path.exists(self.banksys_dir)

    @property
    def banksys_dir(self):
        return os.path.join(
            "cache",
            "banksys",
            f"{self.cardsim.n_payers}-payers",
            f"{self.cardsim.n_days}-days",
            f"start-{self.cardsim.start_date}",
        )

    def create_banksys(self, save: bool = True):
        from banksys import Banksys

        cards, terminals, transactions = self.cardsim.get_simulation_data()
        banksys = Banksys(
            cards=cards,
            terminals=terminals,
            aggregation_windows=self.aggregation_windows,
            attackable_terminal_factor=self.terminal_fract,
            clf_params=self.clf_params,
        )
        banksys.fit(transactions)
        if save:
            banksys.save(self.cardsim, self.banksys_dir)
        return banksys

    def get_device_by_seed(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        device = f"cuda:{self.seed_value % torch.cuda.device_count()}"
        return torch.device(device)

    def save(self):
        if self.logdir in ("test", "debug", "logs/test", "log/tests", "log/debug"):
            try:
                shutil.rmtree(self.logdir)
            except FileNotFoundError:
                pass
        os.makedirs(self.logdir, exist_ok=True)
        file_path = os.path.join(self.logdir, "params.json")
        logging.debug(file_path)
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(self, default=serialize_unknown))

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

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            data = orjson.loads(f.read())
        assert isinstance(data, dict), "Parameters should be a dictionary"
        match data["agent_name"]:
            case "ppo" | "rppo":
                data["agent"] = PPOParameters.from_json(data["agent"])
            case "vae":
                data["agent"] = VAEParameters(**data["agent"])
            case _:
                raise ValueError(f"Unknown agent type: {data['agent_name']}")
        data["cardsim"] = CardSimParameters(**data["cardsim"])
        data["clf_params"] = ClassificationParameters(**data["clf_params"])
        return Parameters(**data)


def schedule_from_json(data: dict[str, Any]):
    """Create a Schedule from a JSON-like dictionary."""
    classname = data["name"]
    if classname == "LinearSchedule":
        return Schedule.linear(data["start_value"], data["end_value"], data["n_steps"])
    elif classname == "ExpSchedule":
        return Schedule.exp(data["start_value"], data["end_value"], data["n_steps"])
    elif classname == "ConstantSchedule":
        return Schedule.constant(data["value"])
    raise NotImplementedError(f"Unsupported deserialization for schedule type: {classname}")


def serialize_unknown(data):
    match data:
        case timedelta():
            return data.total_seconds()
    raise NotImplementedError(f"Unsupported serialization for type: {type(data)}. Value={data}")
