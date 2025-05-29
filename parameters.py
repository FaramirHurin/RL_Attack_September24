import random
from dataclasses import asdict, dataclass, replace
from datetime import timedelta, datetime
import os
from optuna import Trial
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
        return PPOParameters(
            is_recurrent=True,
            train_on="episode",
            gamma=0.99,
            lr_actor=0.0013655647166021928,
            lr_critic=0.007255685546096761,
            n_epochs=27,
            eps_clip=0.2,
            critic_c1=Schedule.linear(
                start_value=0.9375751577962954,
                end_value=0.38048446480609044,
                n_steps=3127,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.0957619650038549,
                end_value=0.007744880113458132,
                n_steps=2537,
            ),
            train_interval=10,
            minibatch_size=8,
            gae_lambda=0.95,
            grad_norm_clipping=8.934885848478487,
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
            lr_actor=0.00117126625357408,
            lr_critic=0.0007648237767940683,
            n_epochs=21,
            eps_clip=0.2,
            critic_c1=Schedule.linear(
                start_value=0.16450187834828542,
                end_value=0.45697380802021975,
                n_steps=3291,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.08774192037356557,
                end_value=0.017361163706258554,
                n_steps=1171,
            ),
            train_interval=22,
            minibatch_size=20,
            gae_lambda=0.95,
            grad_norm_clipping=None,
        )

    @staticmethod
    def suggest_rppo(trial: Trial):
        train_interval = trial.suggest_int("train_interval", 4, 64)
        minibatch_size = trial.suggest_int("minibatch_size", 2, train_interval)
        enable_clipping = trial.suggest_categorical("enable_clipping", [True, False])
        if enable_clipping:
            grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 10)
        else:
            grad_norm_clipping = None
        return PPOParameters(
            is_recurrent=True,
            train_on="episode",
            critic_c1=Schedule.linear(
                trial.suggest_float("critic_c1_start", 0.1, 1.0),
                trial.suggest_float("critic_c1_end", 0.001, 0.5),
                trial.suggest_int("critic_c1_steps", 1000, 4000),
            ),
            entropy_c2=Schedule.linear(
                trial.suggest_float("entropy_c2_start", 0.001, 0.2),
                trial.suggest_float("entropy_c2_end", 0.0001, 0.1),
                trial.suggest_int("entropy_c2_steps", 1000, 4000),
            ),
            n_epochs=trial.suggest_int("n_epochs", 10, 100),
            minibatch_size=minibatch_size,
            train_interval=train_interval,
            lr_actor=trial.suggest_float("lr_actor", 0.0001, 0.01),
            lr_critic=trial.suggest_float("lr_critic", 0.0001, 0.01),
            grad_norm_clipping=grad_norm_clipping,
        )

    @staticmethod
    def suggest_ppo(trial: Trial):
        params = PPOParameters.suggest_rppo(trial)
        params.is_recurrent = False
        if trial.suggest_categorical("train_on_episode", [True, False]):
            params.train_on = "episode"
        else:
            params.train_on = "transition"
        return params


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
    generated_size: int = 1000
    n_infiltrated_terminals: int = 5

    def get_agent(self, env: CardSimEnv, device: torch.device, know_client: bool, quantile: float):
        from agents import VaeAgent

        return VaeAgent(
            device=device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            trees=self.trees,
            banksys=env.system,
            terminal_codes=env.system.terminals[-self.n_infiltrated_terminals :],
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            know_client=know_client,
            supervised=self.supervised,
            current_time=env.t,
            quantile=quantile,
            generated_size=self.generated_size,
        )

    @staticmethod
    def best_vae():
        # Best 0 [latent_dim: 6, hidden_dim: 140, lr: 0.00046673940763915635, trees: 84, batch_size: 27, num_epochs: 9238, quantile: 0.9001873838227034]
        # Best 1 latent_dim: 2, hidden_dim: 157, lr: 0.0007161633748676655, trees: 54, batch_size: 29, num_epochs: 6672, quantile: 0.9844833640628634, generated_size: 970
        return VAEParameters(
            n_infiltrated_terminals=100,
            latent_dim=6,
            hidden_dim=128,
            lr=0.0046673940763915635,
            trees=84,
            batch_size=27,
            num_epochs=9238,
            quantile=0.9001873838227034,
            supervised=False,
            generated_size=1000,
        )

    @staticmethod
    def suggest(trial: Trial):
        return VAEParameters(
            latent_dim=trial.suggest_int("latent_dim", 2, 64),
            hidden_dim=trial.suggest_int("hidden_dim", 64, 192),
            lr=trial.suggest_float("lr", 0.0001, 0.001),
            trees=trial.suggest_int("trees", 20, 100),
            batch_size=trial.suggest_int("batch_size", 8, 32),
            num_epochs=trial.suggest_int("num_epochs", 2_000, 10_000),
            quantile=trial.suggest_float("quantile", 0.9, 0.999),
            generated_size=trial.suggest_int("generated_size", 100, 1000),
        )


@dataclass(eq=True)
class Parameters:
    agent: PPOParameters | VAEParameters | None
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
    agent_name: Literal["ppo", "rppo", "vae", ""]

    def __init__(
        self,
        agent: PPOParameters | VAEParameters | None = None,
        cardsim: CardSimParameters = CardSimParameters(),
        clf_params: ClassificationParameters = ClassificationParameters(),
        n_episodes: int = 4000,
        know_client: bool = False,
        terminal_fract: float = 0.2,
        seed_value: Optional[int] = None,
        card_pool_size: int = 50,
        avg_card_block_delay_days: int = 7,
        logdir: Optional[str] = None,
        save: bool = True,
        aggregation_windows: Sequence[timedelta | float] =
        (timedelta(days=1), timedelta(days=7), timedelta(days=30)),
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
            case None:
                self.agent_name = ""
            case _:
                raise ValueError("Unknown agent type")
        if logdir is None:
            logdir = self.default_logdir()
        self.logdir = logdir
        self.seed()
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
        if device is None:
            device = self.get_device_by_seed()
        match self.agent:
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.clf_params.quantiles_values[0])
            case PPOParameters():
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

        banksys.set_up_run(rules_values=self.clf_params.rules, use_anomaly=self.clf_params.use_anomaly)
        env = CardSimEnv(
            system=banksys,
            avg_card_block_delay=timedelta(days=self.avg_card_block_delay_days),
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

    def create_banksys(self, save: bool = True, save_directory: Optional[str] = None):
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
            if save_directory is None:
                save_directory = self.banksys_dir
            banksys.save(self.cardsim, save_directory)
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
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(self, default=serialize_unknown))

    def default_logdir(self):
        timestamp = datetime.now().isoformat().replace(":", "-")
        return os.path.join("logs", self.agent_name, timestamp)

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
