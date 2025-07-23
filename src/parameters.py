import random
from dataclasses import asdict, dataclass
import pandas as pd
import polars as pl
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

CARD_POOL_SIZE = 100


@dataclass(eq=True)
class CardSimParameters:
    n_days: int = 365
    start_date: str = "2023-01-01"
    n_payers: int = 10_000

    def get_simulation_data(self, use_cache: bool = True, ulb_data=False):
        from cardsim import Cardsim

        if ulb_data:
            transactions = pd.read_csv("MLG_Simulator/transactions.csv")
            cards = pd.read_csv("MLG_Simulator/customer_profiles.csv")
            terminals = pd.read_csv("MLG_Simulator/terminal_profiles.csv")
        else:
            simulator = Cardsim()
            transactions, cards, terminals = simulator.simulate(
                n_days=self.n_days,
                n_payers=self.n_payers,
                start_date=self.start_date,
                use_cache=use_cache,
            )
        return transactions, cards, terminals

    @staticmethod
    def paper_params():
        """
        - n_days: 365 * 2 + 150 + 30
        - n_payers: 20_000
        - start_date: "2023-01-01"
        """
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
    contamination: float | Literal["auto"]
    training_duration: timedelta
    quantiles: dict[str, tuple[float, float]]
    _rules: dict[float, float]

    def __init__(
        self,
        use_anomaly: bool = True,
        n_trees: int = 50,
        balance_factor: float = 0.1,
        contamination: float | Literal["auto"] = "auto",
        training_duration: timedelta | float = timedelta(days=30),
        quantiles: dict[str, tuple[float, float]] = {"amount": (0.01, 0.99)},
        rules: dict[timedelta, float] = {
            timedelta(hours=1): 6,
            timedelta(days=1): 16,
            timedelta(weeks=1): 30,
        },
    ):
        self.use_anomaly = use_anomaly
        self.n_trees = n_trees
        self.balance_factor = balance_factor
        self.contamination = contamination
        if isinstance(training_duration, (float, int)):
            training_duration = timedelta(seconds=training_duration)
        self.training_duration = training_duration
        self.quantiles = quantiles
        self._rules = {td.total_seconds(): value for td, value in rules.items()}

    @property
    def rules(self) -> dict[timedelta, float]:
        """
        Returns the rules as a dictionary with timedelta keys.
        """
        return {timedelta(seconds=key): value for key, value in self._rules.items()}

    @staticmethod
    def paper_params(anomaly: bool):
        """
        - max_trx_hour: 8
        - max_trx_day: 19
        - max_trx_week: 32
        - n_trees: 139
        - balance_factor: 0.06473635736763925
        - quantiles_amount_high: 0.9976319783361984
        - quantiles_risk_high: 0.9999572867664103
        - training_duration: 150 days
        - contamination: "auto"
        - rules:
            - hourly: 8
            - daily: 19
            - weekly: 32
        """
        return ClassificationParameters(
            use_anomaly=anomaly,
            n_trees=139,
            balance_factor=0.06473635736763925,
            contamination="auto",
            training_duration=timedelta(days=150),
            quantiles={
                "amount": (0.0, 0.9976319783361984),
                "terminal_risk_last_1 day, 0:00:00": (0.0, 0.9999572867664103),
            },
            rules={
                timedelta(hours=1): 5,  # 8,
                timedelta(days=1): 12,  # 19,
                timedelta(weeks=1): 32,
            },
        )

    @staticmethod
    def suggest(trial: Trial, training_duration: timedelta):
        max_per_hour = trial.suggest_int("max_trx_hour", 2, 10)
        max_per_day = trial.suggest_int("max_trx_day", max_per_hour, 20)
        max_per_week = trial.suggest_int("max_trx_week", max_per_day, 50)
        return ClassificationParameters(
            training_duration=training_duration,
            n_trees=trial.suggest_int("n_trees", 20, 200),
            contamination="auto",
            balance_factor=trial.suggest_float("balance_factor", 0.05, 0.2),
            quantiles={
                "amount": (0, trial.suggest_float("quantiles_amount_high", 0.995, 1.0)),
                f"terminal_risk_last_{timedelta(days=1)}": (0, trial.suggest_float("quantiles_risk_high", 0.995, 1.0)),
            },
            use_anomaly=False,
            rules={
                timedelta(hours=1): max_per_hour,
                timedelta(days=1): max_per_day,
                timedelta(weeks=1): max_per_week,
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
    normalize_rewards: bool
    normalize_advantages: bool

    def __init__(
        self,
        is_recurrent: bool = False,
        train_on: Literal["transition", "episode"] = "transition",
        gamma: float = 0.99,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 20,
        eps_clip: float = 0.5,
        critic_c1: Schedule | float = 0.5,
        entropy_c2: Schedule | float = 0.01,
        train_interval: int = 64,
        minibatch_size: int = 32,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
        normalize_rewards: bool = True,
        normalize_advantages: bool = True,
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
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages

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
        data["entropy_c2"] = schedule_from_json(data["entropy_c2"])  #
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
            lr_actor=0.007751751648130268,
            lr_critic=0.003790033882253389,
            n_epochs=52,
            eps_clip=0.5,
            critic_c1=Schedule.linear(
                start_value=0.4105552831006898,
                end_value=0.3271347177719041,
                n_steps=2728,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.19110949972090585,
                end_value=0.030016369088242106,
                n_steps=1699,
            ),
            train_interval=6,
            minibatch_size=5,
            gae_lambda=0.99,
            grad_norm_clipping=2.388555590580865,
            normalize_rewards=False,
            normalize_advantages=False,
        )

        """
        - train_interval: 6
        - minibatch_size: 5
        - enable_clipping: True
        - grad_norm_clipping: 2.388555590580865
        - critic_c1_start: 0.4105552831006898
        - critic_c1_end: 0.3271347177719041
        - critic_c1_steps: 2728
        - entropy_c2_start: 0.19110949972090585
        - entropy_c2_end: 0.030016369088242106
        - entropy_c2_steps: 1699
        - n_epochs: 52
        - lr_actor: 0.007751751648130268
        - lr_critic: 0.003790033882253389
        - normalize_rewards: False
        - normalize_advantages: False
        """

    """            =Schedule.linear(
            start_value=0.19110949972090585,
            end_value=0.030016369088242106,
            n_steps=1699,
        ),"""

    @staticmethod
    def best_ppo():
        """
        The result of the hyperparameter tuning with Optuna for standard PPO (non-recurrent).
        """
        return PPOParameters(
            is_recurrent=False,
            train_on="transition",
            train_interval=50,  # 63
            minibatch_size=40,  # 47
            grad_norm_clipping=None,
            critic_c1=Schedule.linear(
                start_value=0.9210682011725766,
                end_value=0.277828843096964265,
                n_steps=2980,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.25,  # 0.15586561621061853,
                end_value=0.05,  # 0.08458724795592026,
                n_steps=2012,
            ),
            n_epochs=15,
            lr_actor=0.000956264649262804,
            lr_critic=0.006671638920039944,
            normalize_rewards=True,
            normalize_advantages=True,
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
            normalize_rewards=trial.suggest_categorical("normalize_rewards", [True, False]),
            normalize_advantages=trial.suggest_categorical("normalize_advantages", [True, False]),
        )

    @staticmethod
    def suggest_ppo(trial: Trial):
        params = PPOParameters.suggest_rppo(trial)
        params.is_recurrent = False
        params.train_on = "transition"
        return params


@dataclass(eq=True)
class VAEParameters:
    latent_dim: int = 10
    hidden_dim: int = 50
    lr: float = 0.0005
    trees: int = 20
    batch_size: int = 32
    num_epochs: int = 2000
    quantile: float = 0.99
    supervised: bool = False
    generated_size: int = 3000
    n_infiltrated_terminals: int = 100
    beta: float = 0.2

    def get_agent(self, env: CardSimEnv, device: torch.device, know_client: bool, quantile: float):
        from agents import VaeAgent

        infiltrated_terminals = random.choices(env.system.terminals, k=self.n_infiltrated_terminals)
        return VaeAgent(
            device=device,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            trees=self.trees,
            banksys=env.system,
            terminal_codes=infiltrated_terminals,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            know_client=know_client,
            supervised=self.supervised,
            current_time=env.t,
            quantile=quantile,
            generated_size=self.generated_size,
            beta=self.beta,
        )

    @staticmethod
    def best_vae():
        """
        - latent_dim: 86
        - hidden_dim: 106
        - lr: 0.0004961629040757451
        - batch_size: 10
        - num_epochs: 2791
        - quantile: 0.9946175749502564
        - generated_size: 541
        - beta: 0.25391071673841914
        - n_infiltrated_terminals: 82
        """
        return VAEParameters(
            latent_dim=86,
            hidden_dim=106,
            lr=0.0004961629040757451,
            trees=20,
            batch_size=10,
            num_epochs=2791,
            quantile=0.98,  # 0.9946175749502564,
            supervised=False,
            generated_size=150,  # 541
            n_infiltrated_terminals=82,
            beta=0.25391071673841914,
        )

    @staticmethod
    def suggest(trial: Trial):
        logging.info("Suggesting VAE parameters")
        return VAEParameters(
            latent_dim=trial.suggest_int("latent_dim", 2, 92),
            hidden_dim=trial.suggest_int("hidden_dim", 16, 192),
            lr=trial.suggest_float("lr", 1e-5, 1e-3),
            trees=1,  # Not used in VAE because IsolationForest has been removed
            batch_size=trial.suggest_int("batch_size", 8, 64),
            num_epochs=trial.suggest_int("num_epochs", 1000, 10_000),
            quantile=trial.suggest_float("quantile", 0.9, 1.0),
            generated_size=trial.suggest_int("generated_size", 10, 1000),
            beta=trial.suggest_float("beta", 0.0, 1.0),
            n_infiltrated_terminals=trial.suggest_int("n_infiltrated_terminals", 1, 100),
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
    include_weekday: bool
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
        terminal_fract: float = 0.1,
        seed_value: Optional[int] = None,
        card_pool_size: int = CARD_POOL_SIZE,  # TODO It was 50
        avg_card_block_delay_days: int = 7,
        logdir: Optional[str] = None,
        save: bool = True,
        include_weekday: bool = True,
        ulb_data: bool = False,
        aggregation_windows: Sequence[timedelta | float] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
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
        self.include_weekday = include_weekday
        self.aggregation_windows = []
        self.ulb_data = ulb_data
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
            case None:
                raise ValueError("Agent is not set. Please provide an agent.")
            case VAEParameters():
                return self.agent.get_agent(env, device, self.know_client, self.agent.quantile)
            case PPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")

    def create_env(self):
        from banksys import Banksys

        try:
            banksys = Banksys.load(self.banksys_dir)
        except (FileNotFoundError, ValueError):
            print("Banksys not found, creating a new one")
            banksys = self.create_banksys()
            banksys.save(self.banksys_dir)
        env = CardSimEnv(
            system=banksys,
            avg_card_block_delay=timedelta(days=self.avg_card_block_delay_days),
            customer_location_is_known=self.know_client,
            normalize_location=self.agent_name in ("ppo", "rppo"),
            include_weekday=self.include_weekday,
        )
        return env

    def banksys_is_in_cache(self):
        """
        Check if the Banksys directory exists.
        """
        return os.path.exists(self.banksys_dir)

    @property
    def banksys_dir(self):
        if self.ulb_data:
            return os.path.join(
                "ULB",
                "cache",
                "banksys",
                f"{self.cardsim.n_payers}-payers",
                f"{self.cardsim.n_days}-days",
                f"start-{self.cardsim.start_date}",
            )
        else:
            return os.path.join(
                "cache",
                "banksys",
                f"{self.cardsim.n_payers}-payers",
                f"{self.cardsim.n_days}-days",
                f"start-{self.cardsim.start_date}",
            )

    def create_banksys(self, use_cache: bool = True, silent: bool = False, fit: bool = True):
        from banksys import Banksys

        if self.ulb_data:
            transactions = pl.read_csv("MLG_Simulator/transactions.csv")
            cards = pl.read_csv("MLG_Simulator/customer_profiles.csv")
            terminals = pl.read_csv("MLG_Simulator/terminal_profiles.csv")
        else:
            transactions, cards, terminals = self.cardsim.get_simulation_data(use_cache, self.ulb_data)
        return Banksys(
            transactions,
            cards,
            terminals,
            aggregation_windows=self.aggregation_windows,
            attackable_terminal_factor=self.terminal_fract,
            clf_params=self.clf_params,
            fp_rate=0.01,
            fn_rate=0.01,
            silent=silent,
            fit=fit,
        )

    def datasets_exists(self, directory: Optional[str] = None) -> bool:
        """
        Check if the training and test datasets exist in the specified directory.
        If no directory is specified, use the default banksys directory.
        """
        if directory is None:
            directory = self.banksys_dir
        train_path = os.path.join(directory, "train.csv")
        if not os.path.exists(train_path):
            return False
        test_path = os.path.join(directory, "test.csv")
        return os.path.exists(test_path)

    def load_datasets(self, directory: Optional[str] = None):
        """
        Load the training and test datasets from the specified directory.
        If no directory is specified, use the default banksys directory.
        """
        if directory is None:
            directory = self.banksys_dir
        train_x = pd.read_csv(os.path.join(directory, "train.csv"))
        train_y = train_x.pop("label").to_numpy(dtype=np.bool)
        test_x = pd.read_csv(os.path.join(directory, "test.csv"))
        test_y = test_x.pop("label").to_numpy(dtype=np.bool)
        return train_x, train_y, test_x, test_y

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
