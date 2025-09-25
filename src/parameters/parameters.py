import logging
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
import orjson
import hashlib
import pandas as pd
import torch

from agents import Agent
from environment import CardSimEnv

from .ppo_parameters import PPOParameters
from .vae_parameters import VAEParameters
from .cardsim_parameters import CardSimParameters
from .classification_parameters import ClassificationParameters


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
        card_pool_size: int = 50,
        avg_card_block_delay_days: int = 7,
        logdir: Optional[str] = None,
        save: bool = True,
        include_weekday: bool = True,
        ulb_data: bool = False,
        **kwargs,
    ):
        kwargs.pop("agent_name", None)  # agent_name is set automatically with the "repeat" method
        if len(kwargs) > 0:
            logging.warning(f"Ignored unknown parameters: {kwargs}.")
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
        self.clf_params.rules.keys()
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
            logging.info("Banksys not found, creating a new one")
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
        hhash = hashlib.sha256(str((self.clf_params, self.cardsim)).encode("utf-8")).hexdigest()
        return os.path.join("cache", hhash)

    def create_banksys(self, use_cache: bool = True, fit: bool = True, fp_rate: float = 0.0, fn_rate: float = 0.0):
        from banksys import Banksys

        transactions, cards, terminals = self.cardsim.get_simulation_data(use_cache, self.ulb_data)
        return Banksys(
            transactions,
            cards,
            terminals,
            attackable_terminal_factor=self.terminal_fract,
            clf_params=self.clf_params,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
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
        if self.clf_params.use_anomaly:
            anomaly = "anomaly"
        else:
            anomaly = "no-anomaly"
        return os.path.join("logs", anomaly, self.agent_name, timestamp)

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


def serialize_unknown(data):
    match data:
        case timedelta():
            return data.total_seconds()
    raise NotImplementedError(f"Unsupported serialization for type: {type(data)}. Value={data}")
