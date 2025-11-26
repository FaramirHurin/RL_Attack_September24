import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from functools import cached_property
import shutil

import numpy as np
import orjson
import hashlib
import torch

from agents import Agent
from environment import CardSimEnv
from utils import serialize_unknown

from .ppo_parameters import PPOParameters
from .vae_parameters import VAEParameters
from .cardsim_parameters import CardSimParameters
from .classification_parameters import ClassificationParameters
from .env_parameters import EnvParameters


@dataclass(eq=True)
class Parameters:
    agent: PPOParameters | VAEParameters | None
    cardsim: CardSimParameters
    clf_params: ClassificationParameters
    env_params: EnvParameters
    seed: int

    def __init__(
        self,
        agent: PPOParameters | VAEParameters | None = None,
        cardsim: Optional[CardSimParameters] = None,
        clf_params: Optional[ClassificationParameters] = None,
        env_params: Optional[EnvParameters] = None,
        seed: int = 0,
        *,
        regenerate_dataset: bool = False,
        regenerate_banksys: bool = False,
        **kwargs,
    ):
        ######################################
        # Set the seed before ANYTHING else  #
        ######################################
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed

        kwargs.pop("agent_name", None)  # agent_name is set automatically with the "repeat" method
        if len(kwargs) > 0:
            logging.warning(f"Unknown parameters: {kwargs}. They will be ignored.")
        self.agent = agent
        if cardsim is None:
            cardsim = CardSimParameters()
        self.cardsim = cardsim
        if clf_params is None:
            clf_params = ClassificationParameters()
        self.clf_params = clf_params
        if env_params is None:
            env_params = EnvParameters()
        self.env_params = env_params
        if regenerate_dataset:
            shutil.rmtree(self.dataset_dir)
            regenerate_banksys = True
        if regenerate_banksys:
            os.remove(self.banksys_file)

    def make_agent(self, env: CardSimEnv, device: torch.device) -> Agent:
        match self.agent:
            case None:
                raise ValueError("Agent is not set. Please provide an agent.")
            case VAEParameters():
                return self.agent.get_agent(env, device, self.env_params.know_client, self.agent.quantile)
            case PPOParameters():
                return self.agent.get_agent(env, device)
            case _:
                raise ValueError("Unknown agent type")

    @cached_property
    def logdir(self):
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
                agent = PPOParameters.from_json(data["agent"])
            case "vae":
                agent = VAEParameters(**data["agent"])
            case _:
                raise ValueError(f"Unknown agent type: {data['agent_name']}")
        cardsim = CardSimParameters(**data["cardsim"])
        clf_params = ClassificationParameters(**data["clf_params"])
        env_params = EnvParameters(**data["env_params"])
        return Parameters(
            agent=agent,
            cardsim=cardsim,
            clf_params=clf_params,
            env_params=env_params,
            **data,
        )

    @property
    def agent_name(self) -> str:
        if self.agent is None:
            return ""
        return self.agent.name

    def sha256(self) -> str:
        serialized = orjson.dumps(self, default=serialize_unknown)
        return hashlib.sha256(serialized).hexdigest()

    def make_env(self):
        self.prepare_run()
        return CardSimEnv(self.banksys, self.env_params)

    @property
    def banksys(self):
        self.prepare_run()
        from banksys import Banksys

        return Banksys.load(self.banksys_file)

    @cached_property
    def cache_dir(self):
        """
        The unique directory where the data required for the run is cached.

        This data includes the simulation dataset and banksys model trained on this dataset.
        """
        return os.path.join("cache", self.sha256())

    @property
    def banksys_file(self):
        return os.path.join(self.cache_dir, "banksys.pkl")

    @property
    def dataset_dir(self):
        return os.path.join(self.cache_dir, "dataset")

    @property
    def params_file(self):
        return os.path.join(self.cache_dir, "params.json")

    @property
    def is_ready(self) -> bool:
        if not os.path.exists(self.cache_dir):
            return False
        if not os.path.exists(self.banksys_file):
            return False
        if not os.path.exists(self.params_file):
            return False
        return True

    def prepare_run(self):
        """
        Ensures that everything is ready for the run, i.e.:
         1. If there exists a Banksys in the cache, return.
         2. Otherwise:
            a. Generate the dataset
            b. Create and train the Banksys
            c. Save the Banksys to the cache
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        # Save parameters
        if not os.path.exists(self.params_file):
            with open(self.params_file, "wb") as f:
                f.write(orjson.dumps(self, default=serialize_unknown, option=orjson.OPT_INDENT_2))
        # If the banksys file exists, then we are set
        if os.path.exists(self.banksys_file):
            return

        transactions, payers, terminals = self.cardsim.get_simulation_data(self.dataset_dir)
        banksys = self.clf_params.make_banksys(transactions, payers, terminals)
        banksys.save(self.banksys_file)
