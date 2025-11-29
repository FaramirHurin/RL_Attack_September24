import os
import random
import shutil
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import orjson
import torch

from agents import Agent
from environment import CardSimEnv

from .cardsim_parameters import CardSimParameters
from .classification_parameters import ClassificationParameters
from .env_parameters import EnvParameters
from .ppo_parameters import PPOParameters
from .vae_parameters import VAEParameters


@dataclass(eq=True)
class Parameters:
    agent: PPOParameters | VAEParameters | None
    cardsim: CardSimParameters
    clf_params: ClassificationParameters
    env_params: EnvParameters
    seed: int
    cache_root: str

    def __init__(
        self,
        agent: PPOParameters | VAEParameters | None = None,
        cardsim: Optional[CardSimParameters] = None,
        clf_params: Optional[ClassificationParameters] = None,
        env_params: Optional[EnvParameters] = None,
        seed: int = 0,
        cache_root: str | None = None,
        *,
        invalidate_dataset_cache: bool = False,
        invalidate_banksys_cache: bool = False,
    ):
        ######################################
        # Set the seed before ANYTHING else  #
        ######################################
        self.seed = seed
        self.seed_random()

        self.agent = agent
        if cache_root is None:
            cache_root = os.path.join("cache", f"seed-{seed}")
        self.cache_root = cache_root
        if cardsim is None:
            cardsim = CardSimParameters()
        self.cardsim = cardsim
        if clf_params is None:
            clf_params = ClassificationParameters()
        self.clf_params = clf_params
        if env_params is None:
            env_params = EnvParameters()
        self.env_params = env_params
        if invalidate_dataset_cache:
            shutil.rmtree(self.dataset_dir)
            invalidate_banksys_cache = True
        if invalidate_banksys_cache:
            try:
                os.remove(self.banksys_file)
            except OSError:
                pass
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root, exist_ok=True)
        self.cardsim.save(self.cache_root)
        self.clf_params.save(self.dataset_dir)

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

    def seed_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def agent_name(self) -> str:
        if self.agent is None:
            return ""
        return self.agent.name

    def make_env(self):
        banksys = self.load_banksys()
        return CardSimEnv(banksys, self.env_params)

    @property
    def banksys_file(self):
        return self.clf_params.banksys_file(self.dataset_dir)

    @cached_property
    def dataset_dir(self):
        return self.cardsim.cache_dir(self.cache_root)

    def load_banksys(self):
        transactions, payers, terminals = self.cardsim.get_simulation_data(self.cache_root)
        if os.path.exists(self.banksys_file):
            from banksys import Banksys

            return Banksys.load(self.banksys_file)
        banksys = self.clf_params.make_banksys(transactions, payers, terminals)
        banksys.save(self.banksys_file)
        return banksys
