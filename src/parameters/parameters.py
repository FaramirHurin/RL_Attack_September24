import logging
import os
import random
import shutil
from dataclasses import dataclass
from typing import final

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


@final
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
        cardsim: CardSimParameters | None = None,
        clf_params: ClassificationParameters | None = None,
        env_params: EnvParameters | None = None,
        seed: int = 0,
        cache_root: str = "cache",
        *,
        invalidate_dataset_cache: bool = False,
        invalidate_banksys_cache: bool = False,
    ):
        self.seed = seed
        self.agent = agent
        self._cache_root = cache_root
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
                self.save()

    def make_agent(self, env: CardSimEnv, device: torch.device) -> Agent:
        self.seed_random()
        match self.agent:
            case None:
                raise ValueError("Agent is not set. Please provide an agent.")
            case VAEParameters():
                return self.agent.get_agent(env, device, self.env_params.know_client, self.agent.quantile)
            case PPOParameters():
                return self.agent.get_agent(env, device)
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

    def repeat(self, n: int):
        for i in range(n):
            yield Parameters(
                agent=self.agent,
                cardsim=self.cardsim,
                clf_params=self.clf_params,
                env_params=self.env_params,
                seed=self.seed + i,
                cache_root=self._cache_root,
            )

    def save(self):
        self.cardsim.save(self.dataset_dir)
        self.clf_params.save(self.banksys_dir)

    @property
    def agent_name(self) -> str:
        if self.agent is None:
            return ""
        return self.agent.name

    def make_env(self):
        self.seed_random()
        banksys = self.load_banksys()
        self.save()
        return CardSimEnv(banksys, self.env_params)

    @property
    def banksys_file(self):
        return os.path.join(self.banksys_dir, "banksys.pkl")

    @property
    def banksys_dir(self):
        return self.clf_params.cache_dir(self.dataset_dir)

    @property
    def cache_dir(self):
        """Cache directory taking the seed into account."""
        return os.path.join(self._cache_root, f"seed-{self.seed}")

    @property
    def dataset_dir(self):
        return self.cardsim.cache_dir(self.cache_dir)

    def load_banksys(self):
        logging.info(f"Loading banksys from {self.banksys_file}")
        self.seed_random()
        if os.path.exists(self.banksys_file):
            from banksys import Banksys

            return Banksys.load(self.banksys_file)
        logging.info("Banksys does not exist, creating one")
        transactions, payers, terminals = self.cardsim.load_simulation_data(self.cache_dir)
        banksys = self.clf_params.make_banksys(transactions, payers, terminals)
        logging.info(f"Caching banksys to {self.banksys_file}")
        banksys.save(self.banksys_file)
        return banksys
