import logging
import os
import random
from dataclasses import dataclass, field
from typing import final

import numpy as np
import orjson
import torch

from agents import Agent
from environment import CardSimEnv

from .cardsim_parameters import CardSimParameters
from .classification_parameters import ClassificationParameters
from .env_parameters import EnvParameters
from .agent_parameters import RandomParameters, PPOParameters, VAEParameters


@final
@dataclass(eq=True, frozen=True)
class Parameters:
    agent: PPOParameters | VAEParameters | RandomParameters = field(default_factory=RandomParameters)
    cardsim: CardSimParameters = field(default_factory=CardSimParameters)
    clf_params: ClassificationParameters = field(default_factory=ClassificationParameters)
    env_params: EnvParameters = field(default_factory=EnvParameters)
    seed: int = 0
    cache_root: str = "cache"

    def make_agent(self, env: CardSimEnv, device: torch.device) -> Agent:
        self.seed_random()
        match self.agent:
            case RandomParameters():
                from agents import RandomAgent

                return RandomAgent(env.action_space)
            case VAEParameters():
                return self.agent.get_agent(env, device, self.env_params.know_client, self.agent.quantile)
            case PPOParameters():
                return self.agent.get_agent(env, device)
        raise ValueError(f"Unknown agent type: {self.agent}")

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            data = orjson.loads(f.read())
        assert isinstance(data, dict), "Parameters should be a dictionary"
        match data["agent"]["name"].lower():
            case "ppo" | "rppo":
                agent = PPOParameters.from_json(data.pop("agent"))
            case "vae":
                agent = VAEParameters(**data.pop("agent"))
            case "random":
                agent = RandomParameters(**data.pop("agent"))
            case _:
                raise ValueError(f"Unknown agent type: {data['agent_name']}")
        cardsim = CardSimParameters(**data.pop("cardsim"))

        # --- Handle clf_params safely ---
        clf_params_dict = data.pop("clf_params", {}).copy()

        # Rename old keys to new internal names
        if "training_duration" in clf_params_dict:
            clf_params_dict["_training_duration"] = clf_params_dict.pop("training_duration")

        if "aggregation_windows" in clf_params_dict:
            clf_params_dict["_aggregation_windows"] = clf_params_dict.pop("aggregation_windows")

        # --- Handle env_params safely ---
        env_params_dict = data.pop("env_params", {}).copy()

        # Rename avg_block_delay → _avg_block_delay
        if "avg_block_delay" in env_params_dict:
            env_params_dict["_avg_block_delay"] = env_params_dict.pop("avg_block_delay")

        # Remove legacy aggregation_windows if present
        env_params_dict.pop("aggregation_windows", None)

        # --- Instantiate parameter objects ---
        clf_params = ClassificationParameters(**clf_params_dict)
        env_params = EnvParameters(**env_params_dict)

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
                cache_root=self.cache_root,
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
        return os.path.join(self.cache_root, f"seed-{self.seed}")

    @property
    def dataset_dir(self):
        return self.cardsim.cache_dir(self.cache_dir)

    def load_banksys(self):
        logging.info(f"Loading banksys from {self.banksys_file}")
        file = self.banksys_file
        self.seed_random()
        if os.path.exists(self.banksys_file):
            from banksys import Banksys

            assert file == self.banksys_file
            return Banksys.load(self.banksys_file)
        logging.info("Banksys does not exist, creating one")
        transactions, payers, terminals = self.cardsim.load_simulation_data(self.cache_dir)
        assert file == self.banksys_file
        banksys = self.clf_params.make_banksys(transactions, payers, terminals)
        assert file == self.banksys_file
        logging.info(f"Caching banksys to {self.banksys_file}")
        assert file == self.banksys_file
        banksys.save(self.banksys_file)
        assert file == self.banksys_file
        return banksys
