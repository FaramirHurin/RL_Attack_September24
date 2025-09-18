import logging
import random
from dataclasses import dataclass

import torch
from optuna import Trial

from environment import CardSimEnv


@dataclass(eq=True, unsafe_hash=True)
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
    def best_vae(anomaly: bool):
        if anomaly:
            return VAEParameters(
                latent_dim=70,
                hidden_dim=58,
                lr=0.00010348989480030503,
                trees=1,  # Not used in VAE because IsolationForest has been removed
                batch_size=10,
                num_epochs=4102,
                quantile=0.9968454105129477,
                supervised=False,
                generated_size=19,
                n_infiltrated_terminals=24,
                beta=0.7395612377633194,
            )
        return VAEParameters(
            latent_dim=86,
            hidden_dim=26,
            lr=3.333660794659185e-05,
            trees=1,  # Not used in VAE because IsolationForest has been removed
            batch_size=12,
            num_epochs=8038,
            quantile=0.9936280503332743,
            supervised=False,
            generated_size=170,
            n_infiltrated_terminals=12,
            beta=0.9969498006633586,
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
