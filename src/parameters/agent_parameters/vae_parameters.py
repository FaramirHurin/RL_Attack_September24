import random
from dataclasses import dataclass

import torch
from optuna import Trial
from environment import CardSimEnv
from .agent_parameters import AgentParameters


@dataclass(eq=True, unsafe_hash=True)
class VAEParameters(AgentParameters):
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
    name: str = "vae"

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
    def best_vae(anomaly: bool, modification: bool):
        from agents_tuning import load_study

        try:
            study = load_study("vae", modification, anomaly, only_load=True)
            return VAEParameters(**study.best_params, supervised=False)
        except KeyError:
            raise NotImplementedError("No best VAE parameters found for the given configuration.")

    @staticmethod
    def suggest(trial: Trial):
        return VAEParameters(
            latent_dim=trial.suggest_int("latent_dim", 2, 92),
            hidden_dim=trial.suggest_int("hidden_dim", 16, 192),
            lr=trial.suggest_float("lr", 1e-5, 1e-3),
            trees=-1,
            supervised=False,
            batch_size=trial.suggest_int("batch_size", 8, 256),
            num_epochs=trial.suggest_int("num_epochs", 1000, 20_000, step=200),
            quantile=trial.suggest_float("quantile", 0.0, 1.0),
            generated_size=trial.suggest_int("generated_size", 10, 1000),
            beta=trial.suggest_float("beta", 0.0, 1.0),
            n_infiltrated_terminals=trial.suggest_int("n_infiltrated_terminals", 1, 100),
        )
