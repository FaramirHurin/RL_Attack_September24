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
        match (anomaly, modification):
            case (False, False):
                # Optuna trial 82
                # [latent_dim: 92, hidden_dim: 49, lr: 9.959838177633226e-05, batch_size: 10, num_epochs: 1800, quantile: 0.9131073566439921, generated_size: 44, beta: 0.8432341654489954, n_infiltrated_terminals: 45]
                return VAEParameters(
                    latent_dim=92,
                    hidden_dim=49,
                    lr=9.959838177633226e-05,
                    batch_size=10,
                    num_epochs=1800,
                    quantile=0.9131073566439921,
                    supervised=False,
                    generated_size=44,
                    n_infiltrated_terminals=45,
                    beta=0.8432341654489954,
                )
            case (False, True):
                # Optuna trial 34
                # [latent_dim: 11, hidden_dim: 121, lr: 1.1712864680260006e-05, batch_size: 9, num_epochs: 6000, quantile: 0.9953732623329556, generated_size: 91, beta: 0.6087043524048532, n_infiltrated_terminals: 68]
                return VAEParameters(
                    latent_dim=11,
                    hidden_dim=121,
                    lr=1.1712864680260006e-05,
                    batch_size=9,
                    num_epochs=6000,
                    quantile=0.9953732623329556,
                    supervised=False,
                    generated_size=91,
                    n_infiltrated_terminals=68,
                    beta=0.6087043524048532,
                )
            case (True, False):
                # Optuna trial 23
                #  [latent_dim: 66, hidden_dim: 17, lr: 3.2264426994127435e-05, batch_size: 18, num_epochs: 6600, quantile: 0.5985052875072323, generated_size: 29, beta: 0.42151271274078445, n_infiltrated_terminals: 100]
                return VAEParameters(
                    latent_dim=66,
                    hidden_dim=17,
                    lr=3.2264426994127435e-05,
                    batch_size=18,
                    num_epochs=6600,
                    quantile=0.5985052875072323,
                    supervised=False,
                    generated_size=29,
                    n_infiltrated_terminals=100,
                    beta=0.42151271274078445,
                )
            case (True, True):
                # Optuna trial 98
                # [latent_dim: 55, hidden_dim: 103, lr: 1.1699546326454397e-05, batch_size: 48, num_epochs: 4000, quantile: 0.9993816480073827, generated_size: 69, beta: 0.5420575862666193, n_infiltrated_terminals: 21]
                return VAEParameters(
                    latent_dim=55,
                    hidden_dim=103,
                    lr=1.1699546326454397e-05,
                    batch_size=48,
                    num_epochs=4000,
                    quantile=0.9993816480073827,
                    supervised=False,
                    generated_size=69,
                    n_infiltrated_terminals=21,
                    beta=0.5420575862666193,
                )

    @staticmethod
    def suggest(trial: Trial):
        return VAEParameters(
            latent_dim=trial.suggest_int("latent_dim", 2, 92),
            hidden_dim=trial.suggest_int("hidden_dim", 16, 192),
            lr=trial.suggest_float("lr", 1e-5, 1e-3),
            trees=-1,
            supervised=False,
            batch_size=trial.suggest_int("batch_size", 8, 64),
            num_epochs=trial.suggest_int("num_epochs", 1000, 10_000, step=200),
            quantile=trial.suggest_float("quantile", 0.0, 1.0),
            generated_size=trial.suggest_int("generated_size", 10, 1000),
            beta=trial.suggest_float("beta", 0.0, 1.0),
            n_infiltrated_terminals=trial.suggest_int("n_infiltrated_terminals", 1, 100),
        )
