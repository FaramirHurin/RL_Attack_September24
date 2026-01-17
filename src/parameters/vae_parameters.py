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
                # Optuna trial 82, value=869690.81 (file tuning/vae_tuning.journal)
                # [latent_dim: 92, hidden_dim: 49, lr: 9.959838177633226e-05, batch_size: 10, num_epochs: 1800, quantile: 0.9131073566439921, generated_size: 44, beta: 0.8432341654489954, n_infiltrated_terminals: 45]
                # Optuna trial 80, value=873901.74  (file tuning/agents_tuning.journal)
                # [latent_dim: 90, hidden_dim: 76, lr: 1.4983950667384355e-05, batch_size: 8, num_epochs: 6288, quantile: 0.9933303450751483, generated_size: 525, beta: 0.5686349293691934, n_infiltrated_terminals: 96]
                return VAEParameters(
                    latent_dim=90,
                    hidden_dim=76,
                    lr=1.4983950667384355e-05,
                    batch_size=8,
                    num_epochs=6288,
                    quantile=0.9933303450751483,
                    supervised=False,
                    generated_size=525,
                    n_infiltrated_terminals=96,
                    beta=0.5686349293691934,
                )
            case (False, True):
                # Trial 17, value=685366.01 (file tuning/agents_tuning.journal)
                # Params = [latent_dim: 36, hidden_dim: 143, lr: 0.0002833361052823374, batch_size: 39, num_epochs: 7779, quantile: 0.9981315336722992, generated_size: 993, beta: 0.5362266193781453, n_infiltrated_terminals: 46]

                # Optuna trial 34, value=843656.42 (file tuning/vae_tuning.journal)
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
                # Optuna trial 82 with value 393930
                # Params = [latent_dim: 89, hidden_dim: 131, lr: 0.0009388578140355416, batch_size: 17, num_epochs: 4800, quantile: 0.9937020257932133, generated_size: 115, beta: 0.004630972484961304, n_infiltrated_terminals: 79]
                return VAEParameters(
                    latent_dim=89,
                    hidden_dim=131,
                    lr=0.0009388578140355416,
                    batch_size=17,
                    num_epochs=4800,
                    quantile=0.9937020257932133,
                    supervised=False,
                    generated_size=115,
                    n_infiltrated_terminals=79,
                    beta=0.004630972484961304,
                )
            case (True, True):
                # Trial 86, value=2078.11 in (file tuning/agents_tuning.journal)
                # [latent_dim: 57, hidden_dim: 160, lr: 1.0399119421042868e-05, batch_size: 13, num_epochs: 2901, quantile: 0.9796006640858727, generated_size: 466, beta: 0.2979145387502194, n_infiltrated_terminals: 1]
                # Trial 98, value=90087.71 (file tuning/vae_tuning.journal)
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
            batch_size=trial.suggest_int("batch_size", 8, 256),
            num_epochs=trial.suggest_int("num_epochs", 1000, 20_000, step=200),
            quantile=trial.suggest_float("quantile", 0.0, 1.0),
            generated_size=trial.suggest_int("generated_size", 10, 1000),
            beta=trial.suggest_float("beta", 0.0, 1.0),
            n_infiltrated_terminals=trial.suggest_int("n_infiltrated_terminals", 1, 100),
        )
