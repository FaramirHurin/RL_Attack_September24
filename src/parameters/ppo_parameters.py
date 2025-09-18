import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import torch
from marlenv.utils import Schedule
from optuna import Trial

from environment import CardSimEnv


@dataclass(eq=True, unsafe_hash=True)
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
        from agents.rl.networks import LinearActorCritic, RecurrentActorCritic
        from agents.rl.ppo import PPO
        from agents.rl.replay_memory import EpisodeMemory, TransitionMemory

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
        self_dict = self.as_dict()
        self_dict.pop("is_recurrent")
        self_dict.pop("train_on")
        return PPO(network, memory, **self_dict, device=device)

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

    @staticmethod
    def best_ppo(anomaly: bool):
        """
        The result of the hyperparameter tuning with Optuna for standard PPO (non-recurrent).
        """
        if anomaly:
            return PPOParameters(
                is_recurrent=False,
                train_on="transition",
                train_interval=6,
                minibatch_size=4,
                grad_norm_clipping=2.9821909373292796,
                critic_c1=Schedule.linear(
                    start_value=0.6914911855828353,
                    end_value=0.2877063934847368,
                    n_steps=3572,
                ),
                entropy_c2=Schedule.linear(
                    start_value=0.08521542110698155,
                    end_value=0.08272396424417085,
                    n_steps=2311,
                ),
                n_epochs=73,
                lr_actor=0.0005459901195471092,
                lr_critic=0.0004241921268503483,
                normalize_rewards=True,
                normalize_advantages=False,
            )
        return PPOParameters(
            is_recurrent=False,
            train_on="transition",
            train_interval=60,
            minibatch_size=53,
            grad_norm_clipping=None,
            critic_c1=Schedule.linear(
                start_value=0.2474884474147402,
                end_value=0.45684372656802896,
                n_steps=1537,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.1720759514831205,
                end_value=0.020839775092720596,
                n_steps=3622,
            ),
            n_epochs=15,
            lr_actor=0.009869271609124462,
            lr_critic=0.00020047940328712973,
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
            lr_actor=trial.suggest_float("lr_actor", 0.0001, 0.01, log=True),
            lr_critic=trial.suggest_float("lr_critic", 0.0001, 0.01, log=True),
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
