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
        return PPO(network, memory, **self.as_dict(), device=device)

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

        """
        - train_interval: 6
        - minibatch_size: 5
        - enable_clipping: True
        - grad_norm_clipping: 2.388555590580865
        - critic_c1_start: 0.4105552831006898
        - critic_c1_end: 0.3271347177719041
        - critic_c1_steps: 2728
        - entropy_c2_start: 0.19110949972090585
        - entropy_c2_end: 0.030016369088242106
        - entropy_c2_steps: 1699
        - n_epochs: 52
        - lr_actor: 0.007751751648130268
        - lr_critic: 0.003790033882253389
        - normalize_rewards: False
        - normalize_advantages: False
        """

    """            =Schedule.linear(
            start_value=0.19110949972090585,
            end_value=0.030016369088242106,
            n_steps=1699,
        ),"""

    @staticmethod
    def best_ppo():
        """
        The result of the hyperparameter tuning with Optuna for standard PPO (non-recurrent).
        """
        return PPOParameters(
            is_recurrent=False,
            train_on="transition",
            train_interval=50,  # 63
            minibatch_size=40,  # 47
            grad_norm_clipping=None,
            critic_c1=Schedule.linear(
                start_value=0.9210682011725766,
                end_value=0.277828843096964265,
                n_steps=2980,
            ),
            entropy_c2=Schedule.linear(
                start_value=0.25,  # 0.15586561621061853,
                end_value=0.05,  # 0.08458724795592026,
                n_steps=2012,
            ),
            n_epochs=15,
            lr_actor=0.000956264649262804,
            lr_critic=0.006671638920039944,
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
            lr_actor=trial.suggest_float("lr_actor", 0.0001, 0.01),
            lr_critic=trial.suggest_float("lr_critic", 0.0001, 0.01),
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
