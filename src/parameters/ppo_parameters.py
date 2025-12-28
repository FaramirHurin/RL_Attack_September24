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
    normalize_advantages: bool
    use_covariance_matrix: bool
    name: Literal["ppo", "rppo"]

    def __init__(
        self,
        is_recurrent: bool = False,
        train_on: Literal["transition", "episode"] = "transition",
        gamma: float = 0.99,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 20,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        entropy_c2: Schedule | float = 0.01,
        train_interval: int = 64,
        minibatch_size: int = 32,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
        normalize_advantages: bool = True,
        use_covariance_matrix: bool = True,
    ):
        assert train_interval > 0, "`train_interval` must be positive."
        assert train_interval >= minibatch_size, "`train_interval` must be greater than or equal to `minibatch_size`."
        assert 0.0 < gamma <= 1.0, "`gamma` must be in (0.0, 1.0]."
        assert 0.0 <= gae_lambda <= 1.0, "`gae_lambda ` must be in [0.0, 1.0]."
        assert 0.0 <= eps_clip < 1.0, "`eps_clip` must be in [0.0, 1.0)."
        assert n_epochs > 0, "`n_epochs` must be positive."
        assert lr_actor > 0.0, "`lr_actor` must be positive."
        assert lr_critic > 0.0, "`lr_critic` must be positive."
        assert grad_norm_clipping is None or grad_norm_clipping > 0.0, "`grad_norm_clipping` must be positive or None."
        assert not is_recurrent or train_on == "episode", "Recurrent PPO only supports episode training."
        self.name = "rppo" if is_recurrent else "ppo"
        self.is_recurrent = is_recurrent
        self.train_on = train_on
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self.use_covariance_matrix = use_covariance_matrix
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
            network = RecurrentActorCritic(
                env.observation_size,
                env.action_space,
                device,
                self.use_covariance_matrix,
            )
        else:
            network = LinearActorCritic(
                env.observation_size,
                env.action_space,
                device,
                self.use_covariance_matrix,
            )
        self_dict = self.as_dict()
        self_dict.pop("is_recurrent")
        self_dict.pop("train_on")
        self_dict.pop("use_covariance_matrix")
        self_dict.pop("name")
        return PPO(network, memory, **self_dict, device=device)

    @staticmethod
    def best_rppo(anomaly: bool, modification: bool):
        match (anomaly, modification):
            case (False, False):
                # Optuna trial 121
                # Params = [train_interval: 9, minibatch_size: 5, grad_norm_clipping: 0.018075514993835563, critic_c1_start: 0.9230322797219449, critic_c1_end: 0.3690399676267296, critic_c1_steps: 3741, entropy_c2_start: 0.1837360921799257, entropy_c2_end: 0.09817927437616152, entropy_c2_steps: 701, n_epochs: 20, lr_actor: 0.00019476641687574513, lr_critic: 0.00039743028447575556, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=True,
                    train_on="episode",
                    train_interval=9,
                    minibatch_size=5,
                    grad_norm_clipping=0.018075514993835563,
                    critic_c1=Schedule.linear(
                        start_value=0.9230322797219449,
                        end_value=0.3690399676267296,
                        n_steps=3741,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.1837360921799257,
                        end_value=0.09817927437616152,
                        n_steps=701,
                    ),
                    n_epochs=20,
                    lr_actor=0.00019476641687574513,
                    lr_critic=0.00039743028447575556,
                    normalize_advantages=False,
                )
            case (False, True):
                # Optuna trial 68
                # Params = [train_interval: 17, minibatch_size: 13, grad_norm_clipping: 0.8777594561163811, critic_c1_start: 0.24974225773295028, critic_c1_end: 0.10544686676716412, critic_c1_steps: 3430, entropy_c2_start: 0.0013550962620539116, entropy_c2_end: 0.04674500282092511, entropy_c2_steps: 1356, n_epochs: 13, lr_actor: 0.0006746651394977868, lr_critic: 0.00028738665522909563, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=True,
                    train_on="episode",
                    train_interval=17,
                    minibatch_size=13,
                    grad_norm_clipping=0.8777594561163811,
                    critic_c1=Schedule.linear(
                        start_value=0.24974225773295028,
                        end_value=0.10544686676716412,
                        n_steps=3430,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.0013550962620539116,
                        end_value=0.04674500282092511,
                        n_steps=1356,
                    ),
                    n_epochs=13,
                    lr_actor=0.0006746651394977868,
                    lr_critic=0.00028738665522909563,
                    normalize_advantages=False,
                )
            case (True, False):
                # Optuna trial 54
                # Params = [train_interval: 44, minibatch_size: 33, grad_norm_clipping: 0.048428316027431084, critic_c1_start: 0.7125282508145335, critic_c1_end: 0.16286778125492998, critic_c1_steps: 241, entropy_c2_start: 0.07854662818273801, entropy_c2_end: 0.08216516313921232, entropy_c2_steps: 2302, n_epochs: 5, lr_actor: 0.005856005059230748, lr_critic: 0.0013681256233237888, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=True,
                    train_on="episode",
                    train_interval=44,
                    minibatch_size=33,
                    grad_norm_clipping=0.048428316027431084,
                    critic_c1=Schedule.linear(
                        start_value=0.7125282508145335,
                        end_value=0.16286778125492998,
                        n_steps=241,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.07854662818273801,
                        end_value=0.08216516313921232,
                        n_steps=2302,
                    ),
                    n_epochs=5,
                    lr_actor=0.005856005059230748,
                    lr_critic=0.0013681256233237888,
                    normalize_advantages=False,
                )
            case (True, True):
                # Optuna trial 58
                # Params = [train_interval: 15, minibatch_size: 6, grad_norm_clipping: 4.300306541157133, critic_c1_start: 0.10312961820069343, critic_c1_end: 0.4813358684576971, critic_c1_steps: 701, entropy_c2_start: 0.1708107295846123, entropy_c2_end: 0.029362965791121597, entropy_c2_steps: 1253, n_epochs: 23, lr_actor: 0.00016481198970555, lr_critic: 0.008264238504481972, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=True,
                    train_on="episode",
                    train_interval=15,
                    minibatch_size=6,
                    grad_norm_clipping=4.300306541157133,
                    critic_c1=Schedule.linear(
                        start_value=0.10312961820069343,
                        end_value=0.4813358684576971,
                        n_steps=701,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.1708107295846123,
                        end_value=0.029362965791121597,
                        n_steps=1253,
                    ),
                    n_epochs=23,
                    lr_actor=0.00016481198970555,
                    lr_critic=0.008264238504481972,
                    normalize_advantages=False,
                )

    @staticmethod
    def best_ppo(anomaly: bool, modification: bool):
        """
        The result of the hyperparameter tuning with Optuna for standard PPO (non-recurrent).
        """
        match (anomaly, modification):
            case (False, False):
                # Optuna trial number 76 (cf: tuning/agents-tuning.journal)
                # Params = [train_interval: 57, minibatch_size: 28, grad_norm_clipping: 10.324662966280236, critic_c1_start: 0.32980225996920964, critic_c1_end: 0.0727049296457613, critic_c1_steps: 3535, entropy_c2_start: 0.1564628009423709, entropy_c2_end: 0.07259805698828797, entropy_c2_steps: 1307, n_epochs: 13, lr_actor: 0.00031959245717122, lr_critic: 0.00010032690412000196, normalize_advantages: False]
                return PPOParameters(
                    train_on="transition",
                    is_recurrent=False,
                    train_interval=57,
                    minibatch_size=28,
                    grad_norm_clipping=10.324662966280236,
                    critic_c1=Schedule.linear(
                        start_value=0.32980225996920964,
                        end_value=0.0727049296457613,
                        n_steps=3535,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.1564628009423709,
                        end_value=0.07259805698828797,
                        n_steps=1307,
                    ),
                    n_epochs=13,
                    lr_actor=0.00031959245717122,
                    lr_critic=0.00010032690412000196,
                    normalize_advantages=False,
                )
            case (False, True):
                # Optuna trial number 79 (cf: tuning/agents-tuning.journal)
                # Params = [train_interval: 60, minibatch_size: 33, grad_norm_clipping: 0.011791604576331605, critic_c1_start: 0.8286882375218461, critic_c1_end: 0.4166607963061905, critic_c1_steps: 1521, entropy_c2_start: 0.15453790731608788, entropy_c2_end: 0.05144345719538858, entropy_c2_steps: 2450, n_epochs: 9, lr_actor: 0.0005991165920540275, lr_critic: 0.00015607240842916922, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=False,
                    train_on="transition",
                    train_interval=60,
                    minibatch_size=33,
                    grad_norm_clipping=0.011791604576331605,
                    critic_c1=Schedule.linear(
                        start_value=0.8286882375218461,
                        end_value=0.4166607963061905,
                        n_steps=1521,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.15453790731608788,
                        end_value=0.05144345719538858,
                        n_steps=2450,
                    ),
                    n_epochs=9,
                    lr_actor=0.0005991165920540275,
                    lr_critic=0.00015607240842916922,
                    normalize_advantages=False,
                )
            case (True, False):
                # Optuna trial number 54 (cf: tuning/agents-tuning.journal)
                # Params = [train_interval: 44, minibatch_size: 33, grad_norm_clipping: 0.048428316027431084, critic_c1_start: 0.7125282508145335, critic_c1_end: 0.16286778125492998, critic_c1_steps: 241, entropy_c2_start: 0.07854662818273801, entropy_c2_end: 0.08216516313921232, entropy_c2_steps: 2302, n_epochs: 5, lr_actor: 0.005856005059230748, lr_critic: 0.0013681256233237888, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=False,
                    train_on="transition",
                    train_interval=44,
                    minibatch_size=33,
                    grad_norm_clipping=0.048428316027431084,
                    critic_c1=Schedule.linear(
                        start_value=0.7125282508145335,
                        end_value=0.16286778125492998,
                        n_steps=241,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.07854662818273801,
                        end_value=0.08216516313921232,
                        n_steps=2302,
                    ),
                    n_epochs=5,
                    lr_actor=0.005856005059230748,
                    lr_critic=0.0013681256233237888,
                    normalize_advantages=False,
                )
            case (True, True):
                # Optuna trial number 121 (cf: tuning/agents-tuning.journal)
                # Params = [train_interval: 9, minibatch_size: 5, grad_norm_clipping: 0.018075514993835563, critic_c1_start: 0.9230322797219449, critic_c1_end: 0.3690399676267296, critic_c1_steps: 3741, entropy_c2_start: 0.1837360921799257, entropy_c2_end: 0.09817927437616152, entropy_c2_steps: 701, n_epochs: 20, lr_actor: 0.00019476641687574513, lr_critic: 0.00039743028447575556, normalize_advantages: False]
                return PPOParameters(
                    is_recurrent=False,
                    train_on="transition",
                    train_interval=9,
                    minibatch_size=5,
                    grad_norm_clipping=0.018075514993835563,
                    critic_c1=Schedule.linear(
                        start_value=0.9230322797219449,
                        end_value=0.3690399676267296,
                        n_steps=3741,
                    ),
                    entropy_c2=Schedule.linear(
                        start_value=0.1837360921799257,
                        end_value=0.09817927437616152,
                        n_steps=701,
                    ),
                    n_epochs=20,
                    lr_actor=0.00019476641687574513,
                    lr_critic=0.00039743028447575556,
                    normalize_advantages=False,
                )

    @staticmethod
    def suggest_rppo(trial: Trial):
        train_interval = trial.suggest_int("train_interval", 4, 256)
        minibatch_size = trial.suggest_int("minibatch_size", 2, train_interval)
        grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.01, 50, log=True)
        return PPOParameters(
            is_recurrent=True,
            train_on="episode",
            critic_c1=Schedule.linear(
                trial.suggest_float("critic_c1_start", 0.1, 1.0),
                trial.suggest_float("critic_c1_end", 0.001, 0.5),
                trial.suggest_int("critic_c1_steps", 1, 4000),
            ),
            entropy_c2=Schedule.linear(
                trial.suggest_float("entropy_c2_start", 0.001, 0.2),
                trial.suggest_float("entropy_c2_end", 0.0001, 0.1),
                trial.suggest_int("entropy_c2_steps", 1, 4000),
            ),
            n_epochs=trial.suggest_int("n_epochs", 5, 40),
            minibatch_size=minibatch_size,
            train_interval=train_interval,
            lr_actor=trial.suggest_float("lr_actor", 0.0001, 0.01, log=True),
            lr_critic=trial.suggest_float("lr_critic", 0.0001, 0.01, log=True),
            grad_norm_clipping=grad_norm_clipping,
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
