from dataclasses import dataclass
from datetime import timedelta
from optuna import Trial
from typing import Literal
import hashlib


@dataclass(eq=True)
class ClassificationParameters:
    use_anomaly: bool
    n_trees: int
    balance_factor: float
    contamination: float | Literal["auto"]
    training_duration: timedelta
    quantiles: dict[str, tuple[float, float]]
    rules: dict[timedelta, float]
    risk_aggregation_window: timedelta = timedelta(days=1)

    def __init__(
        self,
        use_anomaly: bool = True,
        n_trees: int = 50,
        balance_factor: float = 0.1,
        contamination: float | Literal["auto"] = "auto",
        training_duration: timedelta | float = timedelta(days=30),
        quantiles: dict[str, tuple[float, float]] = {"amount": (0.01, 0.99)},
        rules: dict[float | timedelta, float] = {
            timedelta(hours=1): 6,
            timedelta(days=1): 16,
            timedelta(weeks=1): 30,
        },
    ):
        self.use_anomaly = use_anomaly
        self.n_trees = n_trees
        self.balance_factor = balance_factor
        self.contamination = contamination
        if isinstance(training_duration, (float, int)):
            training_duration = timedelta(seconds=training_duration)
        self.training_duration = training_duration
        self.quantiles = quantiles
        self.rules = {}
        for key, value in rules.items():
            if isinstance(key, (float, int)):
                key = timedelta(seconds=key)
            self.rules[key] = value

    @staticmethod
    def paper_params(anomaly: bool):
        if anomaly:
            return ClassificationParameters(
                use_anomaly=True,
                n_trees=98,
                balance_factor=0.06268092204600313,
                contamination="auto",
                training_duration=timedelta(days=150),
                quantiles={
                    "amount": (0.0, 0.9999170024954384),
                    "terminal_risk_last_1 day, 0:00:00": (0.0, 0.9999132292246781),
                },
                rules={
                    timedelta(hours=1): 5,
                    timedelta(days=1): 19,
                    timedelta(weeks=1): 27,
                },
            )
        return ClassificationParameters(
            use_anomaly=False,
            n_trees=127,
            balance_factor=0.05594667336369366,
            contamination=0.005,
            training_duration=timedelta(days=150),
            quantiles={
                "amount": (0.0, 0.9999924062983265),
                "terminal_risk_last_1 day, 0:00:00": (0.0, 0.9999996860191219),
            },
            rules={
                timedelta(hours=1): 7,
                timedelta(days=1): 8,
                timedelta(weeks=1): 37,
            },
        )

    @staticmethod
    def suggest(trial: Trial, training_duration: timedelta, use_anomaly: bool):
        max_per_hour = trial.suggest_int("max_trx_hour", 2, 10)
        max_per_day = trial.suggest_int("max_trx_day", max_per_hour, 20)
        max_per_week = trial.suggest_int("max_trx_week", max_per_day, 50)
        return ClassificationParameters(
            training_duration=training_duration,
            n_trees=trial.suggest_int("n_trees", 20, 200),
            contamination="auto",
            balance_factor=trial.suggest_float("balance_factor", 0.05, 0.2),
            quantiles={
                "amount": (0, trial.suggest_float("quantiles_amount_high", 0.995, 1.0)),
                f"terminal_risk_last_{timedelta(days=1)}": (0, trial.suggest_float("quantiles_risk_high", 0.995, 1.0)),
            },
            use_anomaly=use_anomaly,
            rules={
                timedelta(hours=1): max_per_hour,
                timedelta(days=1): max_per_day,
                timedelta(weeks=1): max_per_week,
            },
        )

    def __hash__(self):
        to_hash = [self.use_anomaly, self.n_trees, self.balance_factor, self.contamination, self.training_duration]
        to_hash.extend(sorted(self.quantiles.items()))
        to_hash.extend(sorted(self.rules.items()))
        return int(hashlib.sha256(str(tuple(to_hash)).encode("utf-8")).hexdigest(), 16)

    @property
    def max_aggregation_duration(self) -> timedelta:
        return max(self.rules.keys()) if len(self.rules) > 0 else timedelta(0)
