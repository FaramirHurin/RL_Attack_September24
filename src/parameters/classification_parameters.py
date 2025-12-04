from dataclasses import dataclass
from datetime import timedelta
from optuna import Trial
from typing import Literal, Sequence
import polars as pl
import orjson
from banksys.terminal import Terminal
from utils import serialize_unknown
import os
import hashlib


@dataclass(eq=True)
class ClassificationParameters:
    use_anomaly: bool
    n_trees: int
    balance_factor: float
    """Balance factor for the Balanced Random Forest."""
    contamination: float | Literal["auto"]
    training_duration: timedelta
    quantiles: dict[str, tuple[float, float]]
    aggregation_windows: Sequence[timedelta]
    _rules: dict[float, int]
    fp_rate: float
    fn_rate: float
    classify_simulated_trx: bool = False
    """
    Whether to classify simulated transactions (i.e. not the attacker's ones). 
    If False, the classifier uses the ground truth labels corrected by the `fp_rate` and `fn_rate` parameters.
    """

    def __init__(
        self,
        use_anomaly: bool = True,
        n_trees: int = 100,
        balance_factor: float = 0.1,
        contamination: float | Literal["auto"] = "auto",
        training_duration: timedelta | float = timedelta(days=30),
        quantiles: dict[str, tuple[float, float]] = {"amount": (0.01, 0.99)},
        rules: dict[timedelta, int] = {
            timedelta(hours=1): 6,
            timedelta(days=1): 16,
            timedelta(weeks=1): 30,
        },
        aggregation_windows: Sequence[timedelta | float] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        fp_rate: float = 0.0,
        fn_rate: float = 0.0,
    ):
        self.use_anomaly = use_anomaly
        self.n_trees = n_trees
        self.balance_factor = balance_factor
        self.contamination = contamination
        if isinstance(training_duration, (float, int)):
            training_duration = timedelta(seconds=training_duration)
        self.training_duration = training_duration
        self.quantiles = quantiles
        self._rules = {td.total_seconds(): value for td, value in rules.items()}
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.aggregation_windows = []
        for window in aggregation_windows:
            if isinstance(window, (float, int)):
                window = timedelta(seconds=window)
            self.aggregation_windows.append(window)

    @property
    def rules(self) -> dict[timedelta, int]:
        """
        Returns the rules as a dictionary with timedelta keys.
        """
        return {timedelta(seconds=key): value for key, value in self._rules.items()}

    @staticmethod
    def paper_params(with_anomaly: bool, with_modification: bool):
        match (with_modification, with_anomaly):
            case (False, False):
                # [max_trx_hour: 7, max_trx_day: 10, max_trx_week: 45, n_trees: 200, balance_factor: 0.05010330218871149, quantiles_amount_high: 0.9999906444487933, quantiles_risk_high: 0.9998050395914746, fp_rate: 0.008757552666375281, fn_rate: 0.01853373993723769]
                return ClassificationParameters(
                    training_duration=timedelta(days=30),
                    n_trees=200,
                    contamination="auto",
                    balance_factor=0.05010330218871149,
                    quantiles={
                        "amount": (0, 0.9999906444487933),
                        Terminal.colname("risk", timedelta(days=1)): (0, 0.9998050395914746),
                    },
                    use_anomaly=False,
                    rules={
                        timedelta(hours=1): 7,
                        timedelta(days=1): 10,
                        timedelta(weeks=1): 45,
                    },
                    fp_rate=0.008757552666375281,
                    fn_rate=0.01853373993723769,
                )
            case (False, True):
                raise NotImplementedError()
            case (True, False):
                raise NotImplementedError()
            case (True, True):
                raise NotImplementedError()

    @staticmethod
    def suggest(trial: Trial, use_anomaly: bool):
        max_per_hour = trial.suggest_int("max_trx_hour", 2, 10)
        max_per_day = trial.suggest_int("max_trx_day", max_per_hour, 20)
        max_per_week = trial.suggest_int("max_trx_week", max_per_day, 50)
        return ClassificationParameters(
            training_duration=timedelta(days=30),
            n_trees=trial.suggest_int("n_trees", 20, 200),
            contamination="auto",
            balance_factor=trial.suggest_float("balance_factor", 0.05, 0.2),
            quantiles={
                "amount": (0, trial.suggest_float("quantiles_amount_high", 0.995, 1.0)),
                Terminal.colname("risk", timedelta(days=1)): (0, trial.suggest_float("quantiles_risk_high", 0.995, 1.0)),
            },
            use_anomaly=use_anomaly,
            rules={
                timedelta(hours=1): max_per_hour,
                timedelta(days=1): max_per_day,
                timedelta(weeks=1): max_per_week,
            },
            fp_rate=trial.suggest_float("fp_rate", 0.0, 0.02),
            fn_rate=trial.suggest_float("fn_rate", 0.0, 0.02),
        )

    def make_banksys(
        self,
        transactions: pl.DataFrame,
        payers: pl.DataFrame,
        terminals: pl.DataFrame,
        *,
        silent: bool = False,
    ):
        from banksys import Banksys

        return Banksys(
            transactions,
            payers,
            terminals,
            params=self,
            silent=silent,
        )

    def cache_dir(self, dataset_dir: str):
        serialized = orjson.dumps(self, default=serialize_unknown)
        hash_digest = hashlib.sha256(serialized).hexdigest()
        return os.path.join(dataset_dir, f"banksys-{hash_digest}")

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "classification_params.json"), "wb") as f:
            f.write(orjson.dumps(self, default=serialize_unknown, option=orjson.OPT_INDENT_2))
