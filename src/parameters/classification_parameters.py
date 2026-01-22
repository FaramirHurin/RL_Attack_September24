import hashlib
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal, Sequence

import orjson
import polars as pl
from optuna import Trial

from banksys.terminal import Terminal
from utils import serialize_unknown


@dataclass(eq=True, frozen=True)
class ClassificationParameters:
    use_anomaly: bool = True
    n_trees: int = 100
    balance_factor: float = 0.1
    """Balance factor for the Balanced Random Forest."""
    contamination: float | Literal["auto"] = "auto"
    _training_duration: float | timedelta = timedelta(days=30)
    quantiles: dict[str, tuple[float, float]] = field(default_factory=lambda: {"amount": (0.01, 0.99)})
    _aggregation_windows: Sequence[float | timedelta] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30))
    _rules: dict[float, int] | dict[timedelta, int] = field(
        default_factory=lambda: {timedelta(hours=1): 6, timedelta(days=1): 16, timedelta(weeks=1): 30}
    )
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    classify_simulated_trx: bool = False
    retrain_interval: timedelta | None = None
    """
    Whether to classify simulated transactions (i.e. not the attacker's ones).
    If False, the classifier uses the ground truth labels corrected by the `fp_rate` and `fn_rate` parameters.
    """

    # def __init__(
    #     self,
    #     use_anomaly: bool = True,
    #     n_trees: int = 100,
    #     balance_factor: float = 0.1,
    #     contamination: float | Literal["auto"] = "auto",
    #     training_duration: timedelta | float = timedelta(days=30),
    #     quantiles: dict[str, tuple[float, float]] = {"amount": (0.01, 0.99)},
    #     rules: dict[timedelta, int] = {
    #         timedelta(hours=1).total_seconds(): 6,
    #         timedelta(days=1): 16,
    #         timedelta(weeks=1): 30,
    #     },
    #     aggregation_windows: Sequence[timedelta | float] = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
    #     retrain_interval: timedelta | None = None,
    #     fp_rate: float = 0.0,
    #     fn_rate: float = 0.0,
    #     classify_simulated_trx: bool = False,
    # ):
    #     self.use_anomaly = use_anomaly
    #     self.n_trees = n_trees
    #     self.balance_factor = balance_factor
    #     self.contamination = contamination
    #     if isinstance(training_duration, (float, int)):
    #         training_duration = timedelta(seconds=training_duration)
    #     self.training_duration = training_duration
    #     self.quantiles = quantiles
    #     self._rules = {td.total_seconds(): value for td, value in rules.items()}
    #     self.fp_rate = fp_rate
    #     self.fn_rate = fn_rate
    #     self._aggregation_windows = []
    #     for window in aggregation_windows:
    #         if isinstance(window, (float, int)):
    #             window = timedelta(seconds=window)
    #         self._aggregation_windows.append(window)
    #     self.classify_simulated_trx = classify_simulated_trx
    #     self.retrain_interval = retrain_interval

    @property
    def training_duration(self):
        """
        Returns the training duration as a timedelta.
        """
        if isinstance(self._training_duration, timedelta):
            return self._training_duration
        return timedelta(seconds=self._training_duration)

    @property
    def rules(self):
        """
        Returns the rules as a dictionary with timedelta keys.
        """
        return {timedelta(seconds=key) if isinstance(key, (float, int)) else key: value for key, value in self._rules.items()}

    @property
    def aggregation_windows(self):
        """
        Returns the aggregation windows as a list of timedeltas.
        """
        return [window if isinstance(window, timedelta) else timedelta(seconds=window) for window in self._aggregation_windows]

    @property
    def longest_window(self):
        if len(self.aggregation_windows) == 0:
            raise ValueError("No aggregation windows provided.")
        return max(*self.aggregation_windows) if len(self.aggregation_windows) > 1 else self.aggregation_windows[0]

    @staticmethod
    def paper_params(with_anomaly: bool, with_modification: bool, *, retrain_interval: timedelta | None = None):
        match (with_modification, with_anomaly):
            case (False, False):
                # [max_trx_hour: 7, max_trx_day: 10, max_trx_week: 45, n_trees: 200, balance_factor: 0.05010330218871149, quantiles_amount_high: 0.9999906444487933, quantiles_risk_high: 0.9998050395914746, fp_rate: 0.008757552666375281, fn_rate: 0.01853373993723769]
                return ClassificationParameters(
                    _training_duration=timedelta(days=30),
                    n_trees=200,
                    contamination="auto",
                    balance_factor=0.05010330218871149,
                    quantiles={
                        "amount": (0, 0.9999906444487933),
                        Terminal.colname("risk", timedelta(days=1)): (0, 0.9998050395914746),
                    },
                    use_anomaly=False,
                    _rules={
                        timedelta(hours=1): 7,
                        timedelta(days=1): 10,
                        timedelta(weeks=1): 45,
                    },
                    fp_rate=0.008757552666375281,
                    fn_rate=0.01853373993723769,
                    retrain_interval=retrain_interval,
                )
            case (False, True):
                # [max_trx_hour: 7, max_trx_day: 19, max_trx_week: 35, n_trees: 163, balance_factor: 0.12817163708045945, quantiles_amount_high: 0.9995238540405597, quantiles_risk_high: 0.9990276449433015, fp_rate: 0.019241410257713323, fn_rate: 0.004683490375549398]
                return ClassificationParameters(
                    _training_duration=timedelta(days=30),
                    n_trees=163,
                    contamination="auto",
                    balance_factor=0.12817163708045945,
                    quantiles={
                        "amount": (0, 0.9995238540405597),
                        Terminal.colname("risk", timedelta(days=1)): (0, 0.9990276449433015),
                    },
                    use_anomaly=True,
                    _rules={
                        timedelta(hours=1): 7,
                        timedelta(days=1): 19,
                        timedelta(weeks=1): 35,
                    },
                    fp_rate=0.019241410257713323,
                    fn_rate=0.004683490375549398,
                    retrain_interval=retrain_interval,
                )
            case (True, False):
                #  [max_trx_hour: 5, max_trx_day: 8, max_trx_week: 35, n_trees: 165, balance_factor: 0.05720386427396055, quantiles_amount_high: 0.9998653727810304, quantiles_risk_high: 0.9996872427746293, fp_rate: 0.0023521979812597383, fn_rate: 0.01342192652282362]
                return ClassificationParameters(
                    _training_duration=timedelta(days=30),
                    n_trees=165,
                    contamination="auto",
                    balance_factor=0.05720386427396055,
                    quantiles={
                        "amount": (0, 0.9998653727810304),
                        Terminal.colname("risk", timedelta(days=1)): (0, 0.9996872427746293),
                    },
                    use_anomaly=False,
                    _rules={
                        timedelta(hours=1): 5,
                        timedelta(days=1): 8,
                        timedelta(weeks=1): 35,
                    },
                    fp_rate=0.0023521979812597383,
                    fn_rate=0.01342192652282362,
                    retrain_interval=retrain_interval,
                )
            case (True, True):
                # [max_trx_hour: 5, max_trx_day: 5, max_trx_week: 24, n_trees: 169, balance_factor: 0.14187681622345746, quantiles_amount_high: 0.998725033037342, quantiles_risk_high: 0.9993087542222269, fp_rate: 0.016083989915547495, fn_rate: 0.014686295327843348]
                return ClassificationParameters(
                    _training_duration=timedelta(days=30),
                    n_trees=169,
                    contamination="auto",
                    balance_factor=0.14187681622345746,
                    quantiles={
                        "amount": (0, 0.998725033037342),
                        Terminal.colname("risk", timedelta(days=1)): (0, 0.9993087542222269),
                    },
                    use_anomaly=True,
                    _rules={
                        timedelta(hours=1): 5,
                        timedelta(days=1): 5,
                        timedelta(weeks=1): 24,
                    },
                    fp_rate=0.016083989915547495,
                    fn_rate=0.014686295327843348,
                    retrain_interval=retrain_interval,
                )

    @staticmethod
    def suggest(trial: Trial, use_anomaly: bool):
        max_per_hour = trial.suggest_int("max_trx_hour", 2, 40)
        max_per_day = trial.suggest_int("max_trx_day", max_per_hour, 400)
        max_per_week = trial.suggest_int("max_trx_week", max_per_day, 450)
        return ClassificationParameters(
            _training_duration=timedelta(days=30).total_seconds(),
            n_trees=trial.suggest_int("n_trees", 20, 200),
            contamination="auto",
            balance_factor=trial.suggest_float("balance_factor", 0.0001, 0.2),
            quantiles={
                "amount": (0, trial.suggest_float("quantiles_amount_high", 0.9995, 1.0)),
                Terminal.colname("risk", timedelta(days=1)): (0, trial.suggest_float("quantiles_risk_high", 0.995, 1.0)),
            },
            use_anomaly=use_anomaly,
            _rules={
                timedelta(hours=1): max_per_hour,
                timedelta(days=1): max_per_day,
                timedelta(weeks=1): max_per_week,
            },
            fp_rate=0, #trial.suggest_float("fp_rate", 0.0, 0.02),
            fn_rate=0, #trial.suggest_float("fn_rate", 0.0, 0.02),
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
