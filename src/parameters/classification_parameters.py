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
    iforest_n_estimators: int = 100
    iforest_max_features: float = 1.0
    iforest_bootstrap: bool = False
    retrain_interval: timedelta | None = None
    """
    Whether to classify simulated transactions (i.e. not the attacker's ones).
    If False, the classifier uses the ground truth labels corrected by the `fp_rate` and `fn_rate` parameters.
    """

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
    def suggest(trial: Trial, use_anomaly: bool, use_rules: bool, use_statistical: bool):
        if not use_rules:
            max_per_hour = 100
            max_per_day = 500
            max_per_week = 1000
        else:
            max_per_hour = trial.suggest_int("max_trx_hour", 2, 40)
            max_per_day = trial.suggest_int("max_trx_day", max_per_hour, 400)
            max_per_week = trial.suggest_int("max_trx_week", max_per_day, 450)
        if not use_statistical:
            quantiles_amount_high = 1.0
            quantiles_risk_high = 1.0
        else:
            quantiles_amount_high = trial.suggest_float("quantiles_amount_high", 0.9995, 1.0)
            quantiles_risk_high = trial.suggest_float("quantiles_risk_high", 0.995, 1.0)
        if use_anomaly:
            if not trial.suggest_categorical("auto_contamination", [True, False]):
                contamination = trial.suggest_float("contamination", 0.001, 0.1, log=True)
            else:
                contamination = "auto"
            n_estimators = trial.suggest_int("iforest_n_estimators", 50, 200)
            max_features = trial.suggest_float("iforest_max_features", 0.5, 1.0)
            bootstrap = trial.suggest_categorical("iforest_bootstrap", [True, False])
        else:
            contamination = "auto"
            n_estimators = 100
            max_features = 1.0
            bootstrap = False

        return ClassificationParameters(
            _training_duration=timedelta(days=30).total_seconds(),
            n_trees=trial.suggest_int("n_trees", 20, 200),
            contamination=contamination,
            balance_factor=trial.suggest_float("balance_factor", 0.02, 0.2, log=True),
            quantiles={
                "amount": (0, quantiles_amount_high),
                Terminal.colname("risk", timedelta(days=1)): (0, quantiles_risk_high),
            },
            use_anomaly=use_anomaly,
            iforest_n_estimators=n_estimators,
            iforest_max_features=max_features,
            iforest_bootstrap=bootstrap,
            _rules={
                timedelta(hours=1): max_per_hour,
                timedelta(days=1): max_per_day,
                timedelta(weeks=1): max_per_week,
            },
            fp_rate=0,  # trial.suggest_float("fp_rate", 0.0, 0.02),
            fn_rate=0,  # trial.suggest_float("fn_rate", 0.0, 0.02),
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
