from typing import overload
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from ..transaction import Transaction
import logging
from .rule_based import RuleBasedClassifier, rules_dict
from .statistical import StatisticalClassifier
from sklearn.ensemble import IsolationForest

# Import isolation forest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banksys import Banksys


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier
    anomaly_detection_classifier: IsolationForest
    banksys: "Banksys"
    use_anomaly: bool

    def __init__(
        self,
        features_for_quantiles: list[str],
        quantiles: list[float],
        banksys: "Banksys",
        contamination: float,
        balance_factor: float,
        use_anomaly: bool = True,

    ):
        self.ml_classifier = BalancedRandomForestClassifier(n_jobs=1, sampling_strategy=balance_factor)  # type: ignore[assignment]
        self.banksys = banksys
        self.anomaly_detection_classifier = IsolationForest(contamination=contamination)
        self.statistical_classifier = StatisticalClassifier(features_for_quantiles, quantiles)
        self.rule_classifier = RuleBasedClassifier([], self.banksys, {})
        self.use_anomaly = use_anomaly

    def fit(self, transactions: pd.DataFrame, is_fraud: np.ndarray):
        logging.info("Fitting random forest")
        self.ml_classifier.n_jobs = -1  # type: ignore[assignment]
        self.ml_classifier.fit(transactions, is_fraud)
        self.ml_classifier.n_jobs = 1  # type: ignore[assignment]
        logging.info("Fitting anomaly classifier")
        self.anomaly_detection_classifier.fit(transactions)
        logging.info("Fitting statistical classifier")
        self.statistical_classifier.fit(transactions)
        logging.info("Done !")

    def set_rules(self, rules_values: dict):
        rules = [rules_dict[rule] for rule in rules_values.keys()]
        self.rule_classifier = RuleBasedClassifier(rules, self.banksys, rules_values)

    @overload
    def predict(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]: ...
    @overload
    def predict(self, transaction: Transaction, /) -> bool: ...

    def _predict_transaction(self, transaction: Transaction, /):
        if self.rule_classifier.predict_transaction(transaction):
            return True
        if self.statistical_classifier.predict_transaction(transaction):
            return True
        df = self.banksys.make_features_df(transaction, False)
        is_fraud = bool(self.ml_classifier.predict(df).item())
        if is_fraud:
            return True
        if self.use_anomaly:
            res = self.anomaly_detection_classifier.predict(df).item()
            is_fraud = res == -1
        return is_fraud

    def _predict_dataframe(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]:
        l1 = self.rule_classifier.predict_dataframe(df)
        l2 = self.ml_classifier.predict(df).astype(np.bool)
        l3 = self.statistical_classifier.predict_dataframe(df)
        result = l1 | l2 | l3
        if self.use_anomaly:
            l4 = self.anomaly_detection_classifier.predict(df) == -1
            result = result | l4
        return result

    def predict(self, transaction_or_df, /):
        match transaction_or_df:
            case Transaction() as t:
                return self._predict_transaction(t)
            case pd.DataFrame() as df:
                return self._predict_dataframe(df)
        raise ValueError("Invalid input type. Expected `Transaction` or `DataFrame`.")
