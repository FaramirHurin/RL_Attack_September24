import logging
from datetime import timedelta
from typing import TYPE_CHECKING, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ..transaction import Transaction
from .rule_based import RuleBasedClassifier, rules_dict
from .statistical import StatisticalClassifier

if TYPE_CHECKING:
    from banksys import Banksys
    from parameters import ClassificationParameters


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier
    anomaly_detection_classifier: IsolationForest
    banksys: "Banksys"
    use_anomaly: bool
    training_duration: timedelta

    def __init__(
        self,
        banksys: "Banksys",
        params: "ClassificationParameters",
    ):
        self.ml_classifier = BalancedRandomForestClassifier(n_estimators=params.n_trees, n_jobs=-1, sampling_strategy=params.balance_factor)  # type: ignore[assignment]
        self.banksys = banksys
        self.anomaly_detection_classifier = IsolationForest(n_estimators=params.n_trees, n_jobs=-1, contamination=params.contamination)
        self.statistical_classifier = StatisticalClassifier(params.quantiles_features, params.quantiles_values)
        self.rule_classifier = RuleBasedClassifier([], self.banksys, {})
        self.use_anomaly = params.use_anomaly
        self.training_duration = params.training_duration

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
        logging.debug("Predicting with rule-based")
        l1 = self.rule_classifier.predict_dataframe(df)
        logging.debug("Predicting with RF")
        l2 = self.ml_classifier.predict(df).astype(np.bool)
        logging.debug("Predicting with statistical classifier")
        l3 = self.statistical_classifier.predict_dataframe(df)
        result = l1 | l2 | l3
        if self.use_anomaly:
            logging.debug("Predicting with anomaly detection")
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
