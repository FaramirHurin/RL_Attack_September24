import logging
from datetime import timedelta
from typing import TYPE_CHECKING
import numpy.typing as npt
import pandas as pd
import polars as pl
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .rule_based import RuleBasedClassifier
from .statistical import StatisticalClassifier

if TYPE_CHECKING:
    from parameters import ClassificationParameters


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier
    anomaly_detection_classifier: IsolationForest
    use_anomaly: bool
    training_duration: timedelta
    dataset: dict
    retrain_every: timedelta

    def __init__(self, params: "ClassificationParameters"):
        self.ml_classifier = BalancedRandomForestClassifier(n_estimators=params.n_trees, n_jobs=-1, sampling_strategy=params.balance_factor)  # type: ignore[assignment]
        self.anomaly_detection_classifier = IsolationForest(n_jobs=-1, contamination=0.005)  # , contamination="auto"
        self.statistical_classifier = StatisticalClassifier(params.quantiles)
        self.rule_classifier = RuleBasedClassifier(params.rules)
        self.use_anomaly = params.use_anomaly
        self.training_duration = params.training_duration
        self.l1 = np.array([], dtype=np.bool)  # Placeholder for the first prediction, to be replaced in predict method
        self.l2 = np.array([], dtype=np.bool)  # Placeholder for the second prediction, to be replaced in predict method
        self.l3 = np.array([], dtype=np.bool)  # Placeholder for the third prediction, to be replaced in predict method
        self.l4 = np.array([], dtype=np.bool)  # Placeholder for the anomaly detection prediction, to be replaced in predict method
        # assert not params.use_anomaly, "Anomaly detection is not supported in this version of the classification system."
        self.dataset = {}

    def fit(self, transactions: pl.DataFrame, is_fraud: np.ndarray):
        logging.info("Fitting random forest")
        self.ml_classifier.n_jobs = -1  # type: ignore[assignment]
        self.ml_classifier.fit(transactions, is_fraud)
        self.ml_classifier.n_jobs = 1  # type: ignore[assignment]
        logging.info("Fitting anomaly classifier")
        self.anomaly_detection_classifier.fit(transactions)
        logging.info("Fitting statistical classifier")
        self.statistical_classifier.fit(transactions)
        logging.info("Done !")

        self.add_transactions(transactions, is_fraud)

    def predict(self, df: pl.DataFrame, true_labels, t) -> npt.NDArray[np.bool]:
        logging.debug("Predicting with RF")
        l1 = self.ml_classifier.predict(df).astype(np.bool)
        logging.debug("Predicting with statistical classifier")
        l2 = self.statistical_classifier.predict(df)
        logging.debug("Predicting with rule-based")
        l3 = self.rule_classifier.predict(df)
        result = l1 | l2 | l3
        if self.use_anomaly:
            logging.debug("Predicting with anomaly detection")
            label = self.anomaly_detection_classifier.predict(df)
            l4 = label == -1
            result = result | l4

        self.l1, self.l2, self.l3 = l1, l2, l3
        if self.use_anomaly:
            self.l4 = l4
        else:
            l4 = np.zeros_like(l1, dtype=np.bool)
            self.l4 = l4
        if l1.sum() == 0 and l2.sum() == 0 and l3.sum() == 0 and l4.sum() == 0:
            assert result[0] == 0

        return result

    def get_details(self):
        detected_by = pl.DataFrame(
            {
                "BRF": self.l1,
                "Statistical": self.l2,
                "Rules": self.l3,
                **self.rule_classifier.get_details(),
                **self.statistical_classifier.get_details(),
            }
        )
        if self.use_anomaly:
            detected_by = detected_by.with_columns(pl.Series("Anomaly", self.l4))
        return detected_by

    def add_transactions(self, transactions: pl.DataFrame, true_labels: npt.NDArray[np.bool_] | list[bool] | pl.Series):
        # Ensure true_labels is a Polars Series
        if not isinstance(true_labels, pl.Series):
            true_labels = pl.Series([true_labels]) if isinstance(true_labels, bool) else pl.Series(true_labels)

        if self.dataset == {}:
            self.dataset["Transactions"] = transactions
            self.dataset["Labels"] = true_labels
        else:
            # Align schemas before vstack
            existing_schema = self.dataset["Transactions"].schema
            transactions = transactions.cast(existing_schema)

            self.dataset["Transactions"] = self.dataset["Transactions"].vstack(transactions)
            self.dataset["Labels"] = np.concatenate((self.dataset["Labels"], true_labels))
