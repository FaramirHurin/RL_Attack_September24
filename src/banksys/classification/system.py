import logging
from datetime import timedelta
from typing import TYPE_CHECKING
import numpy.typing as npt
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

    def __init__(self, params: "ClassificationParameters"):
        self.ml_classifier = BalancedRandomForestClassifier(n_estimators=params.n_trees, n_jobs=1, sampling_strategy=params.balance_factor)  # type: ignore[assignment]
        self.anomaly_detection_classifier = IsolationForest(n_estimators=params.n_trees, n_jobs=1, contamination=params.contamination)
        self.statistical_classifier = StatisticalClassifier(params.quantiles_features, params.quantiles_values)
        self.rule_classifier = RuleBasedClassifier(params.rules)
        self.use_anomaly = params.use_anomaly
        self.training_duration = params.training_duration
        self.l1 = np.array([], dtype=np.bool)  # Placeholder for the first prediction, to be replaced in predict method
        self.l2 = np.array([], dtype=np.bool)  # Placeholder for the second prediction, to be replaced in predict method
        self.l3 = np.array([], dtype=np.bool)  # Placeholder for the third prediction, to be replaced in predict method
        self.l4 = np.array([], dtype=np.bool)  # Placeholder for the anomaly detection prediction, to be replaced in predict method

    def fit(self, transactions: pl.DataFrame, is_fraud: np.ndarray):
        logging.info("Fitting random forest")
        self.ml_classifier.n_jobs = -1  # type: ignore[assignment]
        self.ml_classifier.fit(transactions, is_fraud)
        self.ml_classifier.n_jobs = 1  # type: ignore[assignment]
        logging.info("Fitting anomaly classifier")
        self.anomaly_detection_classifier.fit(transactions)
        logging.info("Fitting statistical classifier")
        self.statistical_classifier.fit(transactions.to_pandas())
        logging.info("Done !")

    def predict(self, df: pl.DataFrame) -> npt.NDArray[np.bool]:
        logging.debug("Predicting with RF")
        self.l1 = self.ml_classifier.predict(df).astype(np.bool)
        logging.debug("Predicting with statistical classifier")
        self.l2 = self.statistical_classifier.predict_dataframe(df.to_pandas())
        logging.debug("Predicting with rule-based")
        self.l3 = self.rule_classifier.predict(df)
        result = self.l1 | self.l2 | self.l3
        if self.use_anomaly:
            logging.debug("Predicting with anomaly detection")
            self.l4 = self.anomaly_detection_classifier.predict(df) == -1
            result = result | self.l4
        return result

    def get_details(self):
        detected_by = pl.DataFrame({"BRF": self.l1, "Statistical": self.l2, "Rules": self.l3, **self.rule_classifier.get_details()})
        if self.use_anomaly:
            detected_by = detected_by.with_columns(pl.Series("Anomaly", self.l4))
        return detected_by
