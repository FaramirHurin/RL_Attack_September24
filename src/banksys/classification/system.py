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
    dataset: pl.DataFrame
    retrain_every: timedelta


    def __init__(self, params: "ClassificationParameters", attack_start):
        self.ml_classifier = BalancedRandomForestClassifier(n_estimators=params.n_trees, n_jobs=1, sampling_strategy=params.balance_factor)  # type: ignore[assignment]
        self.anomaly_detection_classifier = IsolationForest(n_estimators=params.n_trees, n_jobs=1, contamination="auto")
        self.statistical_classifier = StatisticalClassifier(params.quantiles)
        self.rule_classifier = RuleBasedClassifier(params.rules)
        self.use_anomaly = params.use_anomaly
        self.training_duration = params.training_duration
        self.l1 = np.array([], dtype=np.bool)  # Placeholder for the first prediction, to be replaced in predict method
        self.l2 = np.array([], dtype=np.bool)  # Placeholder for the second prediction, to be replaced in predict method
        self.l3 = np.array([], dtype=np.bool)  # Placeholder for the third prediction, to be replaced in predict method
        self.l4 = np.array([], dtype=np.bool)  # Placeholder for the anomaly detection prediction, to be replaced in predict method
        assert not params.use_anomaly, "Anomaly detection is not supported in this version of the classification system."
        self.current_time = attack_start
        self.dataset = None
        # Timedelta of two weeks, used to determine the training period
        self.retrain_every = timedelta(days=1)  # type: ignore[assignment]

    def fit(self, transactions: pl.DataFrame, is_fraud: np.ndarray):
        logging.info("Fitting random forest")
        self.ml_classifier.fit(transactions, is_fraud)
        logging.info("Fitting anomaly classifier")
        self.anomaly_detection_classifier.fit(transactions)
        logging.info("Fitting statistical classifier")
        self.statistical_classifier.fit(transactions)
        logging.info("Done !")

        self.initial_transactions = transactions
        self.add_transactions(transactions, is_fraud)

    def predict(self, df: pl.DataFrame, true_labels, t) -> npt.NDArray[np.bool]:
        logging.debug("Predicting with RF")
        self.l1 = self.ml_classifier.predict(df).astype(np.bool)
        logging.debug("Predicting with statistical classifier")
        self.l2 = self.statistical_classifier.predict(df)
        logging.debug("Predicting with rule-based")
        self.l3 = self.rule_classifier.predict(df)
        result = self.l1 | self.l2 | self.l3
        if self.use_anomaly:
            logging.debug("Predicting with anomaly detection")
            label = self.anomaly_detection_classifier.predict(df)
            self.l4 = label == -1
            result = result | self.l4

        self.add_transactions(df, true_labels)
        self.evaluate_retraining(df, t)

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

    def add_transactions(self, transactions: pl.DataFrame, true_labels):
        """
        Add new transactions to the dataset.
        """
        # Ensure true_labels is a list/array of bools and matches length
        if not isinstance(true_labels, (list, tuple, pl.Series)):
            true_labels = [true_labels]

        if isinstance(true_labels, pl.Series):
            true_labels = true_labels.to_list()

        # Cast to bool
        try:
            true_labels = [bool(label) for label in true_labels]
        except:
            DEBUG = True

        assert len(true_labels) == len(transactions), "Length mismatch between labels and transactions"

        # Add the boolean column
        transactions = transactions.with_columns(
            pl.Series("is_fraud", true_labels).cast(pl.Boolean)
        )

        if self.dataset is None:
            self.dataset = transactions
        else:
            try:
                self.dataset = pl.concat([self.dataset, transactions], rechunk=True)
            except:
                # Modify self.datasrt to accept booolean as is_fraud
                self.dataset = self.dataset.with_columns(
                    pl.Series("is_fraud", self.dataset["is_fraud"].to_numpy().astype(bool))
                )


    def evaluate_retraining(self, transactions: pl.DataFrame, t):
        """
        Evaluate if the model should be retrained based on the number of new transactions.
        """
        if self.current_time + self.retrain_every < t:
            logging.info("Retraining model")
            self.fit(self.dataset, self.dataset["is_fraud"].to_numpy())
            self.current_time = t
