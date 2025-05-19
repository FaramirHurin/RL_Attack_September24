from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .transaction import Transaction
from .card import Card

# Import isolation forest
from sklearn.ensemble import IsolationForest


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """

    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles = quantiles
        self.quantiles_df = None

    def fit(self, x: pd.DataFrame):
        # Select the quantiles of all the considered_features in X
        self.quantiles_df = x[self.considered_features].quantile(q=self.quantiles)

    def predict(self, x: pd.DataFrame):
        assert self.quantiles_df is not None, "StatisticalClassifier not fitted yet"
        return x[self.considered_features].isin(self.quantiles_df).all(axis=1).to_numpy()


class RuleBasedClassifier:
    def __init__(self, rules: list[Callable[[Transaction, list[Transaction]], bool]], banksys):
        self.rules = rules
        self.banksys = banksys

    def predict(self, transaction: Transaction) -> np.ndarray:
        return False
        card = self.banksys.cards[transaction.card_id]
        transactions = [trx for trx in card.transactions if trx.timestamp < transaction.timestamp]
        for rule in self.rules:
            if rule(transaction, transactions):
                DEBUG = True
                return True
        return False


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier
    anomaly_detection_classifier: IsolationForest

    def __init__(
        self,
        clf: RandomForestClassifier,
        anomaly_detection_clf: IsolationForest,
        features_for_quantiles: list[str],
        quantiles: list[float],
        banksys,
        rules,
    ):
        self.ml_classifier = clf
        self.anomaly_detection_classifier = anomaly_detection_clf
        self.rule_classifier = RuleBasedClassifier(rules, banksys)
        self.statistical_classifier = StatisticalClassifier(features_for_quantiles, quantiles)

    def fit(self, transactions: pd.DataFrame, is_fraud: np.ndarray):
        self.ml_classifier.fit(transactions, is_fraud)
        self.anomaly_detection_classifier.fit(transactions)
        self.statistical_classifier.fit(transactions)

    def predict(self, transactions_df: pd.DataFrame, transaction: Transaction) -> npt.NDArray[np.bool_]:
        # transactions_df = pd.DataFrame(transactions)

        classification_prediction = self.ml_classifier.predict(transactions_df)
        # anomaly_prediction = self.anomaly_detection_classifier.predict(transactions_df) == -1
        statistical_prediction = self.statistical_classifier.predict(transactions_df)
        rule_based_prediction = False  #  self.rule_classifier.predict(transaction)

        # or anomaly_prediction
        to_return = int(classification_prediction or statistical_prediction or rule_based_prediction)

        if to_return:
            debug = True

        return to_return


'''
class RuleClassification:
    @abstractmethod
    def classify(self, transaction: Transaction) -> bool:
        """
        Classify a transaction based on the rule.
        """

    def __call__(self, transaction: Transaction):
        return self.classify(transaction)
'''
