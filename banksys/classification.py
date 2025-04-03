from abc import ABC, abstractmethod
from typing import Callable
from transaction import Transaction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """
    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles_values = quantiles

    def fit(self, x: pd.DataFrame):
        # Select the quantiles of all the considered_features in X
        self.quantiles_values = x[self.considered_features].quantile(self.quantiles)

    def predict(self, x: pd.DataFrame):
        return x[self.considered_features].isin(self.quantiles_values).all(axis=1)


class RuleClassification(ABC):
    @abstractmethod
    def classify(self, transaction: Transaction) -> bool:
        """
        Classify a transaction based on the rule.
        """

    def __call__(self, transaction: Transaction):
        return self.classify(transaction)

class RuleBasedClassifier:
    """
    Implement rules logic. It could be a query frequency check, a value check, etc.
    TODO
    """

    def __init__(self, rules: list[Callable[[Transaction], bool]]):
        self.rules = rules

    def predict(self, x:pd.DataFrame) -> np.ndarray:
        # Return False for each trx in x
        return np.zeros(x.shape[0], dtype=bool)


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier

    def __init__(self, clf: RandomForestClassifier, features_for_quantiles: list, quantiles: list, rules):
        self.ml_classifier = clf
        self.rule_classifier = RuleBasedClassifier(rules)
        self.statistical_classifier = StatisticalClassifier(["amount"], [0.02, 0.98])

    def fit(self, transactions: pd.DataFrame, is_fraud: np.ndarray):
        self.ml_classifier.fit(transactions, is_fraud)

    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        classification_prediction:np.ndarray = self.ml_classifier.predict(transactions)
        statistical_prediction:np.ndarray = self.statistical_classifier.predict(transactions)
        rule_based_prediction:np.ndarray = self.rule_classifier.predict(transactions)

        classification_prediction = np.logical_or(classification_prediction, statistical_prediction)
        classification_prediction = np.logical_or(classification_prediction, rule_based_prediction)

        return classification_prediction

