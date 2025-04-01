from abc import ABC, abstractmethod
from typing import Callable
from banksys.transaction import Transaction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """

    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles = quantiles

    def fit(self, x: pd.DataFrame):
        # Select the quantiles of all the considered_features in X
        self.quantiles_values = x[self.considered_features].quantile(self.quantiles)

    def predict(self, x: pd.DataFrame):
        raise NotImplementedError()
        # Check if the value of each considered_feature in X is in the quantiles_values
        # return x[self.considered_features].isin(self.quantiles_values).all(axis=1)


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

    def predict(self, x):
        """
        If any of the rules is not satisfied, return False.
        Otherwise (i.e. all the rules are satisfied), return True.
        """
        for rule in self.rules:
            if not rule(x):
                return False
        return True


class ClassificationSystem:
    ml_classifier: RandomForestClassifier
    rule_classifier: RuleBasedClassifier
    statistical_classifier: StatisticalClassifier

    def __init__(self, clf: RandomForestClassifier, features_for_quantiles: list, quantiles: list, rules):
        self.ml_classifier = clf
        self.rule_classifier = RuleBasedClassifier(rules)
        self.statistical_classifier = StatisticalClassifier(["amount"], [0.02, 0.98])

    def fit(self, x: list[Transaction], is_fraud: np.ndarray | pd.Series):
        df = pd.DataFrame([t.features for t in x])
        self.ml_classifier.fit(df, is_fraud)

    def _make_df(self, transactions: list[Transaction]) -> pd.DataFrame:
        """
        TODO
        """
        raise NotImplementedError()

    def predict(self, transactions: list[Transaction]):
        df = self._make_df(transactions)
        classification_prediction = self.ml_classifier.predict(df)
        statistical_prediction = self.statistical_classifier.predict(df)
        rule_based_prediction = self.rule_classifier.predict(df)
        return classification_prediction or statistical_prediction or rule_based_prediction
