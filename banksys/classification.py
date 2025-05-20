from typing import Callable, overload
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .transaction import Transaction

# Import isolation forest
from sklearn.ensemble import IsolationForest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banksys import Banksys


def max_trx_day(transaction: Transaction, transactions: list[Transaction], max_number: int = 7) -> bool:  # 7
    same_day_transactions = [trx for trx in transactions if trx.timestamp.date() == transaction.timestamp.date()]
    return len(same_day_transactions) > max_number


def max_trx_hour(transaction: Transaction, transactions: list[Transaction], max_number: int = 4) -> bool:
    same_hour_transactions = [trx for trx in transactions if trx.timestamp.hour == transaction.timestamp.hour]
    return len(same_hour_transactions) > max_number


def max_trx_week(transaction: Transaction, transactions: list[Transaction], max_number: int = 20) -> bool:
    same_week_transactions = [trx for trx in transactions if trx.timestamp.isocalendar()[1] == transaction.timestamp.isocalendar()[1]]
    return len(same_week_transactions) > max_number


def positive_amount(transaction: Transaction, transactions: list[Transaction], value: None = None) -> bool:
    return transaction.amount < 0.01


rules_dict = {
    "max_trx_day": max_trx_day,
    "max_trx_hour": max_trx_hour,
    "max_trx_week": max_trx_week,
    "positive_amount": positive_amount,
}


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """

    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles = quantiles
        self.quantiles_df = None

    def fit(self, transactions_df: pd.DataFrame):
        assert all(f in transactions_df.columns for f in self.considered_features)
        self.quantiles_df = transactions_df[self.considered_features].quantile(q=self.quantiles)

    def predict_transaction(self, transaction: Transaction) -> bool:
        assert self.quantiles_df is not None, "The classifier has not been fitted yet."
        for feature in self.considered_features:
            value = getattr(transaction, feature)
            mmin, mmax = self.quantiles_df[feature]
            if not mmin <= value <= mmax:
                return True
        return False

    def predict_dataframe(self, df: pd.DataFrame) -> npt.NDArray[np.bool]:
        assert self.quantiles_df is not None, "The classifier has not been fitted yet."
        return df[self.considered_features].isin(self.quantiles_df).all(axis=1).to_numpy()

    @overload
    def predict(self, transaction: Transaction, /) -> bool: ...
    @overload
    def predict(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]: ...

    def predict(self, df_or_transaction, /):
        match df_or_transaction:
            case Transaction() as transaction:
                return self.predict_transaction(transaction)
            case pd.DataFrame() as df:
                return self.predict_dataframe(df)
        raise ValueError("Invalid input type. Expected Transaction or DataFrame.")


class RuleBasedClassifier:
    def __init__(self, rules: list[Callable[[Transaction, list[Transaction], float], bool]], banksys: "Banksys", rule_values: dict):
        self.rules = dict(zip(rule_values.keys(), rules))
        self.rule_values = rule_values
        self.banksys = banksys

    def predict_transaction(self, transaction: Transaction):
        card = self.banksys.cards[transaction.card_id]
        i = card._find_index(transaction.timestamp)
        transactions = card.transactions[:i]
        for rule_name in self.rules.keys():
            rule = self.rules[rule_name]
            value = self.rule_values[rule_name]
            if rule(transaction, transactions, value):
                return True
        return False

    def predict_dataframe(self, df: pd.DataFrame):
        raise NotImplementedError("Rule-based classifier does not support DataFrame input.")
        return df.apply(lambda row: self.predict_transaction(row), axis=1).to_numpy()

    @overload
    def predict(self, transaction: Transaction, /) -> bool: ...

    @overload
    def predict(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]: ...

    def predict(self, transaction_or_df, /):
        match transaction_or_df:
            case Transaction():
                return self.predict_transaction(transaction_or_df)
            case pd.DataFrame():
                return self.predict_dataframe(transaction_or_df)
        raise ValueError("Invalid input type. Expected Transaction or DataFrame.")


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
        banksys: "Banksys",
    ):
        self.ml_classifier = clf
        self.banksys = banksys
        self.anomaly_detection_classifier = anomaly_detection_clf
        self.statistical_classifier = StatisticalClassifier(features_for_quantiles, quantiles)

    def fit(self, transactions: pd.DataFrame, is_fraud: np.ndarray):
        self.ml_classifier.n_jobs = -1  # type: ignore[assignment]
        self.anomaly_detection_classifier.n_jobs = -1  # type: ignore[assignment]

        self.ml_classifier.fit(transactions, is_fraud)
        self.anomaly_detection_classifier.fit(transactions)
        self.statistical_classifier.fit(transactions)

        self.ml_classifier.n_jobs = 1  # type: ignore[assignment]
        self.anomaly_detection_classifier.n_jobs = 1  # type: ignore[assignment]

    def set_rules(self, rules_names: list, rules_values: dict):
        rules = [rules_dict[rule] for rule in rules_names]
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
        res = self.anomaly_detection_classifier.predict(df).item()
        is_fraud = res == -1
        return is_fraud

    def _predict_dataframe(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]:
        l1 = self.rule_classifier.predict_dataframe(df)
        l2 = self.ml_classifier.predict(df).astype(np.bool)
        l3 = self.statistical_classifier.predict_dataframe(df)
        l4 = self.anomaly_detection_classifier.predict(df) == -1
        return l1 | l2 | l3 | l4

    def predict(self, transaction_or_df, /):
        match transaction_or_df:
            case Transaction() as t:
                return self._predict_transaction(t)
            case pd.DataFrame() as df:
                return self._predict_dataframe(df)
        raise ValueError("Invalid input type. Expected `Transaction` or `DataFrame`.")
