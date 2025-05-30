# Import isolation forest
from typing import TYPE_CHECKING, Callable, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from ..transaction_registry import TransactionsRegistry
import polars as pl
from datetime import timedelta

from ..transaction import Transaction

if TYPE_CHECKING:
    from banksys import Banksys


def max_trx_day(transaction: Transaction, registry: TransactionsRegistry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(days=1)
    same_day_transactions = registry.get_between(start, transaction.timestamp)
    return len(same_day_transactions) > max_number


def max_trx_hour(transaction: Transaction, registry: TransactionsRegistry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(hours=1)
    same_hour_transactions = registry.get_between(start, transaction.timestamp)
    return len(same_hour_transactions) > max_number


def max_trx_week(transaction: Transaction, registry: TransactionsRegistry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(weeks=1)
    same_week_transactions = registry.get_between(start, transaction.timestamp)
    return len(same_week_transactions) > max_number


rules_dict = {
    "max_trx_day": max_trx_day,
    "max_trx_hour": max_trx_hour,
    "max_trx_week": max_trx_week,
}


class RuleBasedClassifier:
    def __init__(
        self,
        rules: list[Callable[[Transaction, TransactionsRegistry, float], bool]],
        banksys: "Banksys",
        rule_values: dict,
    ):
        self.rules = dict(zip(rule_values.keys(), rules))
        self.rule_values = rule_values
        self.banksys = banksys

    def predict_transaction(self, transaction: Transaction):
        registry = self.banksys.cards[transaction.card_id]
        for rule_name in self.rules.keys():
            rule = self.rules[rule_name]
            value = self.rule_values[rule_name]
            if rule(transaction, registry, value):
                return True
        return False

    def predict_dataframe(self, df: pd.DataFrame):
        # Does not matter whether we put is_fraud=True or False here because we are not using it
        data = pl.from_pandas(df)
        transactions = [Transaction.from_features(False, **kwargs) for kwargs in data.iter_rows(named=True)]
        labels = []
        for transaction in transactions:
            predicted = self.predict_transaction(transaction)
            labels.append(predicted)
        return np.array(labels)

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
