# Import isolation forest
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import polars as pl

from ..transaction import Transaction


def max_trx_hour(transaction: Transaction, registry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(minutes=60)
    recent_transactions = [trx for trx in registry.transactions if start <= trx.timestamp < transaction.timestamp]
    to_return = len(recent_transactions) >= max_number
    if to_return:
        Debug = True
    return to_return


def max_trx_day(transaction: Transaction, registry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(hours=24)
    recent_transactions = [trx for trx in registry.transactions if start <= trx.timestamp <= transaction.timestamp]
    return len(recent_transactions) >= max_number


def max_trx_week(transaction: Transaction, registry, max_number: float) -> bool:
    start = transaction.timestamp - timedelta(days=7)
    recent_transactions = [trx for trx in registry.transactions if start <= trx.timestamp <= transaction.timestamp]
    return len(recent_transactions) >= max_number


rules_dict = {
    "max_trx_day": max_trx_day,
    "max_trx_hour": max_trx_hour,
    "max_trx_week": max_trx_week,
}


class RuleBasedClassifier:
    def __init__(
        self,
        max_values: dict[timedelta, float],
    ):
        self.max_values = max_values
        self.rule_values = max_values

    def predict(self, df: pl.DataFrame):
        detected_by = pl.DataFrame()
        labels = np.full(len(df), False, dtype=np.bool)
        for td, max_value in self.max_values.items():
            colname = f"card_n_trx_last_{td}"
            if colname not in df.columns:
                raise ValueError(f"DataFrame does not contain column for {td}.")
            y = df[colname] > max_value
            detected_by = detected_by.with_columns(pl.Series(colname, y))
            labels = labels | y.to_numpy().astype(np.bool)
        return labels, detected_by
