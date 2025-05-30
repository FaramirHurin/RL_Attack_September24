# Import isolation forest
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from ..transaction import Transaction
from ..transaction_registry import TransactionsRegistry

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

    def _predict_job(self, data: pl.DataFrame):
        labels = []
        transactions = [Transaction.from_features(False, **kwargs) for kwargs in data.iter_rows(named=True)]
        for transaction in transactions:
            predicted = self.predict_transaction(transaction)
            labels.append(predicted)
        return np.array(labels, dtype=np.bool)

    def predict_dataframe(self, df: pd.DataFrame):
        n_jobs = mp.cpu_count()
        data = pl.from_pandas(df)
        logging.debug(f"Starting rule predictions with {n_jobs}, total rows: {len(df)}")
        start = datetime.now()
        with mp.Pool(n_jobs) as pool:
            results = pool.map(self._predict_job, data.iter_slices())
        logging.debug(f"Rule predictions completed in {datetime.now() - start}")
        return np.concatenate(results)

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
