from dataclasses import dataclass
from typing import Sequence
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd

from .transaction import Transaction
from .transaction_registry import TransactionsRegistry


@dataclass
class Terminal(TransactionsRegistry):
    id: int
    x: float
    y: float
    transactions: list[Transaction]
    """Transactions, ordered by timestamp"""

    def __init__(self, id: int, x: float, y: float):
        super().__init__()
        self.id = id
        self.x = x
        self.y = y

    @staticmethod
    def feature_names(aggregation_windows: Sequence[timedelta]):
        prefix = "TERMINAL_ID_"
        nb = "NB_TX_"
        risk = "RISK_"
        suffix = "DAY_WINDOW"

        AGGREGATE_NB = [prefix + nb + str(days) + suffix for days in aggregation_windows]
        AGGREGATE_RISK = [prefix + risk + str(days) + suffix for days in aggregation_windows]

        to_return = ["terminal_x", "terminal_y"] + AGGREGATE_NB + AGGREGATE_RISK
        return to_return

    def features(self, current_time: datetime, aggregation_windows: Sequence[timedelta]):
        nb = list[float]()
        risk = list[float]()

        stop_index = self._find_index(current_time)
        for n_days in sorted(aggregation_windows):
            start_index = self._find_index(current_time - n_days)
            # Select transactions from the last n_days
            trx_days = self.transactions[start_index:stop_index]
            nb.append(len(trx_days))

            # Compute risk
            positive_transactions = [transaction for transaction in trx_days if transaction.predicted_label]
            if len(positive_transactions) == 0:
                risk.append(0)
            else:
                # Compute the average amount of the transactions
                risk.append(len(positive_transactions) / len(trx_days))
        return np.array([self.x, self.y] + nb + risk, dtype=np.float32)

    @staticmethod
    def from_df(df: pd.DataFrame):
        terminals = list[Terminal]()
        for _, (payee_id, payee_x, payee_y) in df[["payee_id", "payee_x", "payee_y"]].iterrows():
            terminals.append(Terminal(payee_id, payee_x, payee_y))
        return terminals
