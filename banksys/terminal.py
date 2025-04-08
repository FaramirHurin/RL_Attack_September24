from dataclasses import dataclass
import numpy as np
from datetime import datetime
from datetime import timedelta

from .transaction import Transaction
from .has_ordered_transactions import HasOrderedTransactions


@dataclass
class Terminal(HasOrderedTransactions):
    id: int
    x: float
    y: float
    days_aggregation: tuple[timedelta, ...]
    transactions: list[Transaction]
    """Transactions, ordered by timestamp"""

    def __init__(self, id: int, x: float, y: float, days_aggregation: tuple[timedelta, ...] = (timedelta(1), timedelta(7))):
        self.id = id
        self.x = x
        self.y = y
        self.days_aggregation = days_aggregation
        super().__init__()

    @property
    def feature_names(self):
        prefix = "TERMINAL_ID_"
        nb = "NB_TX_"
        risk = "RISK_"
        suffix = "DAY_WINDOW"

        AGGREGATE_NB = [prefix + nb + str(days) + suffix for days in self.days_aggregation]
        AGGREGATE_RISK = [prefix + risk + str(days) + suffix for days in self.days_aggregation]

        to_return = ["terminal_x", "terminal_y"] + AGGREGATE_NB + AGGREGATE_RISK
        return to_return

    def features(self, current_time: datetime):
        nb = list[float]()
        risk = list[float]()

        for n_days in self.days_aggregation:
            start_index = self._find_index(current_time - n_days)
            stop_index = self._find_index(current_time)
            # Select transactions from the last n_days
            trx_days = self.transactions[start_index:stop_index]
            nb.append(len(trx_days))

            # Compute risk
            positive_transactions = [transaction for transaction in trx_days if transaction.label == 1]
            if len(positive_transactions) == 0:
                risk.append(0)
            else:
                # Compute the average amount of the transactions
                risk.append(len(positive_transactions) / len(trx_days))
        return np.array([self.x, self.y] + nb + risk, dtype=np.float32)
