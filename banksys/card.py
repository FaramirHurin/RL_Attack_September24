from dataclasses import dataclass
import numpy as np
from datetime import datetime
from datetime import timedelta

from .transaction import Transaction
from .has_ordered_transactions import HasOrderedTransactions


# In case there is an equality in the priority queue, it compares
# the cards. Therefore, we want the order to be defined.
@dataclass(order=True)
class Card(HasOrderedTransactions):
    id: int
    is_credit: bool
    customer_x: float
    customer_y: float
    transactions: list[Transaction]
    """Transactions, ordered by timestamp"""
    days_aggregation: tuple[timedelta, ...]

    def __init__(
        self, id: int, is_credit: bool, x: float, y: float, days_aggregation: tuple[timedelta, ...] = (timedelta(1), timedelta(7))
    ):
        self.id = id
        self.is_credit = is_credit
        self.customer_x = x
        self.customer_y = y
        self.days_aggregation = days_aggregation
        super().__init__()

    @property
    def feature_names(self):
        prefix = "CUSTOMER_ID_"
        nb = "NB_TX_"
        avg = "AVG_AMOUNT_"
        suffix = "DAY_WINDOW"

        AGGREGATE_NB = [prefix + nb + str(days) + suffix for days in self.days_aggregation]
        AGGREGATE_RISK = [prefix + avg + str(days) + suffix for days in self.days_aggregation]

        to_return = ["customer_x", "customer_y"] + AGGREGATE_NB + AGGREGATE_RISK
        return to_return

    def features(self, current_time: datetime):
        # TODO: add the mean/median terminal location ?
        nb = list[float]()
        avg = list[float]()
        for n_days in self.days_aggregation:
            start_index = self._find_index(current_time - n_days)
            stop_index = self._find_index(current_time)
            # Select transactions from the last n_days
            trx_days = self.transactions[start_index:stop_index]
            # Compute count
            nb.append(len(trx_days))

            # Compute mean
            if len(trx_days) == 0:
                avg.append(0)
            else:
                # Compute the average amount of the transactions
                avg.append(np.mean([transaction.amount for transaction in trx_days]).item())
        return np.array([self.customer_x, self.customer_y] + nb + avg, dtype=np.float32)

    def __hash__(self) -> int:
        return self.id
