from dataclasses import dataclass
import numpy as np
from typing import Sequence
from datetime import datetime
from datetime import timedelta
import pandas as pd

from .transaction import Transaction
from .transaction_registry import TransactionsRegistry


# In case there is an equality in the priority queue, it compares
# the cards. Therefore, we want the order to be defined.
@dataclass(order=True)
class Card(TransactionsRegistry):
    id: int
    is_credit: bool
    customer_x: float
    customer_y: float
    transactions: list[Transaction]
    balance: float
    """Transactions, ordered by timestamp"""

    def __init__(
        self,
        id: int,
        is_credit: bool,
        x: float,
        y: float,
        balance: float,
    ):
        self.id = int(id)
        self.is_credit = bool(is_credit)
        self.customer_x = int(x)
        self.customer_y = int(y)
        self.balance = balance
        super().__init__()

    @staticmethod
    def feature_names(aggregation_windows: Sequence[timedelta]):
        prefix = "CUSTOMER_ID_"
        nb = "NB_TX_"
        avg = "AVG_AMOUNT_"
        suffix = "DAY_WINDOW"

        AGGREGATE_NB = [prefix + nb + str(days) + suffix for days in aggregation_windows]
        AGGREGATE_RISK = [prefix + avg + str(days) + suffix for days in aggregation_windows]

        to_return = ["customer_x", "customer_y"] + AGGREGATE_NB + AGGREGATE_RISK
        return to_return

    def features(self, current_time: datetime, aggregation_windows: Sequence[timedelta]):
        # TODO: add the mean/median terminal location ?
        nb = list[float]()
        avg = list[float]()
        stop_index = self._find_index(current_time)
        for n_days in aggregation_windows:
            start_index = self._find_index(current_time - n_days)
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

    @staticmethod
    def from_df(df: pd.DataFrame):
        cards = list[Card]()
        for _, (payer_id, payer_x, payer_y, balance) in df[["payer_id", "payer_x", "payer_y", "balance"]].iterrows():
            cards.append(Card(payer_id, False, payer_x, payer_y, balance))
        return cards
