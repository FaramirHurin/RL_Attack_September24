from dataclasses import dataclass
import numpy as np
from datetime import datetime
from datetime import timedelta

from .transaction import Transaction


@dataclass
class Card:
    id: int
    is_credit: bool
    customer_x: float
    customer_y: float
    transactions: list[Transaction]
    days_aggregation: tuple[timedelta, ...]

    def __init__(
        self, id: int, is_credit: bool, x: float, y: float, days_aggregation: tuple[timedelta, ...] = (timedelta(1), timedelta(7))
    ):  # , 30
        self.id = id
        self.is_credit = is_credit
        self.x = x
        self.y = y
        self.days_aggregation = days_aggregation
        self.transactions = []

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

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
        nb = list[float]()
        avg = list[float]()

        transactions = [transaction for transaction in self.transactions if transaction.timestamp < current_time]

        for days in self.days_aggregation:
            # Select transactions from the last days
            trx_days = [transaction for transaction in transactions if transaction.timestamp > current_time - days]
            # Compute count
            nb.append(len(trx_days))

            # Compute mean
            if len(trx_days) == 0:
                avg.append(0)
            else:
                # Compute the average amount of the transactions
                avg.append(np.mean([transaction.amount for transaction in trx_days]).item())
        return np.array([self.x, self.y] + nb + avg, dtype=np.float32)
