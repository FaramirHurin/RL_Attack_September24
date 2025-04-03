from dataclasses import dataclass
import numpy as np
from datetime import datetime
from transaction import Transaction
from datetime import timedelta

@dataclass
class Terminal:
    id: int
    x: float
    y: float
    days_aggregation: tuple[int, ...]
    transactions: list[Transaction]

    def __init__(self, id: int, x: float, y: float, days_aggregation: tuple[timedelta, ...] = (1, 7)): #, 30
        self.id = id
        self.x = x
        self.y = y
        self.days_aggregation:list[timedelta, ...] = [timedelta(days=day) for day in days_aggregation]
        self.transactions = []

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    @property
    def feature_names(self):
        prefix = 'TERMINAL_ID_'
        nb = "NB_TX_"
        risk = "RISK_"
        suffix = "DAY_WINDOW"

        AGGREGATE_NB = [prefix + nb + str(days) + suffix for days in self.days_aggregation]
        AGGREGATE_RISK = [prefix + risk + str(days) + suffix for days in self.days_aggregation]

        to_return = ["x", "y"] + AGGREGATE_NB + AGGREGATE_RISK

        return to_return

        #return ["x", "y"] + [f"terminal_agg_{days}" for days in self.days_aggregation]

    def features(self, current_time: datetime):
        nb:[float] = []
        risk:[float] = []

        transactions = [transaction for transaction in self.transactions if transaction.timestamp < current_time]

        for days in self.days_aggregation:
            # Select transactions from the last days
            trx_days = [transaction for transaction in transactions if transaction.timestamp >
                                          current_time - days]
            # Compute count
            nb.append(len(trx_days))

            # Compute risk
            positive_transactions = [transaction for transaction in trx_days if transaction.label == 1]
            if len(positive_transactions) == 0:
                risk.append(0)
            else:
                # Compute the average amount of the transactions
                risk.append(len(positive_transactions) / len(trx_days))

        return np.array([self.x, self.y] + nb + risk, dtype=np.float32)
