from dataclasses import dataclass
import numpy as np
from datetime import datetime
from .transaction import Transaction


@dataclass
class Terminal:
    id: int
    x: float
    y: float
    days_aggregation: tuple[int, ...]
    transactions: list[Transaction]

    def __init__(self, id: int, x: float, y: float, days_aggregation: tuple[int, ...] = (1, 7, 30)):
        self.id = id
        self.x = x
        self.y = y
        self.days_aggregation = days_aggregation
        self.transactions = []

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    @property
    def feature_names(self):
        return ["x", "y"] + [f"terminal_agg_{days}" for days in self.days_aggregation]

    def features(self, current_time: datetime):
        aggregated_features = [0.0 for _ in self.days_aggregation]
        return np.array([self.x, self.y, *aggregated_features], dtype=np.float32)
        # TODO: compute the aggregated features for the terminal from the past transactions
        # columns_names_avg = {}
        # columns_names_count = {}

        # terminal_transactions = self.terminals_transactions[terminal.id]
        # terminal_transactions = terminal_transactions[terminal_transactions["timestamp"] < current_time]

        # # Compute aggregated features for the terminal
        # for days in self.days_aggregation:
        #     # Select transactions from the last days
        #     terminal_transactions_days = terminal_transactions[terminal_transactions["timestamp"] > current_time - days]
        #     # Compute mean and count
        #     columns_names_avg[days] = terminal_transactions_days.mean()
        #     columns_names_count[days] = terminal_transactions_days.count()

        # trx = pd.Series()
        # for day in columns_names_avg.keys():
        #     # TODO Correct naming of columns
        #     trx["AVG_" + str(day)] = columns_names_avg[day]
        #     trx["COUNT_" + str(day)] = columns_names_count[day]
