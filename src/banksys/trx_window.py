from datetime import datetime, timedelta
from typing import Sequence

from .transaction import Transaction


class TransactionWindow:
    def __init__(self):
        self.transactions = list[Transaction]()
        self.fields = Transaction.field_names()

    def update(self, current_time: datetime, expire_after: timedelta):
        remove_before = current_time - expire_after
        i = 0
        while i < len(self.transactions) and self.transactions[i].timestamp < remove_before:
            i += 1
        self.transactions = self.transactions[i:]

    def add(self, transaction: Transaction):
        self.transactions.append(transaction)

    def get_window(self) -> list[Transaction]:
        return self.transactions

    def count_and_mean(self, aggregation_windows: Sequence[timedelta], timestamp: datetime):
        """
        Count the number of transactions in the given time windows.
        Note: aggregation_windows must be sorted in ascending order.
        """
        results = dict[str, float]()
        total_amount = 0
        self.update(timestamp, max(aggregation_windows))
        window_size = len(self.transactions)
        i = len(self.transactions) - 1
        for delta in aggregation_windows:
            window_start = timestamp - delta
            while i >= 0 and self.transactions[i].timestamp >= window_start:
                total_amount += self.transactions[i].amount
                i -= 1
            n = window_size - i - 1
            results[f"card_n_trx_last_{delta}"] = n
            results[f"card_mean_amount_last_{delta}"] = total_amount / n if n > 0 else 0.0
        return results

    def count_and_risk(self, aggregation_windows: Sequence[timedelta], timestamp: datetime):
        """
        Count the number of transactions in the given time windows and compute the risk.
        Note: aggregation_windows must be sorted in ascending order.
        """
        results = dict[str, float]()
        n_frauds = 0
        self.update(timestamp, max(aggregation_windows))
        window_size = len(self.transactions)
        i = len(self.transactions) - 1
        for duration in aggregation_windows:
            window_start = timestamp - duration
            while i >= 0 and self.transactions[i].timestamp >= window_start:
                if self.transactions[i].predicted_label:
                    n_frauds += 1
                i -= 1
            n = window_size - i - 1
            results[f"terminal_n_trx_last_{duration}"] = n
            results[f"terminal_risk_last_{duration}"] = n_frauds / n if n > 0 else 0.0
        return results

    def __len__(self):
        return len(self.transactions)

    def __iter__(self):
        return iter(self.transactions)

    def __getitem__(self, item: int):
        return self.transactions[item]

    @property
    def is_empty(self):
        return len(self.transactions) == 0
