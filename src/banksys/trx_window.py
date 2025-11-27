from datetime import datetime, timedelta
from typing import Sequence
from .transaction import Transaction


class TransactionWindow:
    def __init__(self, aggregation_windows: Sequence[timedelta]):
        self.aggregation_windows = sorted(aggregation_windows)
        """Aggregation time windows sorted in ascending order."""
        self.max_window = self.aggregation_windows[-1]
        """The largest aggregation time window."""
        self.transactions = list[Transaction]()
        """The timestamps of all transactions associated with this terminal."""

    def add(self, trx: Transaction, *, update=False):
        self.transactions.append(trx)
        if update:
            self.update(trx.timestamp)

    def update(self, t: datetime):
        window_start = t - self.max_window
        i = 0
        while i < len(self.transactions) and self.transactions[i].timestamp < window_start:
            i += 1
        if i > 0:
            self.transactions = self.transactions[i:]

    def __len__(self):
        return len(self.transactions)

    @property
    def is_empty(self):
        return len(self.transactions) == 0

    def __getitem__(self, index: int):
        return self.transactions[index]

    def compute_start_index_by_window(self, t: datetime):
        n_trx = len(self.transactions)
        n_windows = len(self.aggregation_windows)
        if n_trx <= 1:
            return [0] * n_windows
        res = list[int]()
        window_size = 1
        ti = n_trx - 1 - window_size
        wi = 0
        window_start = t - self.aggregation_windows[wi]
        while wi < n_windows - 1 and ti >= 0:  # We ignore the largest window (the last)
            if self.transactions[ti].timestamp < window_start:  # If the timestamp is out of the window
                res.append(n_trx - window_size)
                wi += 1
                window_start = t - self.aggregation_windows[wi]
            else:
                ti -= 1
                window_size += 1
        for wi in range(wi, n_windows):
            # If there are windows that have not been considered,
            # then all the transactions are to be counted in each of them.
            res.append(0)
        return res

    def compute_counts_by_window(self, t: datetime):
        """
        :param t: Current time
        """
        if self.is_empty:
            return [0] * len(self.aggregation_windows)
        n_trx = len(self.transactions)
        n_windows = len(self.aggregation_windows)
        counts = list[int]()
        window_size = 0
        ti = n_trx - 1
        wi = 0
        window_start = t - self.aggregation_windows[wi]
        while wi < n_windows - 1 and ti >= 0:  # We ignore the largest window (the last)
            if self.transactions[ti].timestamp < window_start:  # If the timestamp is out of the window
                counts.append(window_size)
                wi += 1
                window_start = t - self.aggregation_windows[wi]
            else:
                ti -= 1
                window_size += 1
        for wi in range(wi, n_windows):
            # If there are windows that have not been considered,
            # then all the transactions are to be counted in each of them.
            counts.append(n_trx)
        return counts

    def compute_avg_amount_and_count_by_window(self, t: datetime):
        counts = list[int]()
        amounts = list[float]()
        total_amount = 0
        window_size = len(self.transactions)
        i = len(self.transactions) - 1
        for delta in self.aggregation_windows:
            window_start = t - delta
            while i >= 0 and self.transactions[i].timestamp >= window_start:
                total_amount += self.transactions[i].amount
                i -= 1
            n = window_size - i - 1
            counts.append(n)
            if n == 0:
                amounts.append(0.0)
            else:
                amounts.append(total_amount / n)
        return amounts, counts

    def compute_counts_by_window2(self, t: datetime):
        """
        :param timestamps: Timestamps sorted in ascending order
        :param windows: Time windows sorted in ascending order
        :param t: Current time
        """
        n_timestamps = len(self.transactions)
        n_windows = len(self.aggregation_windows)
        if n_timestamps == 0:
            return [0] * n_windows
        res = list[int]()
        # -2 since the last transaction it is always included in all windows
        window_size = 1
        ti = n_timestamps - 1 - window_size
        wi = 0
        window_start = t - self.aggregation_windows[wi]
        while wi < n_windows - 1 and ti >= 0:  # We ignore the largest window (the last)
            if self.transactions[ti].timestamp < window_start:  # If the timestamp is out of the window
                res.append(window_size)
                wi += 1
                window_start = t - self.aggregation_windows[wi]
            else:
                ti -= 1
                window_size += 1
        for _ in range(wi, n_windows):
            # If there are windows that have not been considered,
            # then all the transactions are to be counted in each of them.
            res.append(n_timestamps)
        return res
