from dataclasses import Field, dataclass
from datetime import datetime, timedelta
from typing import Sequence

import polars as pl

from utils import fields2schema
from .transaction import Transaction


@dataclass
class Terminal:
    id: int
    x: float
    y: float

    def __init__(self, id: int, x: float, y: float, aggregation_windows: Sequence[timedelta]):
        self.id = id
        self.x = x
        self.y = y
        self.aggregation_windows = sorted(aggregation_windows)
        """Aggregation time windows sorted in ascending order."""
        self.max_window = aggregation_windows[-1]
        """The largest aggregation time window."""
        self.trx_timestamps = list[datetime]()
        """The timestamps of all transactions associated with this terminal."""
        self.frauds_timestamps = list[datetime]()
        """The timestamps of all detected frauds associated with this terminal."""

    def add(self, trx: Transaction):
        """
        Add the transaction to the terminal's records.
        """
        self.trx_timestamps.append(trx.timestamp)
        if trx.fraud_is_detected:
            self.frauds_timestamps.append(trx.timestamp)
        # Update the queue to remove transactions outside the max window
        window_start = trx.timestamp - self.max_window
        i = 0
        while i < len(self.trx_timestamps) and self.trx_timestamps[i] < window_start:
            i += 1
        if i > 0:
            self.trx_timestamps = self.trx_timestamps[i:]
        i = 0
        while i < len(self.frauds_timestamps) and self.frauds_timestamps[i] < window_start:
            i += 1
        if i > 0:
            self.frauds_timestamps = self.frauds_timestamps[i:]

    @staticmethod
    def _count_within_window(timestamps: list[datetime], windows: Sequence[timedelta], t: datetime):
        """
        Docstring for _count_within_window

        :param timestamps: Timestamps sorted in ascending order
        :param windows: Time windows sorted in ascending order
        :param t: Current time
        """
        res = list[int]()
        ti = len(timestamps) - 1
        wi = 0
        window = windows[wi]
        window_start = t - window
        while wi < len(windows) - 1:  # We ignore the largest window (the last)
            if timestamps[ti] < window_start:
                n_in_window = len(timestamps) - ti
                res.append(n_in_window)
                wi += 1
            ti -= 1
        res.append(len(timestamps))
        return res

    def compute_features(self) -> dict[str, float]:
        """
        Count the number of transactions in the given time windows and compute the risk.
        Note: aggregation_windows must be sorted in ascending order.
        """
        t = self.trx_timestamps[-1]
        n_transactions = self._count_within_window(self.trx_timestamps, self.aggregation_windows, t)
        n_frauds = self._count_within_window(self.frauds_timestamps, self.aggregation_windows, t)
        features = dict[str, float]()
        for window, trx_count, fraud_count in zip(self.aggregation_windows, n_transactions, n_frauds):
            features[f"[Terminal] N_TRX {window}"] = trx_count
            features[f"[Terminal] RISK {window}"] = fraud_count / trx_count
        return features

    @staticmethod
    def from_df(df: pl.DataFrame):
        return [Terminal(**kwargs) for kwargs in df.iter_rows(named=True)]

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]

    @classmethod
    def schema(cls) -> dict:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return fields2schema(fields)
