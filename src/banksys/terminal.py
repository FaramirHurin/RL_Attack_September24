from dataclasses import Field, dataclass
from datetime import datetime, timedelta
from typing import Sequence
from .trx_window import TransactionWindow
import polars as pl

from utils import fields2schema
from .transaction import Transaction


PREFIX_N_TRX = "[Terminal] N_TRX "
PREFIX_RISK = "[Terminal] RISK "


@dataclass
class Terminal:
    id: int
    x: float
    y: float

    def __init__(self, id: int, x: float, y: float, aggregation_windows: Sequence[timedelta]):
        self.id = id
        self.x = x
        self.y = y
        self._genuine_window = TransactionWindow(aggregation_windows)
        self._fraud_window = TransactionWindow(aggregation_windows)
        self._trx_count_feature_names = [f"{PREFIX_N_TRX}{window}" for window in self._genuine_window.aggregation_windows]
        self._risk_feature_names = [f"{PREFIX_RISK}{window}" for window in self._genuine_window.aggregation_windows]

    def add(self, trx: Transaction):
        if trx.fraud_is_detected:
            self._fraud_window.add(trx)
        else:
            self._genuine_window.add(trx)

    def compute_features(self, t: datetime) -> dict[str, float]:
        """
        Count the number of transactions in the given time windows and compute the risk.
        Note: aggregation_windows must be sorted in ascending order.
        """
        self._genuine_window.update(t)
        n_genuine = self._genuine_window.compute_counts_by_window(t)
        self._fraud_window.update(t)
        n_frauds = self._fraud_window.compute_counts_by_window(t)
        features = dict[str, float]()
        for i, (trx_count, fraud_count) in enumerate(zip(n_genuine, n_frauds)):
            total = trx_count + fraud_count
            features[self._trx_count_feature_names[i]] = total
            if total == 0:
                features[self._risk_feature_names[i]] = 0.0
            else:
                features[self._risk_feature_names[i]] = fraud_count / total
        return features

    @staticmethod
    def from_df(df: pl.DataFrame, aggregetion_windows: Sequence[timedelta]):
        return [Terminal(aggregation_windows=aggregetion_windows, **kwargs) for kwargs in df.iter_rows(named=True)]

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
