from dataclasses import Field, dataclass
from datetime import datetime, timedelta
from typing import Literal, Sequence
import polars as pl
import inspect
from humanize import naturaldelta
from utils import fields2schema
from exceptions import InsufficientFundsError
from .transaction import Transaction
from .trx_window import TransactionWindow


# In case there is an equality in the priority queue, it compares
# the payers. Therefore, we want the order to be defined.
@dataclass(order=True)
class Payer:
    id: int
    x: float
    y: float
    balance: float

    def __init__(self, id: int, x: float, y: float, balance: float, agg_windows: Sequence[timedelta]):
        self.id = int(id)
        self.x = int(x)
        self.y = int(y)
        self.balance = balance
        self._window = TransactionWindow(agg_windows)
        self._count_feature_names = [Payer.colname("count", window) for window in self._window.aggregation_windows]
        self._avg_feature_names = [Payer.colname("avg", window) for window in self._window.aggregation_windows]

    def add(self, trx: Transaction, update_balance: bool):
        assert trx.predicted_label is not None
        if trx.fraud_is_detected:
            return
        if update_balance:
            if trx.amount > self.balance:
                raise InsufficientFundsError(trx)
            self.balance -= trx.amount
        self._window.add(trx)

    def notify_detected_fraud(self, trx: Transaction):
        pass

    def __hash__(self) -> int:
        return self.id

    @staticmethod
    def from_df(df: pl.DataFrame, agg_windows: Sequence[timedelta]):
        return [Payer(agg_windows=agg_windows, **kwargs) for kwargs in df.iter_rows(named=True)]

    @classmethod
    def field_names(cls):
        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]

    @classmethod
    def schema(cls) -> dict:
        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return fields2schema(fields)

    def compute_features(self, t: datetime) -> dict[str, float]:
        """
        Compute the number of transactions and their average amount within each aggregation window.
        """
        self._window.update(t)
        if self._window.is_empty:
            return {
                **{f: 0 for f in self._count_feature_names},
                **{f: 0.0 for f in self._avg_feature_names},
            }
        amounts, counts = self._window.compute_avg_amount_and_count_by_window(t)
        results = dict[str, float]()
        for i, (count, avg_amount) in enumerate(zip(counts, amounts)):
            results[self._count_feature_names[i]] = count
            results[self._avg_feature_names[i]] = avg_amount
        return results

    @staticmethod
    def colname(data: Literal["count", "avg"], window: timedelta) -> str:
        prefix = "[PAYER] TRX COUNT " if data == "count" else "[PAYER] TRX AVG "
        return f"{prefix}{naturaldelta(window)}"
