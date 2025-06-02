from dataclasses import Field, dataclass
from typing import Sequence

import polars as pl

from datetime import datetime, timedelta
from .transaction import Transaction
from .trx_window import TransactionWindow


COLUMN_PREFIX_COUNT = "term_trx_count_"
COLUMN_PREFIX_AMOUNT = "term_trx_amount_"


@dataclass
class Terminal:
    id: int
    x: float
    y: float

    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.transactions = TransactionWindow()

    def add(self, transaction: Transaction):
        self.transactions.add(transaction)

    def get_features(self, agg: Sequence[timedelta], timestamp: datetime):
        return self.transactions.count_and_risk(agg, timestamp)

    @staticmethod
    def from_df(df: pl.DataFrame):
        return [Terminal(**kwargs) for kwargs in df.iter_rows(named=True)]

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]
