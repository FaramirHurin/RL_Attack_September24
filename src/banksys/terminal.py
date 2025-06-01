from dataclasses import Field, dataclass

import polars as pl

from .transaction import Transaction
from .trx_window import TransactionWindow


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

    @staticmethod
    def from_df(df: pl.DataFrame):
        return [Terminal(**kwargs) for kwargs in df.iter_rows(named=True)]

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]
