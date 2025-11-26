from dataclasses import Field, dataclass
import polars as pl
import inspect
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
    """Transactions, ordered by timestamp"""

    def __init__(self, id: int, x: float, y: float, balance: float):
        self.id = int(id)
        self.x = int(x)
        self.y = int(y)
        self.balance = balance
        self.transactions = TransactionWindow()
        super().__init__()

    def add(self, transaction: Transaction, update_balance: bool):
        if update_balance:
            if transaction.amount > self.balance:
                raise InsufficientFundsError(transaction)
            self.balance -= transaction.amount
        self.transactions.add(transaction)

    def __hash__(self) -> int:
        return self.id

    @staticmethod
    def from_df(df: pl.DataFrame):
        return [Payer(**kwargs) for kwargs in df.iter_rows(named=True)]

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
