from dataclasses import Field, dataclass
from typing import Optional
import polars as pl
from datetime import datetime
from .transaction import Transaction
from .trx_window import TransactionWindow


# In case there is an equality in the priority queue, it compares
# the cards. Therefore, we want the order to be defined.
@dataclass(order=True)
class Card:
    id: int
    is_credit: bool
    x: float
    y: float
    balance: float
    current_time: Optional[datetime] = None
    """Transactions, ordered by timestamp"""

    def __init__(self, id: int, x: float, y: float, balance: float, is_credit: bool = False):
        self.id = int(id)
        self.is_credit = bool(is_credit)
        self.x = int(x)
        self.y = int(y)
        self.balance = balance
        self.transactions = TransactionWindow()
        self.current_time = None
        super().__init__()
        self.attempted_attacks = 0

    def add(self, transaction: Transaction):
        self.transactions.add(transaction)

    def __hash__(self) -> int:
        return self.id

    @staticmethod
    def from_df(df: pl.DataFrame):
        return [Card(**kwargs) for kwargs in df.iter_rows(named=True)]

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]

    def set_current_time(self, current_time: datetime):
        """
        Set the current time for the card.
        """
        self.current_time = current_time

    def remove_money(self, amount: float):
        """
        Remove money from the card's balance.
        """
        if amount > self.balance:
            raise ValueError(f"Not enough balance to remove {amount}. Current balance: {self.balance}")
        self.balance -= amount
