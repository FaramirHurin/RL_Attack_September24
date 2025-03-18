from dataclasses import dataclass
from typing import Optional
from .transaction import Transaction
import numpy as np


@dataclass
class CardInfo:
    is_credit: bool
    customer_x: float
    customer_y: float
    transactions: list[Transaction]

    def to_list(self):
        return [self.is_credit, self.customer_x, self.customer_y]

    def to_numpy(self):
        return np.array(self.to_list(), dtype=np.float32)

    @staticmethod
    def from_array(array: np.ndarray | list, transactions: Optional[list[Transaction]] = None):
        is_credit, customer_x, customer_y = array
        if transactions is None:
            transactions = []
        return CardInfo(
            is_credit=bool(is_credit),
            customer_x=float(customer_x),
            customer_y=float(customer_y),
            transactions=transactions,
        )

    def compute_aggregated_features(self) -> np.ndarray:
        raise NotImplementedError()
