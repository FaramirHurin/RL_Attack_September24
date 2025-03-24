from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .transaction import Transaction


@dataclass
class Terminal:
    x: float
    y: float
    transactions: list[Transaction]

    def perform_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    def compute_aggregated_features(self) -> npt.NDArray[np.float32]:
        raise NotImplementedError()
