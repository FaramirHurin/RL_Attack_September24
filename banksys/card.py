from .transaction import Transaction
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class Card:
    id: int
    is_credit: bool
    customer_x: float
    customer_y: float
    transactions: list[Transaction]
    days_aggregation: tuple[int, ...]

    def __init__(self, id: int, is_credit: bool, customer_x: float, customer_y: float, days_aggregation: tuple[int, ...] = (1, 7, 30)):
        self.id = id
        self.is_credit = is_credit
        self.customer_x = customer_x
        self.customer_y = customer_y
        self.transactions = []
        self.days_aggregation = days_aggregation

    def to_list(self):
        return [self.is_credit, self.customer_x, self.customer_y]

    def to_numpy(self):
        return np.array(self.to_list(), dtype=np.float32)

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

    def features(self, current_time: datetime):
        """
        # TODO: compute the aggregated features for the card from the past transactions"
        """
        aggregated_features = [0.0 for _ in self.days_aggregation]
        return [float(self.is_credit), self.customer_x, self.customer_y] + aggregated_features

    @property
    def feature_names(self):
        return ["is_credit", "customer_x", "customer_y"] + [f"card_agg_{days}" for days in self.days_aggregation]
