from dataclasses import dataclass
from datetime import datetime
import pandas as pd

N_MINUTES_IN_DAY = 24 * 60


@dataclass
class Transaction:
    amount: float
    timestamp: datetime
    terminal_id: int
    is_online: bool
    card_id: int
    is_fraud: bool
    """Whether the transaction actually is a fraud or not."""
    predicted_label: bool | None
    """Whether the transaction has been classified as a fraud or not. `None` if not classified yet."""

    def __init__(
        self,
        amount: float,
        timestamp: datetime,
        terminal_id: int,
        card_id: int,
        is_online: bool,
        is_fraud: bool,
        predicted_label: bool | None = None,
    ):
        self.amount = amount
        self.timestamp = timestamp
        self.terminal_id = terminal_id
        self.is_online = is_online
        self.card_id = card_id
        self.is_fraud = is_fraud
        self.predicted_label = predicted_label

    @property
    def features(self):
        return [self.amount, self.hour_ratio, float(self.is_online), *self.one_hot_day_of_week]

    def time_ratio(self, start: datetime, end: datetime):
        """
        Calculate the ratio of the transaction time to the time interval.
        """
        total_seconds = (end - start).total_seconds()
        if total_seconds == 0:
            return 0
        return (self.timestamp - start).total_seconds() / total_seconds

    @property
    def day_of_week(self):
        return self.timestamp.weekday()

    @property
    def one_hot_day_of_week(self):
        one_hot = [0.0] * 7
        one_hot[self.day_of_week] = 1.0
        return one_hot

    @property
    def hour_ratio(self):
        return (self.timestamp.hour * 60 + self.timestamp.minute) / N_MINUTES_IN_DAY

    def add_coordinates(self, payee_x, payee_y):
        """
        Add the x and y coordinates of the payee to the transaction.
        """
        self.payee_x = payee_x
        self.payee_y = payee_y
        return self

    @staticmethod
    def from_row(row: pd.Series, t_0: datetime):
        return Transaction(
            amount=row["amount"],
            timestamp=t_0,
            terminal_id=row["terminal_id"],
            card_id=row["card_id"],
            is_online=row["is_online"],
            is_fraud=row["is_fraud"],
        )
