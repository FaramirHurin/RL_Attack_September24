from dataclasses import Field, dataclass
from datetime import datetime
from enum import Enum
import polars as pl
import numpy as np

N_MINUTES_IN_DAY = 24 * 60


class DeclinedStatus(Enum):
    CUSTOMER_BLOCKED = "customer_blocked"
    DAILY_RULES_DETECTED = "daily_rules_detected"
    WEEKLY_RULES_DETECTED = "weekly_rules_detected"
    MONTHLY_RULES_DETECTED = "monthly_rules_detected"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ML_DETECTED = "ml_detected"
    STATISTICAL_DETECTED = "statistical_detected"


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
        if predicted_label is not None:
            predicted_label = bool(predicted_label)
        self.predicted_label = predicted_label

    def as_df(self, with_label: bool = False, with_predicted_label: bool = False) -> pl.DataFrame:
        """
        Convert the transaction to a Polars DataFrame.
        """
        data = {key: [value] for key, value in self.__dict__.items()}
        if not with_label:
            data.pop("is_fraud", None)
        if not with_predicted_label:
            data.pop("predicted_label", None)
        return pl.DataFrame(data)

    @property
    def features(self):
        return np.array(
            [
                self.amount,
                self.hour_ratio,
                float(self.is_online),
                float(self.terminal_id),
                float(self.card_id),
                *self.one_hot_day_of_week,
                *self.timestamp_features,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names():
        return [
            "amount",
            "hour_ratio",
            "is_online",
            "terminal_id",
            "card_id",
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
            "day",
            "month",
            "year",
            "hour",
            "minute",
            "second",
        ]

    @property
    def timestamp_features(self):
        return [
            self.timestamp.day,
            self.timestamp.month,
            self.timestamp.year,
            self.timestamp.hour,
            self.timestamp.minute,
            self.timestamp.second,
        ]

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
    def from_features(
        is_fraud: bool,
        *,
        amount: np.float32,
        year: np.float32,
        month: np.float32,
        day: np.float32,
        hour: np.float32,
        minute: np.float32,
        second: np.float32,
        terminal_id: np.float32,
        card_id: np.float32,
        is_online: bool,
        **_,
    ) -> "Transaction":
        return Transaction(
            amount=float(amount),
            timestamp=datetime(
                int(year),
                int(month),
                int(day),
                int(hour),
                int(minute),
                int(second),
            ),
            terminal_id=int(terminal_id),
            card_id=int(card_id),
            is_online=is_online,
            is_fraud=is_fraud,
        )

    @staticmethod
    def from_df(df: pl.DataFrame):
        transactions = list[Transaction]()
        for _, date, payer_id, _, is_remote, amount, payee_id, _, _, date, _, is_fraud, _ in df.iter_rows():
            transactions.append(
                Transaction(
                    amount=amount,
                    timestamp=date,
                    terminal_id=payee_id,
                    card_id=payer_id,
                    is_online=is_remote,
                    is_fraud=is_fraud,
                    predicted_label=None,
                )
            )
        return transactions

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]
