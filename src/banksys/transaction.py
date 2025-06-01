from dataclasses import Field, dataclass
from datetime import datetime
import polars as pl
import numpy as np

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

    @classmethod
    def field_names(cls) -> list[str]:
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        return [field.name for field in fields]
