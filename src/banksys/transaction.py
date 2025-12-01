from dataclasses import Field, dataclass, asdict
from datetime import datetime
import polars as pl
from utils import fields2schema


@dataclass
class Transaction:
    amount: float
    timestamp: datetime
    terminal_id: int
    is_online: bool
    payer_id: int
    is_fraud: bool
    """Whether the transaction actually is a fraud or not."""
    is_credit: bool
    """Whether the transaction was made using a credit card (as opposed to a debit card)."""
    predicted_label: bool | None
    """Whether the transaction has been classified as a fraud or not. `None` if not classified yet."""

    def __init__(
        self,
        amount: float,
        timestamp: datetime,
        terminal_id: int,
        payer_id: int,
        is_online: bool,
        is_fraud: bool,
        is_credit: bool = False,
        predicted_label: bool | None = None,
    ):
        self.amount = amount
        self.timestamp = timestamp
        self.terminal_id = terminal_id
        self.is_online = is_online
        self.payer_id = payer_id
        self.is_credit = is_credit
        self.is_fraud = is_fraud
        if predicted_label is not None:
            predicted_label = bool(predicted_label)
        self.predicted_label = predicted_label

    @property
    def fraud_is_detected(self):
        """Return whether the predicted label indicates a fraud (i.e. `True`)."""
        if self.predicted_label is None:
            return False
        return self.predicted_label

    def as_df(self, with_label: bool = False, with_predicted_label: bool = False) -> pl.DataFrame:
        """
        Convert the transaction to a Polars DataFrame.
        """
        data = asdict(self)
        if not with_label:
            data.pop("is_fraud", None)
        if not with_predicted_label:
            data.pop("predicted_label", None)
        return pl.DataFrame(data)

    @classmethod
    def field_names(cls, with_predicted_label: bool = True):
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        names = [field.name for field in fields]
        if not with_predicted_label:
            names.remove("predicted_label")
        return names

    @classmethod
    def schema(cls, with_predicted_label: bool = True):
        import inspect

        members = inspect.getmembers(cls)
        fields = list[Field](dict(members)["__dataclass_fields__"].values())
        schema = fields2schema(fields)
        if not with_predicted_label:
            schema.pop("predicted_label", None)
        return schema
