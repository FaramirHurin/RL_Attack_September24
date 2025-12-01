from dataclasses import asdict, astuple, dataclass, fields
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_hours: float

    def __init__(
        self,
        amount: float,
        terminal_x: float,
        terminal_y: float,
        is_online: bool,
        delay_hours: float,
    ):
        self.amount = max(0.01, amount)
        self.terminal_x = max(0, min(200, terminal_x))
        self.terminal_y = max(0, min(200, terminal_y))
        self.is_online = is_online
        # Ensure delay_hours is positive
        self.delay_hours = abs(delay_hours)

    @property
    def timedelta(self):
        return timedelta(hours=self.delay_hours)

    @staticmethod
    def from_numpy(array: npt.NDArray[np.float32]):
        """Convert a numpy array to an Action object."""
        amount = array[AMOUNT_INDEX]
        terminal_x = array[TERMINAL_X_INDEX]
        terminal_y = array[TERMINAL_Y_INDEX]
        is_online = array[IS_ONLINE_INDEX] > 0.5
        delay_hours = array[DELAY_HOURS_INDEX]
        to_return = Action(
            amount=round(float(amount), 2),
            terminal_x=float(terminal_x),
            terminal_y=float(terminal_y),
            is_online=bool(is_online),
            delay_hours=float(delay_hours),
        )
        return to_return

    def to_numpy(self):
        return np.array(astuple(self), dtype=np.float32)

    def denormalized(self, scale_amount: float, scale_x: float = 200.0, scale_y: float = 200.0):
        return Action(
            amount=self.amount * scale_amount,
            terminal_x=self.terminal_x * scale_x,
            terminal_y=self.terminal_y * scale_y,
            is_online=self.is_online,
            delay_hours=self.delay_hours,
            # is_credit=self.is_credit,
        )

    def as_dict(self):
        return asdict(self)


FIELDS_INDEX = {f.name: i for i, f in enumerate(fields(Action))}

AMOUNT_INDEX = FIELDS_INDEX["amount"]
TERMINAL_X_INDEX = FIELDS_INDEX["terminal_x"]
TERMINAL_Y_INDEX = FIELDS_INDEX["terminal_y"]
IS_ONLINE_INDEX = FIELDS_INDEX["is_online"]
DELAY_HOURS_INDEX = FIELDS_INDEX["delay_hours"]
