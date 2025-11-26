import random
from dataclasses import astuple, dataclass
from datetime import timedelta

import numpy as np


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_hours: float
    is_credit: bool

    def __init__(
        self,
        amount: float,
        terminal_x: float,
        terminal_y: float,
        is_online: bool,
        delay_hours: float,
        is_credit: bool,
    ):
        self.amount = max(0.01, min(100_000, amount))
        self.terminal_x = max(0, min(200, terminal_x))
        self.terminal_y = max(0, min(200, terminal_y))
        self.is_online = is_online
        self.is_credit = is_credit
        # Ensure delay_hours is positive and non-zero (at least 5 minutes)
        if delay_hours <= 0:
            delay_hours = (5 / 60) * random.random()
        self.delay_hours = delay_hours

    @property
    def timedelta(self):
        return timedelta(hours=self.delay_hours)

    @staticmethod
    def from_numpy(array: np.ndarray):
        """Convert a numpy array to an Action object."""
        amount, terminal_x, terminal_y, is_online, delay_hours, is_credit = array
        is_online = is_online > 0.5
        is_credit = is_credit > 0.5
        delay_hours = max(0, delay_hours)
        to_return = Action(
            amount=round(float(amount), 2),
            terminal_x=float(terminal_x),
            terminal_y=float(terminal_y),
            is_online=is_online,
            delay_hours=float(delay_hours),
            is_credit=is_credit,
        )
        return to_return

    def to_numpy(self):
        return np.array(astuple(self), dtype=np.float32)
