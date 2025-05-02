from dataclasses import dataclass, astuple
import numpy as np
from datetime import timedelta
import random


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_days: int
    delay_hours: float

    def __init__(
        self,
        amount: float,
        terminal_x: float,
        terminal_y: float,
        is_online: bool,
        delay_days: int,
        delay_hours: float,
    ):
        self.amount = max(0.01, min(100_000, amount))
        self.terminal_x = max(0, min(200, terminal_x))
        self.terminal_y = max(0, min(200, terminal_y))
        self.is_online = is_online
        self.delay_days = max(0, delay_days)
        self.delay_hours = max(0, delay_hours)
        if self.delay_days == 0 and self.delay_hours == 0:
            # Randomly wait for at most 5 minutes
            self.delay_hours = (5 / 60) * random.random()

    @property
    def timedelta(self):
        return timedelta(days=self.delay_days, hours=self.delay_hours)

    @staticmethod
    def from_numpy(array: np.ndarray):
        """Convert a numpy array to an Action object."""
        amount, terminal_x, terminal_y, online_score, delay_days, delay_hours = array
        is_online = online_score > 0.5
        return Action(
            amount=float(amount),
            terminal_x=float(terminal_x),
            terminal_y=float(terminal_y),
            is_online=bool(is_online),
            delay_days=int(delay_days),
            delay_hours=float(delay_hours),
        )

    def to_numpy(self):
        return np.array(astuple(self), dtype=np.float32)
