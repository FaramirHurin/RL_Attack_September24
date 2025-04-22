from dataclasses import dataclass, astuple
import numpy as np
from datetime import timedelta


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_days: int
    delay_hours: float

    def __init__(self, amount: float, terminal_x: float, terminal_y: float, is_online: bool, delay_days: int, delay_hours: float):
        self.amount = amount
        self.terminal_x = terminal_x
        self.terminal_y = terminal_y
        self.is_online = is_online
        self.delay_days = delay_days
        self.delay_hours = delay_hours

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
