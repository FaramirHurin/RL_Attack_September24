from dataclasses import dataclass, astuple
import numpy as np


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_days: int
    delay_hours: float

    def __init__(self, amount: float, terminal_x: float, terminal_y: float, is_online: bool | float, delay_days: int, delay_hours: float):
        self.amount = amount
        self.terminal_x = terminal_x
        self.terminal_y = terminal_y
        self.is_online = bool(is_online)
        self.delay_days = delay_days
        self.delay_hours = delay_hours

    @staticmethod
    def from_numpy(array: np.ndarray):
        return Action(*array)

    def to_numpy(self):
        return np.ndarray(astuple(self), dtype=np.float32)
