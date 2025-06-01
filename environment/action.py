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
    # delay_days: int
    delay_hours: float

    def __init__(
        self,
        amount: float,
        terminal_x: float,
        terminal_y: float,
        is_online: bool,
        # delay_days: int,
        delay_hours: float,
    ):
        self.amount = max(0.0, min(100_000, amount))
        self.terminal_x = max(0, min(200,  terminal_x))
        self.terminal_y = max(0, min(200, terminal_y))
        self.is_online = is_online
        self.delay_hours =  max(0,  delay_hours)

        #self.delay_days = np.round(delay_hours / 24)

    @property
    def timedelta(self):
        return timedelta( hours=self.delay_hours) #days=self.delay_days,

    @staticmethod
    def from_numpy(array: np.ndarray):
        """Convert a numpy array to an Action object."""
        is_online, amount, terminal_x, terminal_y, delay_hours  = array.flatten() #, delay_days
        """if delay_hours > 1:
            print(f"Delay hours is {delay_hours}")
        else:
            print(f"Delay hours is {delay_hours} (not too high)")
        """
        is_online = is_online > 0.5
        delay_hours = max(0, delay_hours)
        #delay_hours = delay_hours + 2
        #TODO Remove this hack
        to_return = Action(
            amount=np.round(float(amount), 2),
            terminal_x=float(terminal_x),
            terminal_y=float(terminal_y),
            is_online=bool(is_online),
            # delay_days=int(delay_days),
            delay_hours=float(delay_hours) ,
        )

        return to_return

    def to_numpy(self):
        return np.array(astuple(self), dtype=np.float32)
