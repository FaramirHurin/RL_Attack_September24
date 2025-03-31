from dataclasses import dataclass
import numpy as np


@dataclass
class Action:
    amount: float
    terminal_x: float
    terminal_y: float
    is_online: bool
    delay_days: int
    delay_hours: float

    def to_classification_numpy(self):
        return np.array([self.is_online, self.amount], dtype=np.float32)
