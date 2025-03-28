from dataclasses import dataclass
import numpy as np


@dataclass
class Terminal:
    id: int
    x: float
    y: float

    def as_numpy(self):
        return np.array([self.x, self.y], dtype=np.float32)
