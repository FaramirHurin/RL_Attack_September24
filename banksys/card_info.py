from dataclasses import dataclass
import numpy as np


@dataclass
class CardInfo:
    id: int
    is_credit: bool
    customer_x: float
    customer_y: float

    def to_list(self):
        return [self.is_credit, self.customer_x, self.customer_y]

    def to_numpy(self):
        return np.array(self.to_list(), dtype=np.float32)
