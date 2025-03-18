import numpy as np
from dataclasses import dataclass, astuple
from .card_info import CardInfo
from .terminal import Terminal


@dataclass
class Transaction:
    amount: float
    timestamp: float
    terminal: Terminal
    is_online: bool
    card: CardInfo

    def to_list(self):
        return list(astuple(self))

    def to_numpy(self):
        return np.array(astuple(self), dtype=np.float32)

    @property
    def terminal_x(self):
        return self.terminal.x

    @property
    def terminal_y(self):
        return self.terminal.y

    @property
    def customer_x(self):
        return self.card.customer_x

    @property
    def customer_y(self):
        return self.card.customer_y
