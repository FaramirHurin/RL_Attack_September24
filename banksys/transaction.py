import numpy as np
from dataclasses import dataclass, astuple
from .card_info import CardInfo
from .terminal import Terminal
from environment import StepData


@dataclass
class Transaction:
    amount: float
    timestamp: float
    terminal: Terminal
    is_online: bool
    card: CardInfo
    day: int
    hour: int


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

    @staticmethod
    def from_step(step: StepData, terminal: Terminal):
        return Transaction(
            amount=step.amount,
            timestamp=step.timestamp,
            terminal=terminal,
            is_online=step.action.is_online,
            card=CardInfo.from_array(step.card_id),
        )
