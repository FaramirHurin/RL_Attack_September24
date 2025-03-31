from dataclasses import dataclass

from .action import Action


@dataclass
class StepData:
    action: Action
    timestamp: float
    card_id: int

    @property
    def amount(self):
        return self.action.amount

    @property
    def terminal_x(self):
        return self.action.terminal_x

    @property
    def terminal_y(self):
        return self.action.terminal_y
