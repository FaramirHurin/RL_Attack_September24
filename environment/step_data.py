from dataclasses import dataclass
from datetime import datetime
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

    def to_stamp(self) -> datetime:
        # Uses timestamp to convert it to a timestamp
        raise NotImplementedError("This method is not implemented yet.")
        # return datetime.fromtimestamp(self.timestamp)
