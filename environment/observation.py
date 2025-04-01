from dataclasses import dataclass
from datetime import datetime


@dataclass
class Observation:
    timestamp: datetime
    card_id: int

    def todo(self, remaining_time):
        pass
