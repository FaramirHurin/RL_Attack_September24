from dataclasses import dataclass
from datetime import datetime


N_MINUTES_IN_DAY = 24 * 60


@dataclass
class Transaction:
    FEATURE_NAMES = ["amount", "hour_ratio", "is_online"] + [f"day_of_week_{i}" for i in range(7)]

    amount: float
    timestamp: datetime
    terminal_id: int
    is_online: bool
    card_id: int

    @property
    def features(self):
        return [self.amount, self.hour_ratio, float(self.is_online), *self.one_hot_day_of_week]

    def time_ratio(self, start: datetime, end: datetime):
        """
        Calculate the ratio of the transaction time to the time interval.
        """
        total_seconds = (end - start).total_seconds()
        if total_seconds == 0:
            return 0
        return (self.timestamp - start).total_seconds() / total_seconds

    @property
    def day_of_week(self):
        return self.timestamp.weekday()

    @property
    def one_hot_day_of_week(self):
        one_hot = [0.0] * 7
        one_hot[self.day_of_week] = 1.0
        return one_hot

    @property
    def hour_ratio(self):
        return (self.timestamp.hour * 60 + self.timestamp.minute) / N_MINUTES_IN_DAY
