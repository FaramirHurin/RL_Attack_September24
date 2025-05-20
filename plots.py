from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from marlenv import Episode

from banksys import Transaction
from environment import Action


@dataclass
class LogItem:
    episode: Episode

    @cached_property
    def t_start(self):
        return datetime.fromisoformat(self.episode.metrics["t_start"])

    @cached_property
    def t_end(self):
        return datetime.fromisoformat(self.episode.metrics["t_end"])

    @cached_property
    def terminal_ids(self) -> list[int]:
        return self.episode.metrics["terminals"]

    @cached_property
    def card_id(self) -> int:
        return self.episode.metrics["card_id"]

    @cached_property
    def transactions(self):
        res = list[Transaction]()
        t = self.t_start
        for i, transition in enumerate(self.episode.transitions()):
            action = Action.from_numpy(transition.action)
            t = t + action.timedelta
            res.append(Transaction(action.amount, t, self.terminal_ids[i], self.card_id, action.is_online, False))
        res[-1].predicted_label = True
        return res


def plot_transactions(transactions: list[Transaction]):
    fig, ax = plt.subplots()
    ax.set_xlim([transactions[0].timestamp, transactions[-1].timestamp])  # type: ignore
    # Optional: format date axis
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d-%b"))

    # Set labels
    ax.set_title("Transactions over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")

    COLOURS = {}
    fraud_dates, fraud_amounts, fraud_colours = [], [], []
    genuine_dates, genuine_amounts, genuine_colours = [], [], []
    for t in transactions:
        if t.card_id not in COLOURS:
            COLOURS[t.card_id] = np.random.rand(3)
        if t.predicted_label:
            fraud_dates.append(t.timestamp)
            fraud_amounts.append(t.amount)
            fraud_colours.append(COLOURS[t.card_id])
        else:
            genuine_dates.append(t.timestamp)
            genuine_amounts.append(t.amount)
            genuine_colours.append(COLOURS[t.card_id])
    # Create a scatter plot
    ax.scatter(fraud_dates, fraud_amounts, c=fraud_colours, marker="x", s=50)
    ax.scatter(genuine_dates, genuine_amounts, c=genuine_colours, marker="o", s=50)
    fig.show()
