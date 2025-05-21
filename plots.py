from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import orjson
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

    @property
    def n_transactions(self):
        return len(self.terminal_ids)

    @cached_property
    def transactions(self):
        res = list[Transaction]()
        t = self.t_start
        for i, transition in enumerate(self.episode.transitions()):
            action = Action.from_numpy(transition.action)
            t = t + action.timedelta
            res.append(Transaction(action.amount, t, self.terminal_ids[i], self.card_id, action.is_online, True, predicted_label=False))
        res[-1].predicted_label = True
        return res

    @cached_property
    def amount_stolen(self):
        return self.episode.score[0]


@dataclass
class Logs:
    items: list[LogItem]

    @staticmethod
    def from_file(file_path: str):
        with open(file_path, "rb") as f:
            episodes = orjson.loads(f.read())
        logs = list[LogItem]()
        for e_dict in episodes[:4000]:
            assert isinstance(e_dict, dict)
            # e_dict.pop("score", None)
            for key, value in e_dict.items():
                if isinstance(value, list):
                    e_dict[key] = [np.array(items) for items in value]
            e_dict = Episode(**e_dict)
            logs.append(LogItem(e_dict))
        return Logs(logs)

    @cached_property
    def total_amount(self) -> float:
        return sum(e.amount_stolen for e in self.items)

    @cached_property
    def n_transactions_over_time(self):
        return [e.n_transactions for e in self.items]

    @cached_property
    def amount_over_time(self):
        return [e.amount_stolen for e in self.items]

    def __iter__(self):
        return iter(self.items)


@dataclass
class Experiment:
    logs: dict[str, Logs]

    @staticmethod
    def from_directory(directory: str):
        results = dict[str, Logs]()
        for entry in os.listdir(directory):
            if not entry.startswith("seed-"):
                continue
            episodes_path = os.path.join(directory, entry, "episodes.json")
            try:
                results[entry] = Logs.from_file(episodes_path)
            except FileNotFoundError:
                pass
        return Experiment(results)

    @cached_property
    def n_transactions_over_time(self):
        return np.array([e.n_transactions_over_time for e in self.logs.values()])

    @cached_property
    def amounts_over_time(self):
        return np.array([e.amount_over_time for e in self.logs.values()])

    def __iter__(self):
        return iter(self.logs.values())


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
