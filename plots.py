from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, Optional
import logging

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
    t_start: datetime
    t_end: datetime
    # terminal_ids: list[int]
    n_transactions: int
    card_id: int
    amount_stolen: float
    raw_data: dict

    @staticmethod
    def from_json(data: dict[str, Any]):
        metrics = data["metrics"]
        t_start = datetime.fromisoformat(metrics["t_start"])
        try:
            t_end = datetime.fromisoformat(metrics["t_end"])
        except KeyError:
            t_end = t_start
        # terminal_ids = metrics["terminals"]
        card_id = metrics["card_id"]
        n_transactions = metrics["episode_len"]
        amount_stolen = metrics["score-0"]
        return LogItem(t_start, t_end, n_transactions, card_id, amount_stolen, data)

    @cached_property
    def episode(self):
        for key, value in self.raw_data.items():
            if isinstance(value, list):
                self.raw_data[key] = [np.array(items) for items in value]
        return Episode(**self.raw_data)

    @cached_property
    def transactions(self):
        res = list[Transaction]()
        t = self.t_start
        for i, transition in enumerate(self.episode.transitions()):
            action = Action.from_numpy(transition.action)
            t = t + action.timedelta
            res.append(Transaction(action.amount, t, 0, self.card_id, action.is_online, True, predicted_label=False))
        res[-1].predicted_label = True
        return res


@dataclass
class Run:
    items: list[LogItem]
    params: dict[str, Any]

    @staticmethod
    def from_directory(directory: str, params: Optional[dict[str, Any]] = None):
        if params is None:
            params_path = os.path.join(directory, "params.json")
            with open(params_path) as f:
                params = orjson.loads(f.read())
        episodes_path = os.path.join(directory, "episodes.json")
        with open(episodes_path, "rb") as f:
            episodes = orjson.loads(f.read())
        logs = [LogItem.from_json(data) for data in episodes]
        assert params is not None
        return Run(logs, params)

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
    runs: dict[str, Run]
    params: dict[str, Any]

    @property
    def n_runs(self):
        return len(self.runs)

    def __init__(self, logs, params):
        self.runs = logs
        self.params = params

    @staticmethod
    def from_directory(directory: str):
        with open(os.path.join(directory, "params.json")) as f:
            params = orjson.loads(f.read())
        results = dict[str, Run]()
        for entry in os.listdir(directory):
            if not entry.startswith("seed-"):
                continue
            run_dir = os.path.join(directory, entry)
            try:
                results[entry] = Run.from_directory(run_dir, params)
            except FileNotFoundError:
                pass
        logging.debug(f"Loaded {len(results)} logs from {directory}")
        return Experiment(results, params)

    @cached_property
    def n_transactions_over_time(self):
        return np.array([e.n_transactions_over_time for e in self.runs.values()])

    @cached_property
    def amounts_over_time(self):
        return np.array([run.amount_over_time for run in self.runs.values()])

    @cached_property
    def total_amount(self):
        return sum(run.total_amount for run in self.runs.values())

    def __iter__(self):
        return iter(self.runs.values())


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
