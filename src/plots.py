from dataclasses import dataclass, replace
from datetime import datetime
from functools import cached_property
from typing import Any, Optional
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import orjson
from parameters import Parameters, serialize_unknown
from marlenv import Episode


from banksys import Transaction
from environment import Action


@dataclass
class LogItem:
    t_start: datetime
    t_end: datetime
    n_transactions: int
    card_id: int
    amount_stolen: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LogItem":
        t_start = d["t_start"]
        if isinstance(t_start, str):
            t_start = datetime.fromisoformat(t_start)
        t_end = d.get("t_end", d["t_start"])
        if isinstance(t_end, str):
            t_end = datetime.fromisoformat(t_end)
        return LogItem(
            t_start=t_start,
            t_end=t_end,
            n_transactions=d["episode_len"],
            card_id=d["card_id"],
            amount_stolen=d["score-0"],
        )


@dataclass
class Run:
    rundir: str
    params: Parameters
    items: list[LogItem]
    episodes: Optional[list[Episode]] = None

    @staticmethod
    def create(params: Parameters, episodes: list[Episode]):
        """
        Create a new run and saves it to the disk.
        """
        rundir = params.logdir
        os.makedirs(rundir, exist_ok=True)
        params_path = os.path.join(rundir, "params.json")
        with open(params_path, "wb") as f:
            f.write(orjson.dumps(params, default=serialize_unknown))
        episodes_path = os.path.join(rundir, "episodes.json")
        with open(episodes_path, "wb") as f:
            f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))
        metrics_path = os.path.join(rundir, "metrics.json")
        with open(metrics_path, "wb") as f:
            metrics = [e.metrics for e in episodes]
            f.write(orjson.dumps(metrics, option=orjson.OPT_SERIALIZE_NUMPY))
        return Run(rundir, params, [LogItem.from_dict(m) for m in metrics], episodes)

    @staticmethod
    def load_parameters(rundir: str):
        params_path = os.path.join(rundir, "params.json")
        try:
            return Parameters.load(params_path)
        except FileNotFoundError:
            pass
        parent_dir = os.path.dirname(rundir)
        params_path = os.path.join(parent_dir, "params.json")
        return Parameters.load(params_path)

    @staticmethod
    def load(rundir: str, params: Optional[Parameters] = None):
        if params is None:
            params = Run.load_parameters(rundir)
        try:
            metrics_path = os.path.join(rundir, "metrics.json")
            with open(metrics_path, "rb") as f:
                metrics_dict: list[dict] = orjson.loads(f.read())
        except FileNotFoundError:
            episode_path = os.path.join(rundir, "episodes.json")
            with open(episode_path, "rb") as f:
                json_data: list[dict] = orjson.loads(f.read())
            metrics_dict = []
            for json_episode in json_data:
                metrics_dict.append(
                    {
                        "t_start": json_episode["metrics"]["t_start"],
                        "t_end": json_episode.get("t_end", json_episode["metrics"]["t_start"]),
                        "episode_len": len(json_episode["actions"]),
                        "card_id": json_episode["metrics"]["card_id"],
                        "score-0": json_episode["metrics"]["score-0"],
                    }
                )
        items = [LogItem.from_dict(m) for m in metrics_dict]
        return Run(rundir, params, items)

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

    def transactions(self, episode_num: int):
        """
        Returns the transactions for a specific episode.
        """
        if episode_num < 0 or episode_num >= len(self.items):
            raise IndexError("Episode number out of range")
        if self.episodes is None:
            # Load episodes from disk if not already loaded
            filename = os.path.join(self.rundir, "episodes.json")
            with open(filename, "rb") as f:
                json_data: list[dict] = orjson.loads(f.read())
            for json_episode in json_data:
                for key, value in json_episode.items():
                    if isinstance(value, list):
                        json_episode[key] = [np.array(v) for v in value]
            self.episodes = [Episode(**e) for e in json_data]

        episode = self.episodes[episode_num]
        res = list[Transaction]()
        log_item = self.items[episode_num]
        t = log_item.t_start
        for np_action in episode.actions:
            action = Action.from_numpy(np_action)
            t = t + action.timedelta
            res.append(
                Transaction(
                    amount=action.amount,
                    timestamp=t,
                    terminal_id=0,
                    card_id=log_item.card_id,
                    is_online=action.is_online,
                    is_fraud=True,
                    predicted_label=False,
                )
            )
        res[-1].predicted_label = True
        return res


class Experiment:
    def __init__(self, logdir: str, params: Parameters, runs: Optional[dict[str, Run]] = None):
        self.logdir = logdir
        self.params = params
        self._runs = runs

    @property
    def runs(self):
        if self._runs is None:
            self._runs = self.load_runs()
        return self._runs

    def load_runs(self):
        results = dict[str, Run]()
        for entry in os.listdir(self.logdir):
            if not entry.startswith("seed-"):
                continue
            run_dir = os.path.join(self.logdir, entry)
            try:
                results[entry] = Run.load(run_dir, self.params)
            except FileNotFoundError:
                pass
        logging.debug(f"Loaded {len(results)} logs from {self.logdir}")
        return results

    def repeat(self, n: int, initial_seed: Optional[int] = None):
        """
        Repeat the experiment n times.
        """
        if initial_seed is None:
            initial_seed = self.params.seed_value + self.n_runs
        for seed in range(initial_seed, initial_seed + n):
            logdir = os.path.join(self.logdir, f"seed-{seed}")
            # os.makedirs(logdir, exist_ok=True)
            yield replace(self.params, seed_value=seed, save=False, logdir=logdir)

    @property
    def n_runs(self):
        return len(self.runs)

    @staticmethod
    def create(params: Parameters):
        logdir = params.logdir
        os.makedirs(logdir, exist_ok=True)
        params_path = os.path.join(logdir, "params.json")
        with open(params_path, "wb") as f:
            f.write(orjson.dumps(params, default=serialize_unknown, option=orjson.OPT_SERIALIZE_NUMPY))
        return Experiment(logdir, params, {})

    @staticmethod
    def load(directory: str):
        params = Parameters.load(os.path.join(directory, "params.json"))
        return Experiment(directory, params, None)

    def add(self, episodes: list[Episode], seed: int):
        path = os.path.join(self.logdir, f"seed-{seed}")
        params = replace(self.params, seed_value=seed, logdir=path)
        return Run.create(params, episodes)

    @cached_property
    def n_transactions_over_time(self):
        return np.array([e.n_transactions_over_time for e in self.runs.values()])

    @cached_property
    def amounts_over_time(self):
        amounts = []
        maxlen = 0
        for run in self.runs.values():
            maxlen = max(maxlen, len(run.amount_over_time))
            amounts.append(run.amount_over_time)
        # Pad the amounts to the same length
        for i in range(len(amounts)):
            amounts[i] += [float("NaN")] * (maxlen - len(amounts[i]))
        return np.array(amounts)

    def get_actions(self):
        """
        Returns a list of all actions taken in all runs.
        """
        actions = []
        for run in self.runs.values():
            episodes = run.episodes
            assert episodes is not None
            for episode in episodes:
                print(episode)
                actions.extend(episode.actions)
        return actions

    @cached_property
    def mean_std_amounts_over_time(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean and standard deviation of amounts over time for all runs.
        """
        amounts = self.amounts_over_time
        return np.nanmean(amounts, axis=0), np.nanstd(amounts, axis=0)

    @cached_property
    def total_amounts(self):
        """
        The total amount retrieved per run.
        """
        return np.array([run.total_amount for run in self.runs.values()])

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
