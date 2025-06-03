import logging
import os
import pickle
from functools import cached_property
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import polars as pl
from tqdm import tqdm

from .card import Card
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction

if TYPE_CHECKING:
    from parameters import ClassificationParameters


class Banksys:
    def __init__(
        self,
        transactions_df: pl.DataFrame,
        cards_df: pl.DataFrame,
        terminals_df: pl.DataFrame,
        aggregation_windows: Sequence[timedelta],
        clf_params: "ClassificationParameters",
        attackable_terminal_factor: float = 0.1,
        fp_rate=0.0,
        fn_rate=0.0,
    ):
        self.max_aggregation_duration = max(*aggregation_windows) if len(aggregation_windows) > 1 else aggregation_windows[0]
        self.current_time: datetime = transactions_df["timestamp"].min()  # type: ignore
        self.training_start = self.current_time + self.max_aggregation_duration
        self.attack_start = self.training_start + clf_params.training_duration
        self.attack_end: datetime = transactions_df["timestamp"].max()  # type: ignore
        assert self.attack_start < self.attack_end, f"Attack start ({self.attack_start}) must be before attack end ({self.attack_end})."

        self.clf = ClassificationSystem(clf_params)

        self._transactions_df = (
            transactions_df.sort("timestamp")  # Sort by timestamp
            .with_columns(self._approximate_labels(transactions_df, fp_rate=fp_rate, fn_rate=fn_rate))  # Add training "predicted_label"
            .with_columns(
                pl.when(pl.col("timestamp") > self.attack_start)  # Remove 'predicted_label' for the attack set.
                .then(None)
                .otherwise(pl.col("predicted_label"))
                .alias("predicted_label")
            )
        )
        self.trx_iterator = self._transactions_df.iter_rows(named=True)
        self.next_trx = Transaction(**next(self.trx_iterator))
        self.cards = sorted(Card.from_df(cards_df), key=lambda c: c.id)
        self.terminals = sorted(Terminal.from_df(terminals_df), key=lambda t: t.id)
        self.aggregation_windows = aggregation_windows
        self.attackable_terminals = random.sample(self.terminals, round(len(self.terminals) * attackable_terminal_factor))
        self.fit()

    def fit(self):
        """
        Fit the classification system and process the training transactions.

        Automatically called from the constructor.
        """
        logging.info("System warmup for training feature aggregation...")
        self.fast_forward(self.training_start)

        logging.info("Building classifier training features...")
        features = self.fast_forward(self.attack_start)
        train_x = pl.DataFrame(features)
        train_y = self.training_set["is_fraud"].to_numpy().astype(np.bool)
        self.clf.fit(pl.DataFrame(train_x), train_y)

    def fast_forward(self, until: datetime):
        """
        Fast forward the system to the given date, adding all the transactions to the
        system but without classifying them.
        """
        if until > self.attack_end:
            raise ValueError(f"Cannot forward to {until}, it is beyond the attack end date {self.attack_end}.")
        start = self.next_trx.timestamp
        stop = min(until, self.attack_end)
        n = self._transactions_df.filter(pl.col("timestamp").is_between(start, stop)).height
        pbar = tqdm(total=n, desc="Fast-forwarding transactions", unit="trx")
        features = list()
        while self.next_trx.timestamp < stop:
            features.append(self.make_transaction_features(self.next_trx))
            self.cards[self.next_trx.card_id].add(self.next_trx, update_balance=False)
            self.terminals[self.next_trx.terminal_id].add(self.next_trx)
            pbar.set_description(f"{self.next_trx.timestamp.date().isoformat()}")
            pbar.update()
            self.next_trx = Transaction(**next(self.trx_iterator))
        pbar.close()
        self.current_time = until
        return features

    def simulate_until(self, until: datetime):
        """
        Simulate the system until the given date, processing all transactions up to that date.
        A "predicted label" is assigned to each transaction via the classification system.
        """
        if until > self.attack_end:
            raise ValueError(f"Cannot forward to {until}, it is beyond the attack end date {self.attack_end}.")

        cards = set[int]()
        terms = set[int]()
        batch = list[Transaction]()
        features = list[pl.DataFrame]()
        while self.next_trx.timestamp < until:
            if self.next_trx.card_id in cards or self.next_trx.terminal_id in terms:
                features.append(self.process_transactions(batch, update_balance=False, real_label=True))
                cards.clear()
                terms.clear()
                batch.clear()
            cards.add(self.next_trx.card_id)
            terms.add(self.next_trx.terminal_id)
            batch.append(self.next_trx)
            self.next_trx = Transaction(**next(self.trx_iterator))
        if len(batch) > 0:
            features.append(self.process_transactions(batch, update_balance=False, real_label=True))
        self.current_time = until
        return features

    def process_transaction(self, trx: Transaction, update_balance: bool = True, real_label: Optional[bool] = False):
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        If `real_label` is True, it will use the real label from the transaction.
        """
        self.simulate_until(trx.timestamp)
        features = self.make_transaction_features(trx)
        if trx.predicted_label is None:
            if real_label == True:
                trx.predicted_label = trx.is_fraud
            else:
                label = self.clf.predict(pl.DataFrame(features))
                trx.predicted_label = label.item()

        if not real_label:
            try:
                self.seen_cards[trx.card_id] = \
                    self.seen_cards[trx.card_id] = self.seen_cards[trx.card_id] + 1 \
                    if trx.card_id in self.seen_cards.keys() else 1
            except:
                self.seen_cards = {trx.card_id: 1}
            if self.seen_cards[trx.card_id]  >= 9:
                debug =0

        self.terminals[trx.terminal_id].add(trx)
        self.cards[trx.card_id].add(trx, update_balance=update_balance)
        return features

    def process_transactions(self, transactions: list[Transaction], update_balance: bool, real_label=False):
        """
        Receives a list of chronological transactions and processes them, assigning a predicted label to each transaction.
        If `real_label` is True, it will use the real label from the transaction.
        """
        df = pl.DataFrame(self.make_transaction_features(trx) for trx in transactions)
        if real_label:
           labels = np.array([trx.is_fraud for trx in transactions])
        else:
            labels = self.clf.predict(df)
        for trx, label in zip(transactions, labels):
            trx.predicted_label = label
            self.terminals[trx.terminal_id].add(trx)
            self.cards[trx.card_id].add(trx, update_balance=update_balance)
        return df

    def make_transaction_features(self, trx: Transaction):
        weekday = [0.0] * 7
        weekday[trx.timestamp.weekday()] = 1.0
        features = {
            "hour": trx.timestamp.hour,
            "is_online": trx.is_online,
            "amount": trx.amount,
            **{day: val for day, val in zip("Mon Tue Wed Thu Fri Sat Sun".split(), weekday)},
            **self.cards[trx.terminal_id].transactions.count_and_mean(self.aggregation_windows, trx.timestamp),
            **self.terminals[trx.terminal_id].transactions.count_and_risk(self.aggregation_windows, trx.timestamp),
        }
        return features

    def _approximate_labels(self, trx: pl.DataFrame, fp_rate: float = 0.01, fn_rate: float = 0.01):
        assert 0 <= fp_rate <= 1.0 and 0 <= fn_rate <= 1.0, "Rates must be between 0 and 1"
        # Random values for conditional flipping
        trx = trx.with_columns(pl.Series("rand", np.random.rand(len(trx))))
        # Flip logic
        trx = trx.with_columns(
            pl.when((pl.col("is_fraud") == 1) & (pl.col("rand") < fn_rate))
            .then(0)
            .when((pl.col("is_fraud") == 0) & (pl.col("rand") < fp_rate))
            .then(1)
            .otherwise(pl.col("is_fraud"))
            .alias("predicted_label")
        )
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

        truth = trx["is_fraud"]
        pred = trx["predicted_label"]

        cm = confusion_matrix(truth, pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Accuracy: {accuracy_score(truth, pred):.4f}")
        logging.info(f"Recall: {recall_score(truth, pred):.4f}")
        logging.info(f"Precision: {precision_score(truth, pred):.4f}")
        logging.info(f"F1 Score: {f1_score(truth, pred):.4f}")
        return trx["predicted_label"]

    def get_closest_terminal(self, x: float, y: float) -> Terminal:
        closest_terminal = None
        closest_distance = float("inf")
        for terminal in self.attackable_terminals:
            distance = (terminal.x - x) ** 2 + (terminal.y - y) ** 2
            if distance < closest_distance:
                closest_terminal = terminal
                closest_distance = distance
        assert closest_terminal is not None
        return closest_terminal

    def save(self, directory: str = "cache"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, "banksys.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(directory: str = "cache"):
        """
        Load the banksys from the given directory.

        If `params` is given, it will check if the parameters match the saved ones.
        """

        with open(os.path.join(directory, "banksys.pkl"), "rb") as f:
            banksys = pickle.load(f)
            assert isinstance(banksys, Banksys)
        return banksys

    @cached_property
    def training_set(self):
        return self._transactions_df.filter(pl.col("timestamp").is_between(self.training_start, self.attack_start))

    @property
    def max_attack_duration(self):
        """
        Returns the maximum duration of the attack, which is the difference between the attack end and attack start.
        """
        return self.attack_end - self.attack_start

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the transactions iterator to avoid pickling it
        del state["trx_iterator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the transactions iterator
        remaining_trx = self._transactions_df.filter(pl.col("timestamp") > self.next_trx.timestamp)
        self.trx_iterator = remaining_trx.iter_rows(named=True)


def extract_trx_features(df: pl.DataFrame):
    weekday = df["timestamp"].dt.weekday()
    trx_df = df.with_columns(
        pl.col("timestamp").dt.weekday().cast(pl.Float32).alias("day_of_week"),
        pl.col("timestamp").dt.hour().cast(pl.Float32).alias("hour"),
        pl.col("is_online"),
        pl.col("amount"),
        *[pl.Series(name=day, values=(weekday == (i + 1)).cast(pl.Float32)) for i, day in enumerate("Mon Tue Wed Thu Fri Sat Sun".split())],
    )
    return trx_df.drop("timestamp")
