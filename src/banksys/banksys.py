import logging
import os
import pickle
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import polars as pl

from .card import Card
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction

if TYPE_CHECKING:
    from parameters import ClassificationParameters

# TODO: when a card is blocked, we should remove all its future transactions


class Banksys:
    def __init__(
        self,
        transactions_df: pl.DataFrame,
        cards_df: pl.DataFrame,
        terminals_df: pl.DataFrame,
        aggregation_windows: Sequence[timedelta],
        clf_params: "ClassificationParameters",
        attackable_terminal_factor: float = 1.0,
        fp_rate=0.01,
        fn_rate=0.01,
    ):
        self.max_aggregation_duration = max(*aggregation_windows) if len(aggregation_windows) > 1 else aggregation_windows[0]
        self.t0: datetime = transactions_df["timestamp"].min()  # type: ignore
        self.training_start = self.t0 + self.max_aggregation_duration
        self.attack_start = self.training_start + clf_params.training_duration
        self.attack_end: datetime = transactions_df["timestamp"].max()  # type: ignore

        assert self.attack_start < self.attack_end, "Attack start must be before attack end."

        self.current_date = self.t0
        self.clf_features = None
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
        self.cards = sorted(Card.from_df(cards_df), key=lambda c: c.id)
        self.terminals = sorted(Terminal.from_df(terminals_df), key=lambda t: t.id)
        self.aggregation_windows = aggregation_windows
        self.attackable_terminals = random.sample(self.terminals, round(len(self.terminals) * attackable_terminal_factor))
        self.trx_iterator = iter([dict[str, Any]()])
        self.next_transaction = Transaction(0, datetime.min, 0, 0, False, False)
        self._setup()

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

    def _setup(self):
        """
        Fit the classification system and process the training transactions.

        Automatically called from the constructor.
        """
        logging.info("Computing training features...")
        df = self._transactions_df.filter(pl.col("timestamp") < self.attack_start)
        # Since there is no trained predictor yet, we approximate the labels
        all_features = self.make_features(df)
        self.clf_features = all_features.drop("is_fraud", "predicted_label", "timestamp", strict=False).columns
        all_features = all_features.with_columns(df["timestamp"], df["is_fraud"])

        # Only keep training transactions and discard warm-up transactions (required for aggregation)
        df_train = all_features.filter(pl.col("timestamp").is_between(self.training_start, self.attack_start))
        train_x = df_train.select(self.clf_features)
        train_y = df_train["is_fraud"].to_numpy().astype(np.bool)
        self.clf.fit(train_x, train_y)
        print(df)
        print(df_train)
        logging.info("Adding training transactions to the system...")
        # Set the current offset to the transactions that have to be processed for aggregation
        start_offset = df.filter(pl.col("timestamp") < (self.attack_start - self.max_aggregation_duration * 2)).height
        self.trx_iterator = iter(self._transactions_df[start_offset:].iter_rows(named=True))
        self.next_transaction = Transaction(**next(self.trx_iterator))
        self.simulate_until(self.attack_start)

    def make_features(self, df: pl.DataFrame):
        trx_df = extract_trx_features(df).drop("is_fraud")
        card_agg = (
            df.group_by("card_id")
            .map_groups(lambda group: card_aggregation(self.aggregation_windows, group))
            .sort("timestamp")
            .drop("timestamp")
        )
        terminal_agg = (
            df.group_by("terminal_id")
            .map_groups(lambda group: terminal_aggregation(self.aggregation_windows, group))
            .sort("timestamp")
            .drop("timestamp")
        )
        features = pl.concat([trx_df, card_agg, terminal_agg], how="horizontal")
        if self.clf_features is not None:
            features = features.select(self.clf_features)
        return features

    def simulate_until(self, until_date: datetime):
        """
        Simulate the system until the given date, processing all transactions up to that date.
        A "predicted label" is assigned to each transaction via the classification system.
        """
        if until_date < self.current_date:
            raise ValueError(f"Cannot go back in time. Current date is {self.current_date}, required date is {until_date}.")
        if until_date >= self.attack_end:
            raise ValueError(f"Cannot forward to {until_date}, it is beyond the attack end date {self.attack_end}.")

        cards = set[int]()
        terminals = set[int]()
        features, transactions = [], list[Transaction]()
        while self.next_transaction.timestamp < until_date:
            if self.next_transaction.predicted_label is None:
                # Compute its features for classification
                transactions.append(self.next_transaction)
                features.append(self.make_transaction_features(self.next_transaction))
                # We only classify the transactions when the same card or terminal has been seen before,
                # since otherwise the features would not take into account previous transactions from the same
                # card or terminal.
                if self.next_transaction.card_id in cards or self.next_transaction.terminal_id in terminals:
                    # We need to process the transactions
                    df = pl.DataFrame(features)
                    labels = self.clf.predict(df)
                    for label, trx in zip(labels, transactions):
                        trx.predicted_label = label
                    cards.clear()
                    terminals.clear()
            # Then add it to the system
            cards.add(self.next_transaction.card_id)
            terminals.add(self.next_transaction.terminal_id)
            self.add_transaction(self.next_transaction, update_balance=False)
            self.next_transaction = Transaction(**next(self.trx_iterator))
        self.current_date = until_date

    def process_transaction(self, trx: Transaction):
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        """
        self.simulate_until(trx.timestamp)
        features = self.make_transaction_features(trx)
        label, cause = self.clf.predict(features)
        trx.predicted_label = label.item()
        self.add_transaction(trx, update_balance=True)
        return trx.predicted_label, cause

    def make_transaction_features(self, trx: Transaction):
        trx_df = trx.as_df()
        trx_features = extract_trx_features(trx_df)
        f = pl.DataFrame(
            [
                {
                    **self.cards[trx.terminal_id].transactions.count_and_mean(self.aggregation_windows, trx.timestamp),
                    **self.terminals[trx.terminal_id].transactions.count_and_risk(self.aggregation_windows, trx.timestamp),
                }
            ]
        )
        # Concat and re-order the columns to match the training features
        features = pl.concat([trx_features, f], how="horizontal").select(self.clf_features)
        return features

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

    def add_transaction(self, transaction: Transaction, update_balance: bool):
        self.terminals[transaction.terminal_id].add(transaction)
        self.cards[transaction.card_id].add(transaction, update_balance)

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


def card_aggregation(agg_windows: Sequence[timedelta], card_df: pl.DataFrame):
    results = list[pl.Series]()
    for delta in agg_windows:
        count, mean = count_and_mean(card_df["timestamp"].to_list(), card_df["amount"].to_list(), delta)
        results.append(pl.Series(name=f"card_n_trx_last_{delta}", values=count))
        results.append(pl.Series(name=f"card_mean_amount_last_{delta}", values=mean))
    return pl.DataFrame(results).with_columns(card_df["timestamp"])


def terminal_aggregation(agg_windows: Sequence[timedelta], term_df: pl.DataFrame):
    results = list[pl.Series]()
    for delta in agg_windows:
        count, risk = count_and_mean(term_df["timestamp"].to_list(), term_df["predicted_label"].to_list(), delta)
        results.append(pl.Series(name=f"terminal_n_trx_last_{delta}", values=count))
        results.append(pl.Series(name=f"terminal_risk_last_{delta}", values=risk))
    return pl.DataFrame(results).with_columns(term_df["timestamp"])


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


def count_and_mean(timestamps: Sequence[datetime], values: Sequence[float], delta: timedelta, start_index=1):
    counts, means = [0], [0.0]  # The count and mean for the first transaction is always 0
    window_values = values[0]
    left = 0
    window_vals = [window_values]
    # Do not take the current transaction into account for its own aggregation
    for right in range(start_index, len(timestamps)):
        # Slide left boundary of the window
        while left < right and timestamps[left] < timestamps[right] - delta:
            window_values -= window_vals.pop(0)
            left += 1
        # Add the data to the window
        count = right - left
        counts.append(count)
        means.append(window_values / count if count > 0 else 0.0)

        # Then update the windows to include the current transaction
        window_vals.append(values[right])
        window_values += values[right]
    return counts, means
