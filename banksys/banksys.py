import logging
import os
import pickle
import random
from tqdm import tqdm
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Optional, Sequence

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
    ):
        self.max_aggregation_duration = max(*aggregation_windows)
        self.t0: datetime = transactions_df["timestamp"].min()  # type: ignore
        self.training_start = self.t0 + self.max_aggregation_duration
        self.attack_start = self.training_start + clf_params.training_duration
        self.attack_end: datetime = transactions_df["timestamp"].max()  # type: ignore
        self.current_date = self.t0
        self.current_offset = 0
        self.clf_features = None
        self.clf = ClassificationSystem(clf_params)

        self._transactions_df = (
            transactions_df.sort("timestamp")  # Sort by timestamp
            .with_columns(self._approximate_labels(transactions_df).alias("predicted_label"))  # Add training "predicted_label"
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
        from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

        truth = trx["is_fraud"]
        pred = trx["predicted_label"]

        cm = confusion_matrix(truth, pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Accuracy: {accuracy_score(truth, pred):.4f}")
        logging.info(f"Recall: {recall_score(truth, pred):.4f}")
        logging.info(f"Precision: {precision_score(truth, pred):.4f}")
        logging.info(f"F1 Score: {f1_score(truth, pred):.4f}")
        return trx["predicted_label"]

    def fit(self):
        """
        Fit the classification system and process the training transactions.
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

        logging.info("Adding training transactions to the system...")
        # Set the current offset to the transactions that have to be processed for aggregation
        self.current_offset = df.filter(pl.col("timestamp") < (self.attack_start - self.max_aggregation_duration)).height
        self.simulate_until(self.attack_start, show_progress=True, n_transactions_to_process=df.height - self.current_offset)

    def make_features(self, df: pl.DataFrame):
        trx_df = extract_trx_features(df).drop("is_fraud")
        card_agg = df.group_by("card_id", maintain_order=True).map_groups(lambda g: card_aggregation(self.aggregation_windows, g))
        terminal_agg = df.group_by("terminal_id", maintain_order=True).map_groups(
            lambda g: terminal_aggregation(self.aggregation_windows, g)
        )
        features = pl.concat([trx_df, card_agg, terminal_agg], how="horizontal")
        if self.clf_features is not None:
            features = features.select(self.clf_features)
        return features

    def simulate_until(self, until_date: datetime, /, show_progress: bool = False, n_transactions_to_process: Optional[int] = None):
        """
        Simulate the system until the given date, processing all transactions up to that date.
        A "predicted label" is assigned to each transaction via the classification system.
        """
        if until_date < self.current_date:
            raise ValueError(f"Cannot forward to {until_date}, current date is {self.current_date}.")
        if until_date >= self.attack_end:
            raise ValueError(f"Cannot forward to {until_date}, it is beyond the attack end date {self.attack_end}.")

        def flush(batch: list[Transaction]):
            if len(batch) == 0:
                return
            end_idx = self.current_offset + len(batch)
            features = self.make_features(self._transactions_df[self.current_offset : end_idx])
            self.current_offset = end_idx
            labels = self.clf.predict(features)
            for label, trx in zip(labels, batch):
                trx.predicted_label = bool(label)
                self.add_transaction(trx)

        pbar = tqdm(
            self._transactions_df[self.current_offset :].iter_rows(named=True),
            unit="trx",
            total=n_transactions_to_process,
            disable=not show_progress,
        )
        end = until_date.date().isoformat()
        cards = set[int]()
        terminals = set[int]()
        batch = list[Transaction]()
        for kwargs in pbar:
            trx = Transaction(**kwargs)
            if trx.timestamp >= until_date:
                flush(batch)
                break
            if trx.card_id in cards or trx.terminal_id in terminals:
                pbar.set_description(f"{trx.timestamp.date().isoformat()}/{end}")
                flush(batch)
                cards.clear()
                terminals.clear()
                batch.clear()
            batch.append(trx)
            cards.add(trx.card_id)
            terminals.add(trx.terminal_id)
            # features = self.make_transaction_features(trx)
            # trx.predicted_label = self.clf.predict(features).item()
        pbar.close()
        self.current_date = until_date

    def process_transaction(self, trx: Transaction) -> bool:
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        """
        # label, cause_of_detection = self.clf.predict(transaction)
        # if not label:
        #     debug = 0
        # transaction.predicted_label = label
        # self.add_transaction(transaction)
        # return label, cause_of_detection
        self.simulate_until(trx.timestamp)
        features = self.make_transaction_features(trx)
        trx.predicted_label = self.clf.predict(features).item()
        self.add_transaction(trx)
        return trx.predicted_label

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

    def add_transaction(self, transaction: Transaction):
        self.terminals[transaction.terminal_id].add(transaction)
        self.cards[transaction.card_id].add(transaction)

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
    return pl.DataFrame(results)


def terminal_aggregation(agg_windows: Sequence[timedelta], term_df: pl.DataFrame):
    results = list[pl.Series]()
    for delta in agg_windows:
        count, risk = count_and_risk(term_df["timestamp"].to_list(), term_df["predicted_label"].to_list(), delta)
        results.append(pl.Series(name=f"terminal_n_trx_last_{delta}", values=count))
        results.append(pl.Series(name=f"terminal_risk_last_{delta}", values=risk))
    return pl.DataFrame(results)


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


def count_and_mean(timestamps: Sequence[datetime], amounts: Sequence[float], delta: timedelta):
    counts, means = [], []
    start_idx = 0
    window_amount = 0.0
    window_vals = []
    for end_index in range(len(timestamps)):
        # Add current transaction to window
        window_vals.append(amounts[end_index])
        window_amount += amounts[end_index]
        # Slide left boundary of the window
        while start_idx < end_index and timestamps[start_idx] < timestamps[end_index] - delta:
            window_amount -= window_vals[0]
            window_vals.pop(0)
            start_idx += 1
        count = len(window_vals)
        counts.append(count)
        means.append(window_amount / count if count > 0 else 0.0)
    return counts, means


def count_and_risk(timestamps: Sequence[datetime], predicted_label: Sequence[bool], delta: timedelta):
    counts, risks = [], []
    start_idx = 0
    for stop_idx, t_end in enumerate(timestamps):
        # Move left pointer to maintain window
        start = t_end - delta
        while timestamps[start_idx] < start:
            start_idx += 1
        # Count of transactions within the window excluding the current one
        length = stop_idx - start_idx
        counts.append(length)
        # Compute risk as the proportion of fraudulent transactions
        if length > 0:
            # We assume that only a portion of the frauds are detected
            risks.append(sum(predicted_label[start_idx:stop_idx]) / length)
        else:
            risks.append(0.0)
    return counts, risks
