import logging
import os
import pickle
from functools import cached_property
from datetime import datetime
from typing import TYPE_CHECKING, Any
import numpy as np
import polars as pl
from tqdm import tqdm
from .payer import Payer
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction
from utils import tb_log

if TYPE_CHECKING:
    from parameters import ClassificationParameters


WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class Banksys:
    def __init__(
        self,
        transactions_df: pl.DataFrame,
        cards_df: pl.DataFrame,
        terminals_df: pl.DataFrame,
        params: "ClassificationParameters",
        silent: bool = False,
        clf: ClassificationSystem | None = None,
    ):
        max_aggregation_duration = (
            max(*params.aggregation_windows) if len(params.aggregation_windows) > 1 else params.aggregation_windows[0]
        )
        self.current_time: datetime = transactions_df["timestamp"].min()  # type: ignore
        self.training_start = self.current_time + max_aggregation_duration
        self.attack_start = self.training_start + params.training_duration
        self.attack_end: datetime = transactions_df["timestamp"].max()  # type: ignore
        assert self.attack_start < self.attack_end, f"Attack start ({self.attack_start}) must be before attack end ({self.attack_end})."
        self.silent = silent
        if clf is not None:
            self.clf = clf
        else:
            self.clf = ClassificationSystem(params)
        self._transactions_df = (
            transactions_df.sort("timestamp")  # Sort by timestamp
            .with_columns(
                predicted_label=self._approximate_labels(transactions_df, fp_rate=params.fp_rate, fn_rate=params.fn_rate)
            )  # Add training "predicted_label"
            .with_columns(
                pl.when(pl.col("timestamp") > self.attack_start)  # Remove 'predicted_label' for the attack set.
                .then(None)
                .otherwise(pl.col("predicted_label"))
                .alias("predicted_label")
            )
        )
        self.trx_iterator = self._transactions_df.iter_rows(named=True)
        self.next_trx = Transaction(**next(self.trx_iterator))
        self.payers = sorted(Payer.from_df(cards_df, params.aggregation_windows), key=lambda c: c.id)
        self.terminals = sorted(Terminal.from_df(terminals_df, params.aggregation_windows), key=lambda t: t.id)
        self.aggregation_windows = params.aggregation_windows
        self.schema = self.fit()

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
        return train_x.schema

    def fast_forward(self, until: datetime):
        """
        Fast forward the system to the given date, adding all the transactions to the
        system but without classifying them.
        """
        if until > self.attack_end:
            raise ValueError(f"Cannot forward to {until}, it is beyond the attack end date {self.attack_end}.")
        start = self.next_trx.timestamp
        n = self._transactions_df.filter(pl.col("timestamp").is_between(start, until)).height
        pbar = tqdm(total=n, desc="Fast-forwarding transactions", unit="trx", disable=self.silent, mininterval=1.0)
        features = list[dict[str, Any]]()
        date = start.date()
        while self.next_trx.timestamp < until:
            features.append(self.make_transaction_features(self.next_trx))
            self.payers[self.next_trx.payer_id].add(self.next_trx, update_balance=False)
            self.terminals[self.next_trx.terminal_id].add(self.next_trx)
            if self.next_trx.timestamp.date() != date:
                date = self.next_trx.timestamp.date()
                pbar.set_description(f"{date.isoformat()}")
            pbar.update()
            self.next_trx = Transaction(**next(self.trx_iterator))
        pbar.close()
        self.current_time = until
        return features

    def simulate_until(self, until: datetime):
        """
        Simulate the system until the given date, processing all transactions up to that date (excluded).
        A "predicted label" is assigned to each transaction via the classification system.
        """
        if until > self.attack_end:
            raise ValueError(f"Cannot forward to {until}, it is beyond the attack end date {self.attack_end}.")

        cards = set[int]()
        terms = set[int]()
        batch = list[Transaction]()
        features = list[pl.DataFrame]()
        while self.next_trx.timestamp < until:
            if self.next_trx.payer_id in cards or self.next_trx.terminal_id in terms:
                features.append(self.process_transactions(batch))
                cards.clear()
                terms.clear()
                batch.clear()
            cards.add(self.next_trx.payer_id)
            terms.add(self.next_trx.terminal_id)
            batch.append(self.next_trx)
            self.next_trx = Transaction(**next(self.trx_iterator))
        if len(batch) > 0:
            features.append(self.process_transactions(batch))
        self.current_time = until
        return features

    def process_transaction(self, trx: Transaction):
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        If `real_label` is True, it will use the real label from the transaction.
        """
        assert trx.predicted_label is None, "Transaction has already been processed !"
        assert trx.is_fraud, "Method `process_transaction` is meant to process fraudulent transactions only."
        self.simulate_until(trx.timestamp)
        features = self.make_transaction_features(trx)
        elapsed = trx.timestamp - self.attack_start
        tb_log("features", features, elapsed)
        tb_log("trx/payer-balance", self.payers[trx.payer_id].balance, elapsed)
        tb_log("trx/terminal-id", trx.terminal_id, elapsed)
        features_df = pl.DataFrame(features, schema=self.schema)
        trx.predicted_label = self.clf.predict(features_df).item()
        self.payers[trx.payer_id].add(trx, update_balance=True)
        self.terminals[trx.terminal_id].add(trx)
        return features
        to_add = trx.as_df(with_label=True, with_predicted_label=True)
        inserting_pos = self._transactions_df.filter(pl.col("timestamp") <= trx.timestamp).height
        self._transactions_df = pl.concat(
            [
                self._transactions_df.slice(0, inserting_pos),
                to_add,
                self._transactions_df.slice(inserting_pos),
            ],
            how="vertical",
        )

    def process_transactions(self, transactions: list[Transaction]):
        """
        Receives a list of chronological transactions and processes them, assigning a predicted label to each transaction.
        """
        # TODO: After 7 days, update the predicted label of all past transactions to
        # be equal to the real label, to simulate delayed fraud detection.
        df = pl.DataFrame([self.make_transaction_features(trx) for trx in transactions], schema=self.schema)
        labels = self.clf.predict(df)
        # Use transactions labels
        # labels = pl.Series([trx.is_fraud for trx in transactions])
        for trx, label in zip(transactions, labels):
            trx.predicted_label = label
            self.terminals[trx.terminal_id].add(trx)
            self.payers[trx.payer_id].add(trx, update_balance=False)
        return df

    def make_transaction_features(self, trx: Transaction):
        weekday = [0.0] * 7
        weekday[trx.timestamp.weekday()] = 1.0
        return {
            "hour": trx.timestamp.hour,
            "is_online": trx.is_online,
            "amount": trx.amount,
            **{day: val for day, val in zip(WEEKDAYS, weekday)},
            **self.payers[trx.payer_id].compute_features(trx.timestamp),
            **self.terminals[trx.terminal_id].compute_features(trx.timestamp),
        }

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
        return trx["predicted_label"]

    def save(self, file_path: str):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, "rb") as f:
            banksys = pickle.load(f)
            assert isinstance(banksys, Banksys)
        return banksys

    @cached_property
    def training_set(self):
        return self._transactions_df.filter(pl.col("timestamp").is_between(self.training_start, self.attack_start, closed="left"))

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
