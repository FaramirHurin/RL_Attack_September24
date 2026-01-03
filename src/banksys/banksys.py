import logging
import os
import pickle
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from tqdm import tqdm
from utils.transaction_iterator import TransactionIterator

from .classification import ClassificationSystem
from .payer import Payer
from .terminal import Terminal
from .transaction import Transaction

if TYPE_CHECKING:
    from parameters import ClassificationParameters


WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class Banksys:
    def __init__(
        self,
        transactions: pl.DataFrame,
        payers: pl.DataFrame,
        terminals: pl.DataFrame,
        params: "ClassificationParameters",
        *,
        silent: bool = False,
        clf: ClassificationSystem | None = None,
    ):
        if params.classify_simulated_trx:
            raise NotImplementedError("Classification of simulated transactions is no longer supported.")
        self._transactions = transactions.sort("timestamp").with_columns(
            predicted_label=self._approximate_labels(transactions, fp_rate=params.fp_rate, fn_rate=params.fn_rate)
        )
        self.trx_iterator = TransactionIterator(self._transactions)
        self.payers = Payer.from_df(payers.sort("id"), params.aggregation_windows)
        self.terminals = Terminal.from_df(terminals.sort("id"), params.aggregation_windows)
        self.schema = pl.DataFrame(self.make_transaction_features(self.next_trx)).schema
        self.silent = silent
        if clf is None:
            clf = ClassificationSystem(params)
        self.clf = clf

        self.last_training = None
        self.training_features = list[dict[str, Any]]()
        self.training_labels = list[bool]()
        self.retrain_every = params.retrain_interval
        self.attack_start = self.t_start + params.longest_window + params.training_duration
        self.current_time = self.t_start
        assert self.attack_start < self.t_max, f"Attack start ({self.attack_start}) must precede attack end ({self.t_max})."
        logging.info("System warmup for training feature aggregation...")
        self._fast_forward(self.t_start + params.longest_window, show_progress=not self.silent, compute_features=False)
        logging.info("Building classifier training features...")
        self._fast_forward(self.attack_start, show_progress=not self.silent, compute_features=True)
        self.fit()

    @property
    def must_retrain_now(self):
        if self.last_training is None:
            # The system has never been trained, so there is no RE-training to do.
            return False
        if self.retrain_every is None:
            return False
        return self.current_time - self.last_training >= self.retrain_every

    @property
    def is_trained(self):
        return self.last_training is not None

    @cached_property
    def t_max(self) -> datetime:
        return self._transactions["timestamp"].max()  # type: ignore[return-value]

    @cached_property
    def t_start(self) -> datetime:
        return self._transactions["timestamp"].min()  # type: ignore[return-value]

    @property
    def will_retrain(self):
        """Whether the banksys will eventually retrain."""
        return self.retrain_every is not None

    @property
    def next_trx(self):
        """Peek at the next transaction to be processed."""
        return self.trx_iterator.peek()

    def fit(self):
        """
        Fit the classification system and process the training transactions.

        Called for training
        """
        logging.info(f"Training the Banksys classifier on {len(self.training_labels)} datapoints")
        train_x = pl.DataFrame(self.training_features, self.schema)
        train_y = np.array(self.training_labels)
        self.last_training = self.current_time
        self.clf.fit(train_x, train_y)
        # We clean the training set after training to implement the sliding window logic
        self.training_features.clear()
        self.training_labels.clear()

    def _fast_forward(self, until: datetime, *, show_progress: bool = False, compute_features: bool = False):
        """
        Fast forward the system to the given date, adding all the transactions to the
        system but without classifying them.
        """
        if until > self.t_max:
            raise ValueError(f"Cannot forward to {until}, it is beyond the attack end date {self.t_max}.")
        start = self.current_time
        if show_progress:
            n = self._transactions.filter(pl.col("timestamp").is_between(start, until)).height
        else:
            n = 0
        pbar = tqdm(total=n, desc="Fast-forwarding transactions", unit="trx", disable=not show_progress)
        date = start.date()
        while self.next_trx.timestamp < until:
            trx = next(self.trx_iterator)
            assert trx.predicted_label is not None
            if compute_features:
                feature = self.make_transaction_features(trx)
                self.training_features.append(feature)
                self.training_labels.append(trx.is_fraud)
            self.payers[trx.payer_id].add(trx, update_balance=False)
            self.terminals[trx.terminal_id].add(trx)
            if show_progress and trx.date != date:
                date = trx.date
                pbar.set_description(f"{date.isoformat()}")
            if self.must_retrain_now:
                self.fit()
            pbar.update()
        pbar.close()
        self.current_time = until

    def process_transaction(self, trx: Transaction):
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        If `real_label` is True, it will use the real label from the transaction.
        """
        assert trx.predicted_label is None, "Transaction has already been processed !"
        assert trx.is_fraud, "Method `process_transaction` is meant to process fraudulent transactions only."
        self._fast_forward(trx.timestamp, compute_features=self.will_retrain)
        features = self.make_transaction_features(trx)
        self.training_features.append(features)
        self.training_labels.append(trx.is_fraud)
        features_df = pl.DataFrame(features, schema=self.schema)
        trx.predicted_label = self.clf.predict(features_df).item()

        self.payers[trx.payer_id].add(trx, update_balance=True)
        self.terminals[trx.terminal_id].add(trx)
        return features

    def make_transaction_features(self, trx: Transaction):
        weekday = [0.0] * 7
        weekday[trx.weekday_index] = 1.0
        return {
            "hour": trx.timestamp.hour,
            "is_online": trx.is_online,
            "amount": trx.amount,
            **{day: val for day, val in zip(WEEKDAYS, weekday)},
            **self.payers[trx.payer_id].compute_features(trx.timestamp),
            **self.terminals[trx.terminal_id].compute_features(trx.timestamp),
        }

    def _approximate_labels(self, trx: pl.DataFrame, fp_rate: float, fn_rate: float):
        assert 0 <= fp_rate <= 1.0 and 0 <= fn_rate <= 1.0, "Rates must be between 0 and 1"
        # Random values for conditional flipping
        rand = pl.Series("rand", np.random.rand(len(trx)))
        # Flip logic
        trx = trx.with_columns(
            predicted_label=pl.when((pl.col("is_fraud") == 1) & (rand < fn_rate))
            .then(0)
            .when((pl.col("is_fraud") == 0) & (rand < fp_rate))
            .then(1)
            .otherwise(pl.col("is_fraud"))
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
