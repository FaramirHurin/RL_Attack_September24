import logging
import os
import pickle
import random
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm


from .card import Card
from .classification import ClassificationSystem
from .transaction_registry import TransactionsRegistry
from .terminal import Terminal
from .transaction import Transaction

if TYPE_CHECKING:
    from parameters import CardSimParameters, ClassificationParameters

# TODO: when a card is blocked, we should remove all its future transactions


class Banksys:
    def __init__(
        self,
        cards: list[Card],
        terminals: list[Terminal],
        aggregation_windows: Sequence[timedelta],
        clf_params: "ClassificationParameters",
        attackable_terminal_factor: float = 1.0,
    ):
        self.clf = ClassificationSystem(self, clf_params)
        self.cards = cards
        self.terminals = terminals
        self.aggregation_windows = aggregation_windows
        self.feature_names = [
            *Transaction.feature_names(),
            *Card.feature_names(aggregation_windows),
            *Terminal.feature_names(aggregation_windows),
        ]
        self.attackable_terminals = random.sample(terminals, round(len(terminals) * attackable_terminal_factor))
        self.attack_start = datetime.now()
        self.attack_end = datetime.now()

    def _warmup(self, transactions: list[Transaction]):
        logging.info(f"Warming up the system for aggregation until {transactions[-1].timestamp.date().isoformat()}")
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)

    def _train(self, transactions: list[Transaction]):
        logging.info("Adding training transactions & computing features...")
        # Since there are few transactions, we can use a high number of jobs (since the copy of the data is fast)
        df, labels = self.parallel_add_and_make_features(transactions, n_jobs=8)
        logging.info("Training the classification system...")
        self.clf.fit(df, labels)
        return df, labels

    def fit(self, registry: TransactionsRegistry):
        logging.info("Fitting the bank system...")
        aggregation_duration = max(*self.aggregation_windows)
        training_start = registry[0].timestamp + aggregation_duration
        training_end = training_start + self.clf.training_duration
        simulation_end = registry[-1].timestamp

        self.attack_start = training_end
        self.attack_end = simulation_end
        assert self.attack_start < simulation_end, f"The simulation ends ({simulation_end}) before the attack starts ({self.attack_start})"

        self._warmup(registry.get_before(training_start))
        train_x, train_y = self._train(registry.get_between(training_start, training_end))
        return train_x, train_y

    def generate_test_set(self, registry: TransactionsRegistry):
        logging.info("Generating test set...")
        n_jobs = mp.cpu_count()
        test_x, test_y = self.parallel_add_and_make_features(registry.get_after(self.attack_start), n_jobs=min(8, n_jobs))
        return test_x, test_y

    def set_up_run(self, rules_values: dict, use_anomaly: bool):
        self.clf.set_rules(rules_values)
        self.clf.use_anomaly = use_anomaly

    def parallel_add_and_make_features(self, transactions: list[Transaction], n_jobs: int = 4):
        labels = []
        handles = list[AsyncResult[np.ndarray]]()
        chunk_size = len(transactions) // n_jobs + 1
        chunks = [transactions[i * chunk_size : (i + 1) * chunk_size] for i in range(n_jobs)]
        current_chunk_num = 0
        chunk_end = chunks[0][-1].timestamp
        logging.info(f"Extracting features for {len(transactions)} transactions in {n_jobs} parallel jobs")
        pool = mp.Pool(n_jobs)
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)
            labels.append(t.is_fraud)
            if t.timestamp >= chunk_end and current_chunk_num < n_jobs - 1:
                # Submit as soon as the related transactions have been added
                handles.append(pool.apply_async(self.make_batch_features_np, args=(chunks[current_chunk_num],)))
                current_chunk_num += 1
                chunk_end = chunks[current_chunk_num][-1].timestamp
        # Process the last chunk in this process to avoid memory overhead
        features = self.make_batch_features_np(chunks[current_chunk_num], with_label=False)
        results = [h.get() for h in handles] + [features]
        df = pd.DataFrame(np.concatenate(results), columns=self.feature_names)
        return df, np.array(labels, dtype=np.bool)

    def make_batch_features_np(self, transactions: list[Transaction], with_label: bool = False):
        """
        Make features for a batch of transactions.
        """
        features = []
        for transaction in transactions:
            features.append(self.make_features_np(transaction, with_label=with_label))
        return np.array(features)

    def make_features_np(self, transaction: Transaction, with_label: bool = False):
        terminal = self.terminals[transaction.terminal_id]
        card = self.cards[transaction.card_id]
        terminal_features = terminal.features(transaction.timestamp, self.aggregation_windows)
        card_features = card.features(transaction.timestamp, self.aggregation_windows)
        if with_label:
            assert transaction.is_fraud is not None, "Label must be set for the transaction used in the agredated features"
            return np.concatenate([transaction.features, terminal_features, card_features, [transaction.is_fraud]])
        return np.concatenate([transaction.features, terminal_features, card_features])

    def make_features_df(self, transaction: Transaction, with_label: bool):
        features = self.make_features_np(transaction, with_label=with_label)
        feature_names = self.feature_names
        if with_label:
            feature_names = feature_names + ["label"]
        return pd.DataFrame(features.reshape(1, -1), columns=feature_names)

    def process_transaction(self, transaction: Transaction) -> bool:
        """
        Process the transaction (i.e. add it to the system) and return whether it is fraudulent or not.
        """
        label = self.clf.predict(transaction)
        transaction.predicted_label = label
        self.add_transaction(transaction)
        return label

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
        self.terminals[transaction.terminal_id].add_transaction(transaction)
        self.cards[transaction.card_id].add_transaction(transaction)

    def save(self, params: "CardSimParameters", directory: str = "cache"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, "banksys.pkl"), "wb") as f:
            pickle.dump(self, f)
        with open(os.path.join(directory, "params.json"), "wb") as f:
            f.write(orjson.dumps(params))

    @staticmethod
    def load(params: Optional["CardSimParameters"] = None, directory: str = "cache") -> "Banksys":
        """
        Load the banksys from the given directory.

        If `params` is given, it will check if the parameters match the saved ones.
        """
        from parameters import CardSimParameters

        if params is not None:
            with open(os.path.join(directory, "params.json"), "rb") as f:
                simulation_params = CardSimParameters(**orjson.loads(f.read()))
            if simulation_params != params:
                raise ValueError("Simulation parameters do not match the given parameters.")
        with open(os.path.join(directory, "banksys.pkl"), "rb") as f:
            banksys = pickle.load(f)
            assert isinstance(banksys, Banksys)
        return banksys

    def rollback(self, transactions: list[Transaction]):
        """
        Undo the transactions. Typically called when the environment is reset.
        """
        for transaction in transactions:
            self.terminals[transaction.terminal_id].remove(transaction)
            self.cards[transaction.card_id].remove(transaction)

    def test(self, transactions: list[Transaction]):
        """
        Compute the confusion matrix for the given transactions.
        """
        features = []
        labels = []
        for transaction in tqdm(transactions):
            features.append(self.make_features_np(transaction, with_label=False))
            labels.append(transaction.is_fraud)
        features = np.array(features)
        labels = np.array(labels)
        df = pd.DataFrame(features, columns=self.feature_names)
        # self.test_X = df
        # self.test_y = labels
        return self.clf.predict(df)
