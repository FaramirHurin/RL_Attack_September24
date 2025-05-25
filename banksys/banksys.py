import logging
import os
import pickle
import random
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm


from .card import Card
from .classification import ClassificationSystem
from .has_ordered_transactions import OrderedTransactionsRegistry
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
        logging.info(f"Training the system until {self.attack_start.date().isoformat()}")
        rows = []
        labels = []
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)
            rows.append(self.make_features_np(t, with_label=False))
            labels.append(t.is_fraud)
        df = pd.DataFrame(rows, columns=self.feature_names)
        labels = np.array(labels, dtype=np.bool)
        self.clf.fit(df, labels)
        return df, labels

    def _simulate(self, transactions: list[Transaction]):
        logging.info(f"Simulating the system until {self.attack_end.date().isoformat()}")
        features, labels = [], []
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)
            features.append(self.make_features_np(t, with_label=False))
            labels.append(t.is_fraud)
        df = pd.DataFrame(features, columns=self.feature_names)
        return df, np.array(labels, dtype=np.bool)

    def fit(self, transactions: list[Transaction]):
        logging.info("Fitting the bank system...")
        registry = OrderedTransactionsRegistry(transactions)
        aggregation_duration = max(*self.aggregation_windows)
        training_start = registry[0].timestamp + aggregation_duration
        training_end = training_start + self.clf.training_duration
        simulation_end = registry[-1].timestamp

        self.attack_start = training_end
        self.attack_end = simulation_end
        assert self.attack_start < simulation_end, f"The simulation ends ({simulation_end}) before the attack starts ({self.attack_start})"

        self._warmup(registry.get_before(training_start))
        train_x, train_y = self._train(registry.get_between(training_start, training_end))
        test_x, test_y = self._simulate(registry.get_after(training_end))
        return train_x, train_y, test_x, test_y

    def set_up_run(self, rules_values: dict, use_anomaly: bool):
        self.clf.set_rules(rules_values)
        self.clf.use_anomaly = use_anomaly

    def make_features_np(self, transaction: Transaction, with_label: bool):
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
