import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
import random

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
    from parameters import CardSimParameters

# TODO: when a card is blocked, we should remove all its future transactions


class Banksys:
    clf: ClassificationSystem
    attack_start: datetime
    terminals: list[Terminal]
    cards: list[Card]
    feature_names: list[str]
    training_start: datetime
    training_end: datetime
    train_X: pd.DataFrame
    train_y: np.ndarray
    test_X: pd.DataFrame
    test_y: np.ndarray

    def __init__(
        self,
        cards: list[Card],
        terminals: list[Terminal],
        training_duration: timedelta,
        transactions: list[Transaction],
        feature_names: list[str],
        quantiles: list[float],
        contamination: float,
        trees: int,
        balance_factor: float,
        attackable_terminal_factor: float = 1.0,
    ):
        self.clf = ClassificationSystem(
            banksys=self,
            features_for_quantiles=feature_names,
            trees=trees,
            contamination=contamination,
            balance_factor=balance_factor,
            quantiles=quantiles,
        )
        self.cards = cards
        self.terminals = terminals
        self.feature_names = transactions[0].feature_names + self.cards[0].feature_names + self.terminals[0].feature_names
        self.attackable_terminals = random.sample(terminals, round(len(terminals) * attackable_terminal_factor))

        registry = OrderedTransactionsRegistry(transactions)
        aggregation_duration = max(*cards[0].aggregation_windows, *terminals[0].days_aggregation)
        training_start = registry[0].timestamp + aggregation_duration
        training_end = training_start + training_duration

        self.attack_start = training_end
        self.attack_end = registry[-1].timestamp
        assert self.attack_start < self.attack_end, "The duration of the simulation is too short to allow for an attack"

        self._warmup(registry.get_before(training_start))
        self._train(registry.get_between(training_start, training_end))
        self._simulate(registry.get_after(training_end))

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
            features = self.make_features_np(t, with_label=False)
            rows.append(features)
            labels.append(t.is_fraud)
        df = pd.DataFrame(rows, columns=self.feature_names)
        labels = np.array(labels, dtype=np.bool)
        self.train_X = df
        self.train_y = labels
        self.clf.fit(df, labels)

    def _simulate(self, transactions: list[Transaction]):
        logging.info(f"Simulating the system until {self.attack_end.date().isoformat()}")
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)

    def set_up_run(self, rules_values: dict, use_anomaly: bool):
        self.clf.set_rules(rules_values)
        self.clf.use_anomaly = use_anomaly

    def make_features_np(self, transaction: Transaction, with_label: bool):
        terminal = self.terminals[transaction.terminal_id]
        card = self.cards[transaction.card_id]
        terminal_features = terminal.features(transaction.timestamp)
        card_features = card.features(transaction.timestamp)
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

    def test(self, transactions: list[Transaction], predicted_labels: bool = True):
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
        self.test_X = df
        self.test_y = labels
        if predicted_labels:
            predicted_labels = self.clf.predict(df)
            return predicted_labels, labels
        else:
            return
