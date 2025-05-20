import os
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from .card import Card
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction
from .has_ordered_transactions import OrderedTransactionsRegistry

# TODO: when a card is blocked, we should remove all its future transactions


class Banksys:
    clf: ClassificationSystem
    attack_start: datetime
    terminals: list[Terminal]
    cards: list[Card]
    feature_names: list[str]

    def __init__(
        self,
        cards: list[Card],
        terminals: list[Terminal],
        training_duration: timedelta,
        transactions: list[Transaction],
        feature_names: list[str],
        quantiles: list[float],
    ):
        self.clf = ClassificationSystem(banksys=self, features_for_quantiles=feature_names, quantiles=quantiles)
        self.cards = cards
        self.terminals = terminals
        self.feature_names = transactions[0].feature_names + self.cards[0].feature_names + self.terminals[0].feature_names

        registry = OrderedTransactionsRegistry(transactions)
        aggregation_duration = max(*cards[0].aggregation_windows, *terminals[0].days_aggregation)
        training_start = registry[0].timestamp + aggregation_duration
        training_end = training_start + training_duration

        self._warmup(registry.get_before(training_start))
        self._train(registry.get_between(training_start, training_end))
        self._simulate(registry.get_after(training_end))

        self.attack_start = training_end
        # train_transactions = registry.get_before(training_end)
        # self._train_classifier(train_transactions)
        # # Add test set
        # for transaction in test_transactions:
        #     self.add_transaction(transaction)

    def _warmup(self, transactions: list[Transaction]):
        logging.info("Warming up the system for aggregation")
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)

    def _train(self, transactions: list[Transaction]):
        logging.info("Training the system")
        rows = []
        labels = []
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)
            features = self.make_features_np(t, with_label=False)
            rows.append(features)
            labels.append(t.is_fraud)
        df = pd.DataFrame(rows, columns=self.feature_names)
        labels = np.array(labels, dtype=np.bool)
        self.clf.fit(df, labels)

    def _simulate(self, transactions: list[Transaction]):
        logging.info("Simulating the system until the end")
        for t in tqdm(transactions, unit="trx"):
            self.add_transaction(t)

    def set_up_run(self, rules: list, rules_values: dict, return_confusion: bool = False):
        self.clf.set_rules(rules, rules_values)
        # if return_confusion:
        #     return self._confusion_matrix(self.test_transactions)
        # return None

    # def _train_classifier(self, tr_transactions: list[Transaction]):
    #     rows = []
    #     for t in tqdm(tr_transactions):
    #         self.add_transaction(t)
    #         features = self.make_features_np(t, with_label=True)
    #         rows.append(features)

    #     trainign_DF = pd.DataFrame(rows, columns=self.feature_names + ["label"])

    #     # Define the features and label
    #     training_features = [col for col in trainign_DF.columns if col != "label"]
    #     x_train = trainign_DF[training_features]
    #     y_train = trainign_DF[self.label_feature].to_numpy()
    #     self.clf.fit(x_train, y_train)
    #     return training_features

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

    def get_closest_terminal(self, x: float, y: float, atk_terminals: list[Terminal]) -> Terminal:
        closest_terminal = None
        closest_distance = float("inf")
        for terminal in atk_terminals:
            distance = (terminal.x - x) ** 2 + (terminal.y - y) ** 2
            if distance < closest_distance:
                closest_terminal = terminal
                closest_distance = distance
        assert closest_terminal is not None
        return closest_terminal

    def add_transaction(self, transaction: Transaction):
        self.terminals[transaction.terminal_id].add_transaction(transaction)
        self.cards[transaction.card_id].add_transaction(transaction)

    def save(self, filename: str = "cache/banksys.pkl"):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(location: str = "cache/banksys.pkl") -> "Banksys":
        with open(location, "rb") as f:
            banksys = pickle.load(f)
        return banksys

    def rollback(self, transactions: list[Transaction]):
        """
        Undo the transactions. Typically called when the environment is reset.
        """
        for transaction in transactions:
            self.terminals[transaction.terminal_id].remove(transaction)
            self.cards[transaction.card_id].remove(transaction)

    def _confusion_matrix(self, transactions: list[Transaction]):
        """
        Compute the confusion matrix for the transactions.
        """
        from sklearn.metrics import confusion_matrix

        features = []
        labels = []
        for transaction in tqdm(transactions):
            features.append(self.make_features_np(transaction, with_label=False))
            labels.append(transaction.is_fraud)
        features = np.array(features)
        labels = np.array(labels)
        df = pd.DataFrame(features, columns=self.feature_names)
        predicted_labels = self.clf.predict(df)
        return confusion_matrix(labels, predicted_labels)
