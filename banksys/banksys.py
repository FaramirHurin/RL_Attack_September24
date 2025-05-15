from datetime import datetime
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from .card import Card
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction

# TODO: when a card is blocked, we should remove all its future transactions


class Banksys:
    def __init__(
        self,
        inner_clf,
        anomaly_detection_clf,
        cards: list[Card],
        terminals: list[Terminal],
        t_start: datetime,
        transactions: list[Transaction],
        feature_names: list[str] = None,
        quantiles: list[float] = None,
        rules: list = None,

    ):

        # Sort transactions by timestamp
        self.transactions = sorted(transactions, key=lambda t: t.timestamp)
        self.clf = ClassificationSystem(banksys=self, clf=inner_clf, anomaly_detection_clf=anomaly_detection_clf,
        features_for_quantiles=feature_names,
                                        quantiles=quantiles, rules=rules)

        # self.transactions = sorted(transactions, key=lambda t: t.timestamp)
        n_days_warmup = max(*cards[0].days_aggregation, *terminals[0].days_aggregation)
        self.earliest_attackable_moment = t_start + n_days_warmup
        self.cards = cards
        self.terminals = terminals
        self.label_feature = "label"
        week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        self.feature_names = (
            ["amount", "hour_ratio"] + week_days + ["is_online"] + self.cards[0].feature_names
            + self.terminals[0].feature_names
        )

    def _create_df_and_aggregate(self, transactions: list[Transaction]):
        start = transactions[0].timestamp
        ndays_warmup = max(*self.cards[0].days_aggregation, *self.terminals[0].days_aggregation)
        rows = []
        for t in tqdm(transactions):
            self.add_transaction(t)
            if t.timestamp - start > ndays_warmup:
                features = self._make_features(t, with_label=True)
                rows.append(features)
        return pd.DataFrame(rows, columns=self.feature_names + ["label"])

    def train_classifier(self, transactions: list[Transaction], train_split: float = 0.9):
        transactions_df = self._create_df_and_aggregate(transactions)
        # Split the data into training and testing sets
        tr_size = int(len(transactions_df) * train_split)
        training_set = transactions_df.iloc[:tr_size, :]
        testing_set = transactions_df.iloc[tr_size:, :]
        # Define the features and label
        self.training_features = [col for col in training_set.columns if col != "label"]
        x_train = training_set[self.training_features]
        y_train = training_set[self.label_feature].to_numpy()
        self.clf.fit(x_train, y_train)
        return testing_set


    #TODO Allow rules to work on the whole test set to test baselines
    """ 
    def evaluate_classifier(self, testing_set: pd.DataFrame):
        x_test = testing_set[self.training_features]
        y_test = testing_set[self.label_feature].values

        # Evaluate the classifier
        y_pred = self.clf.predict(x_test, )
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.2f}")
        # Print classifier feature importances
        if hasattr(self.clf.ml_classifier, "feature_importances_"):
            feature_importances = self.clf.ml_classifier.feature_importances_
            print("Feature importances:")
            for name, importance in zip(self.training_features, feature_importances):
                print(f"{name}: {importance:.4f}")
    """

    def _make_features(self, transaction: Transaction, with_label: bool) -> np.ndarray:
        terminal = self.terminals[transaction.terminal_id]
        card = self.cards[transaction.card_id]
        terminal_features = terminal.features(transaction.timestamp)
        card_features = card.features(transaction.timestamp)

        if with_label:
            assert transaction.label is not None, "Label must be set for the transaction used in the agredated features"
            return np.concatenate([transaction.features, terminal_features, card_features, [transaction.label]])
        return np.concatenate([transaction.features, terminal_features, card_features])

    def process_transaction(self, transaction: Transaction) -> bool:
        """
        Process the transaction and return whether it is fraudulent or not.
        """
        trx_features = self._make_features(transaction, with_label=False).reshape(1, -1)
        trx = pd.DataFrame(trx_features, columns=self.feature_names)
        label = self.clf.predict(trx, transaction)
        transaction.label = label
        self.add_transaction(transaction)
        return label

    def get_closest_terminal(self, x: float, y: float) -> Terminal:
        closest_terminal = None
        closest_distance = float("inf")
        for terminal in self.terminals:
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
