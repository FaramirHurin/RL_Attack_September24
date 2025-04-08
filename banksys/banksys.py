import datetime

import numpy as np
import pandas as pd

from .card import Card
from .classification import ClassificationSystem
from .terminal import Terminal
from .transaction import Transaction


class Banksys:
    def __init__(
        self,
        clf: ClassificationSystem,
        cards: list[Card],
        terminals: list[Terminal],
        transactions: list[Transaction],
        train_split: float = 0.9,
    ):
        # Sort transactions by timestamp
        transactions = sorted(transactions, key=lambda t: t.timestamp)
        self.clf = clf
        self.cards = cards
        self.terminals = terminals
        self.label_feature = "label"
        week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        self.feature_names = (
            ["amount", "hour_ratio"] + week_days + ["is_online"] + self.cards[0].feature_names + self.terminals[0].feature_names
        )
        self._create_df_and_aggregate(transactions)
        self._setup(train_split)

    def _create_df_and_aggregate(self, transactions: list[Transaction]):
        start = transactions[0].timestamp
        ndays_warmup = max(*self.cards[0].days_aggregation, *self.terminals[0].days_aggregation)

        rows = []
        agg_count = 0
        for t in transactions:
            self.terminals[t.terminal_id].add_transaction(t)
            self.cards[t.card_id].add_transaction(t)
            if t.timestamp - start > ndays_warmup:
                agg_count += 1
                if agg_count % 1000 == 0:
                    print(f"Aggregated {agg_count} transactions" + str(datetime.datetime.now()))
                card_features = self.cards[t.card_id].features(t.timestamp)
                terminal_features = self.terminals[t.terminal_id].features(t.timestamp)
                features = np.concatenate([t.features, card_features, terminal_features, [t.label]])
                rows.append(features)
        self.transactions_df = pd.DataFrame(rows, columns=self.feature_names + ["label"])

    def _setup(self, train_split: float):
        # Split the data into training and testing sets
        tr_size = int(self.transactions_df.shape[0] * train_split)
        training_set = self.transactions_df.iloc[:tr_size, :]
        testing_set = self.transactions_df.iloc[tr_size:, :]

        # Define the features and label
        self.training_features = [col for col in training_set.columns if col != "label"]
        x_train = training_set[self.training_features]
        y_train = training_set[self.label_feature].to_numpy()
        self.clf.fit(x_train, y_train)

        x_test = testing_set[self.training_features]
        y_test = testing_set[self.label_feature].values

        # Evaluate the classifier
        y_pred = self.clf.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.2f}")

    def _make_features(self, transaction: Transaction, with_label: bool) -> np.ndarray:
        terminal = self.terminals[transaction.terminal_id]
        card = self.cards[transaction.card_id]
        terminal_features = terminal.features(transaction.timestamp)
        card_features = card.features(transaction.timestamp)

        if with_label:
            return np.concatenate([transaction.features, terminal_features, card_features, [transaction.label]])
        return np.concatenate([transaction.features, terminal_features, card_features])

    def classify(self, transaction: Transaction) -> bool:
        trx_features = self._make_features(transaction, with_label=False).reshape(1, -1)
        trx = pd.DataFrame(trx_features, columns=self.feature_names)
        transaction.label = self.clf.predict(trx).item()  # type: ignore
        self._add_transaction(transaction)
        return transaction.label

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

    def _add_transaction(self, transaction: Transaction):
        # Add the transaction to the dataframe self.transactions_df without using append
        features = self._make_features(transaction, with_label=True)
        self.transactions_df.loc[len(self.transactions_df)] = features
        self.terminals[transaction.terminal_id].add_transaction(transaction)
        self.cards[transaction.card_id].add_transaction(transaction)

    """
    def compute_terminal_aggregated_features(self, terminal: Terminal, current_time: float) -> pd.Series:
        columns_names_avg = {}
        columns_names_count = {}

        terminal_transactions = self.terminals_transactions[terminal.id]
        terminal_transactions = terminal_transactions[terminal_transactions["timestamp"] < current_time]

        # Compute aggregated features for the terminal
        for days in self.days_aggregation:
            # Select transactions from the last days
            terminal_transactions_days = terminal_transactions[terminal_transactions["timestamp"] > current_time - days]
            # Compute mean and count
            columns_names_avg[days] = terminal_transactions_days.mean()
            columns_names_count[days] = terminal_transactions_days.count()

        trx = pd.Series()
        for day in columns_names_avg.keys():
            # TODO Correct naming of columns
            trx["AVG_" + str(day)] = columns_names_avg[day]
            trx["COUNT_" + str(day)] = columns_names_count[day]
        return trx

    def compute_card_aggregated_features(self, step: StepData, current_time: float) -> pd.Series:
        columns_names_avg = {}
        columns_names_count = {}

        card_transactions = self.cards_transactions[step.card_id]
        card_transactions = card_transactions[card_transactions["timestamp"] < current_time]

        # Compute aggregated features for the card
        for days in self.days_aggregation:
            # Select transactions from the last days
            card_transactions_days = card_transactions[card_transactions["timestamp"] > current_time - days]
            # Compute mean and count
            columns_names_avg[days] = card_transactions_days.mean()
            columns_names_count[days] = card_transactions_days.count()

        trx = pd.Series()
        for day in columns_names_avg.keys():
            # TODO Correct naming of columns
            trx["AVG_" + str(day)] = columns_names_avg[day]
            trx["COUNT_" + str(day)] = columns_names_count[day]
        return trx
    """

    """
    clf: ClassificationSystem
    terminals: list[Terminal]
    cards: list[Card]
    training_set: pd.DataFrame
    tramsactions_df: pd.DataFrame
    """
