import time

import pandas as pd
import numpy as np
import datetime
from transaction import Transaction
from terminal import Terminal
from card import Card
from classification import ClassificationSystem
from environment import StepData


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
        week_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.feature_names = ["amount", "hour_ratio"]+ week_days+ ["is_online"] +\
                             self.cards[0].feature_names + \
                             self.terminals[0].feature_names
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

    def _setup(self, train_split:float):
        # Split the data into training and testing sets
        tr_size = int(self.transactions_df.shape[0] * train_split)
        training_set = self.transactions_df.iloc[:tr_size, :]
        testing_set = self.transactions_df.iloc[tr_size:, :]

        # Define the features and label
        self.training_features = [col for col in training_set.columns if col != "label"]
        self.label_feature = "label"
        x_train = training_set[self.training_features].values
        y_train = training_set[self.label_feature].values
        self.clf.fit(x_train, y_train)

        x_test = testing_set[self.training_features]
        y_test = testing_set[self.label_feature].values

        # Evaluate the classifier
        y_pred = self.clf.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy: {accuracy:.2f}")


    def classify(self, step: StepData) -> bool:
        terminal = self.get_closest_terminal(step.terminal_x, step.terminal_y)
        card = self.cards[step.card_id]
        timestamp = step.to_stamp()
        terminal_features = terminal.features(timestamp)
        card_features = card.features(timestamp)

        # Create Transaction object
        transaction = Transaction(step.amount, timestamp, terminal.id, card.id,  step.action.is_online)

        # Compute the features for the transaction
        day_of_week = transaction.day_of_week
        hour = transaction.hour_ratio

        # Concatenate all features
        features = np.concatenate([transaction.amount,hour, day_of_week,
                                   transaction.is_online, terminal_features, card_features])

        trx = pd.Series(features, index=self.feature_names)

        label = self.clf.predict(trx)
        transaction.label = label

        terminal.add_transaction(transaction)
        card.add_transaction(transaction)
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
        self.transactions_df = self.transactions_df.append(transaction)

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
