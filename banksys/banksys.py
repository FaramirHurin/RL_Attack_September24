import pandas as pd
import numpy as np

from .transaction import Transaction
from .terminal import Terminal
from .card import Card
from .classification import ClassificationSystem
from environment import StepData


class Banksys:
    df: pd.DataFrame
    clf: ClassificationSystem
    terminals: list[Terminal]
    cards: list[Card]
    training_set: pd.DataFrame

    def __init__(
        self,
        clf: ClassificationSystem,
        cards: list[Card],
        terminals: list[Terminal],
        transactions: list[Transaction],
        is_fraud: list[bool],
        train_split: float = 0.8,
    ):
        self.clf = clf

        self.cards = cards
        self.terminals = terminals
        self._simulate(transactions)
        self._setup()

    def _simulate(self, transactions: list[Transaction]):
        """
        Args:
            transactions: transactions sorted by timestamp
        """
        # Add the transactions for the 30 first days
        start = transactions[0].timestamp
        i = 0
        t = transactions[i]
        ndays_warmup = max(*self.cards[0].days_aggregation, *self.terminals[0].days_aggregation)

        while i < len(transactions) and (t.timestamp - start).days < ndays_warmup:
            self.terminals[t.terminal_id].add_transaction(t)
            self.cards[t.card_id].add_transaction(t)
            i += 1
            t = transactions[i]

        rows = []
        for t in transactions[i:]:
            self.terminals[t.terminal_id].add_transaction(t)
            self.cards[t.card_id].add_transaction(t)
            card_features = self.cards[t.card_id].features(t.timestamp)
            terminal_features = self.terminals[t.terminal_id].features(t.timestamp)
            features = np.concatenate([t.features, card_features, terminal_features])
            rows.append(features)
        df = pd.DataFrame(rows, columns=Transaction.FEATURE_NAMES + self.cards[0].feature_names + self.terminals[0].feature_names)
        print(df.head())

    def _setup(self):
        # Create training set and train the classifier
        self.transactions_df = self.transactions_df.sort_values("timestamp")
        training_set = self.transactions_df[self.transactions_df["timestamp"] < self.end_train]
        # Remove id etc
        x_train = training_set[self.training_features].values
        y_train = training_set[self.label_feature].values
        self.clf.fit(x_train, y_train)

        # Group transactions by card_id and terminal_id
        cards_groups = self.transactions_df.groupby("card_id")
        self.cards_transactions = {card_id: df for card_id, df in cards_groups}
        terminals_groups = self.transactions_df.groupby("terminal_id")
        self.terminals_transactions = {terminal_id: df for terminal_id, df in terminals_groups}

    def classify(self, step: StepData) -> bool:
        # TODO: rewrite this classifiction logic such that the terminal/card is responsible for that
        terminal = self.get_closest_terminal(step.terminal_x, step.terminal_y)
        card = self.cards[step.card_id]
        agg_terminal: pd.Series = self.compute_terminal_aggregated_features(terminal, step.timestamp)
        agg_card: pd.Series = self.compute_card_aggregated_features(step, step.timestamp)

        day = self.compute_day(step.timestamp)
        hour = self.compute_hour(step.timestamp)

        transaction = Transaction(step.amount, step.timestamp, terminal, step.action.is_online, card, day, hour)

        features = np.concatenate([transaction.to_numpy(), agg_terminal, agg_card])
        label = self.clf.predict(features)
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

    def add_transaction(self, transaction: Transaction):
        self.cards_transactions[transaction.card.id] = self.cards_transactions[transaction.card.id].append(transaction)
        self.terminals_transactions[transaction.terminal.id] = self.terminals_transactions[transaction.terminal.id].append(transaction)
        # This allows for possible retraining of the classifier
        self.transactions_df = self.transactions_df.append(transaction)

    # TODO Check if this works correctly
    def compute_day(self, timestamp: float) -> int:
        return int(timestamp // 86400) % 7

    # TODO Check if this works correctly
    def compute_hour(self, timestamp: float) -> int:
        return int(timestamp // 3600) % 24
