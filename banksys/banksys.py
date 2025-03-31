import pandas as pd
import numpy as np
from pandas import Series

from .transaction import Transaction
from .terminal import Terminal
from .card_info import CardInfo
from environment import Action, StepData
from CardSim.cardsim import Cardsim

class Banksys:
    df: pd.DataFrame
    clf: ...
    terminals: list[Terminal]
    cards: list[CardInfo]
    training_set: pd.DataFrame

    def __init__(self, transactions_df: pd.DataFrame, card_sim: Cardsim, end_train: float, training_features: list[str],
                 label_feature: str, classifier_type: ..., days_aggregation:list):
        self.transactions_df = transactions_df
        self.end_train = end_train
        self.training_features = training_features
        self.label_feature = label_feature
        self.clf = classifier_type()

        # Select payer from card_sim.payers if payer.payer_id ids is in transactions_df['payer_id'].unique()
        self.cards = [payer for payer in card_sim.payers if payer.payer_id in transactions_df['payer_id'].unique()]

        # Select payee from card_sim.payees if payee.payee_id ids is in transactions_df['payee_id'].unique()
        self.terminals = [payee for payee in card_sim.payees if payee.payee_id in transactions_df['payee_id'].unique()]

        self.days_aggregation = days_aggregation

        self._setup()

    def _setup(self):
        # Create training set and train the classifier
        self.transactions_df = self.transactions_df.sort_values('timestamp')
        training_set = self.transactions_df[self.transactions_df['timestamp'] < self.end_train]
        X_train = training_set[self.training_features].values
        y_train = training_set[self.label_feature].values
        self.clf.fit(X_train, y_train)

        # Group transactions by card_id and terminal_id
        cards_groups = self.transactions_df.groupby('card_id')
        self.cards_transactions = {card_id: df for card_id, df in cards_groups}
        terminals_groups = self.transactions_df.groupby('terminal_id')
        self.terminals_transactions = {terminal_id: df for terminal_id, df in terminals_groups}


    def classify(self, step: StepData) -> bool:
        terminal = self.get_closest_terminal(step.terminal_x, step.terminal_y)
        card = self.cards[step.card_id]
        agg_terminal: pd.Series = self.compute_terminal_aggregated_features(terminal, step.timestamp)
        agg_card: pd.Series = self.compute_card_aggregated_features(step, step.timestamp)

        day = self.compute_day(step.timestamp)
        hour = self.compute_hour(step.timestamp)


        transaction = Transaction(
            step.amount,
            step.timestamp,
            terminal,
            step.action.is_online,
            card,
            day,
            hour
        )

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
        terminal_transactions = terminal_transactions[terminal_transactions['timestamp'] < current_time]

        # Compute aggregated features for the terminal
        for days in self.days_aggregation:
            # Select transactions from the last days
            terminal_transactions_days = terminal_transactions[terminal_transactions['timestamp'] > current_time - days]
            # Compute mean and count
            columns_names_avg[days] = terminal_transactions_days.mean()
            columns_names_count[days] = terminal_transactions_days.count()

        trx = pd.Series()
        for day in columns_names_avg.keys():
            # TODO Correct naming of columns
            trx['AVG_'+str(day)] = columns_names_avg[day]
            trx['COUNT_'+str(day)] = columns_names_count[day]
        return trx


    def compute_card_aggregated_features(self, step: StepData, current_time: float) -> pd.Series:
        columns_names_avg = {}
        columns_names_count = {}

        card_transactions = self.cards_transactions[step.card_id]
        card_transactions = card_transactions[card_transactions['timestamp'] < current_time]

        # Compute aggregated features for the card
        for days in self.days_aggregation:
            # Select transactions from the last days
            card_transactions_days = card_transactions[card_transactions['timestamp'] > current_time - days]
            # Compute mean and count
            columns_names_avg[days] = card_transactions_days.mean()
            columns_names_count[days] = card_transactions_days.count()

        trx = pd.Series()
        for day in columns_names_avg.keys():
            # TODO Correct naming of columns
            trx['AVG_'+str(day)] = columns_names_avg[day]
            trx['COUNT_'+str(day)] = columns_names_count[day]
        return trx


    def add_transaction(self, transaction: Transaction):
        self.cards_transactions[transaction.card.id] = \
            self.cards_transactions[transaction.card.id].append(transaction)
        self.terminals_transactions[transaction.terminal.id] = \
            self.terminals_transactions[transaction.terminal.id].append(transaction)
        # This allows for possible retraining of the classifier
        self.transactions_df = self.transactions_df.append(transaction)

    #TODO Check if this works correctly
    def compute_day(self, timestamp: float) -> int:
        return int(timestamp // 86400) % 7

    # TODO Check if this works correctly
    def compute_hour(self, timestamp: float) -> int:
        return int(timestamp // 3600) % 24
