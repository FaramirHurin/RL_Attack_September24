import pandas as pd
import numpy as np
from .transaction import Transaction
from .terminal import Terminal
from .card_info import CardInfo
from environment import Action, StepData


class Banksys:
    df: pd.DataFrame
    clf: ...
    terminals: list[Terminal]
    cards: list[CardInfo]

    def __init__(self, df: pd.DataFrame, end_train: float):
        self.df = df
        self.cards = []
        self.terminals = []
        self._setup()

    def _setup(self):
        """
        Compute the dictionary of terminals and cards.
        """

    def classify(self, step: StepData) -> bool:
        terminal = self.get_closest_terminal(step.terminal_x, step.terminal_y)
        card = self.cards[step.card_id]
        agg_terminal = self.compute_terminal_aggregated_features(terminal, step.timestamp)
        agg_card = self.compute_card_aggregated_features(step, step.timestamp)

        transaction = Transaction(
            step.amount,
            step.timestamp,
            terminal,
            step.action.is_online,
            card,
        )
        # Somehow add the current day of week and the hour
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

    def compute_terminal_aggregated_features(self, terminal: Terminal, current_time: float) -> pd.DataFrame:
        """
        Compute aggregated features for the terminal, i.e. consider all the transaction before the current timestamp.
        """
        raise NotImplementedError()

    def compute_card_aggregated_features(self, action: Action, current_time: float) -> pd.DataFrame:
        raise NotImplementedError()

    def perform_pending_transactions(self, timestamp: float):
        """Perform all the transactions up to the current time step."""
        pass

    def add_transaction(self, transaction: Transaction):
        """
        Add the transaction to the list of transactions.
        """
        # Add the transaction to the transactions of the user and of the concerned terminal
