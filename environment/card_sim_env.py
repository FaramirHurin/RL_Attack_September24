from typing import Callable
from .terminal import Terminal
from .transaction import Transaction
from .action import Action
from .card_info import CardInfo
import random
import numpy as np
import numpy.typing as npt


class CardSimEnv:
    def __init__(self, classifier: Callable[[Transaction], float], t_max: float = 60 * 24 * 7, customer_location_is_known: bool = False):
        self.terminals = dict[tuple[float, float], Terminal]()
        """Dictionary of terminals indexed by (x, y) coordinates"""
        self.cards = list[CardInfo]()
        """List of all possible cards. One is randomly taken as current card at each reset."""
        self.transaction_queue = list[tuple[Transaction, Terminal]]()
        """List of transactions (ordered by timestamp) that will take place over the cours of an episode."""
        self.transaction_index = 0
        """Index of the next transaction to be processed."""
        self.classifier = classifier
        self.t = 0.0
        """Time in minutes"""
        self.current_card = self.cards[0]
        self.t_max = t_max
        self.customer_location_is_known = customer_location_is_known

    def reset(self) -> npt.NDArray[np.float32]:
        self.t = 0.0
        self.transaction_index = 0
        self.current_card = random.choice(self.cards)
        return self.get_state()

    def find_terminal(self, x: float, y: float) -> Terminal:
        """Find the terminal that is the closest to the given coordinates."""
        closest_terminal = None
        closest_distance = float("inf")
        for terminal in self.terminals.values():
            distance = (terminal.x - x) ** 2 + (terminal.y - y) ** 2
            if distance < closest_distance:
                closest_terminal = terminal
                closest_distance = distance
        assert closest_terminal is not None
        return closest_terminal

    def _perform_pending_transactions(self):
        """Perform all the transactions up to the current time step."""
        while (
            self.transaction_index < len(self.transaction_queue) and self.transaction_queue[self.transaction_index][0].timestamp <= self.t
        ):
            transaction, terminal = self.transaction_queue[self.transaction_index]
            terminal.perform_transaction(transaction)
            self.transaction_index += 1

    @property
    def hour(self) -> float:
        return (self.t // 60) % 24

    @property
    def day(self) -> int:
        return int(self.t // (60 * 24))

    def get_state(self):
        remaining_time = (self.t_max - self.t) / self.t_max
        features = [remaining_time, self.current_card.is_credit, self.hour, self.day]
        if self.customer_location_is_known:
            features += [
                self.current_card.customer_x,
                self.current_card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def make_classifier_inputs(self, transaction: Transaction) -> npt.NDArray[np.float32]:
        features = transaction.to_numpy()
        extras = [self.hour, self.day]
        terminal_features = transaction.terminal.compute_aggregated_features()
        customer_features = transaction.card.compute_aggregated_features()
        return np.concatenate([features, extras, terminal_features, customer_features])

    def step(self, action: Action) -> tuple[npt.NDArray[np.float32], float, bool]:
        """
        Perform a step in the environment, i.e.:
        - move the time forward by the delay specified in the action
        - perform all the transactions up to the current time step
        - perform the transaction specified by the action

        Returns the new state, the reward and whether the episode is done.
        """
        if self.t > self.t_max:
            raise ValueError("Cannot step past t_max: perform a reset() first.")
        self.t += 60 * (action.delay_hours + action.delay_days * 24)
        if self.t > self.t_max:
            return self.get_state(), 0.0, True
        self._perform_pending_transactions()
        terminal = self.find_terminal(action.terminal_x, action.terminal_y)
        transaction = Transaction(action.amount, self.t, terminal, action.is_online, self.current_card)
        fraud_prob = self.classifier(transaction)
        if fraud_prob > 0.5:
            done = True
            reward = 0.0
        else:
            done = False
            reward = action.amount
        return self.get_state(), reward, done

    def seed(self, seed_value: int):
        random.seed(seed_value)
