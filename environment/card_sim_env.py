from banksys import Terminal, Card
from .action import Action
import random
import numpy as np
from .step_data import StepData
import numpy.typing as npt


class CardSimEnv:
    def __init__(self, system, t_max: float = 60 * 24 * 7, customer_location_is_known: bool = False):
        self.terminals = dict[tuple[float, float], Terminal]()
        """Dictionary of terminals indexed by (x, y) coordinates"""
        self.cards = list[Card]()
        """List of all possible cards. One is randomly taken as current card at each reset."""
        from banksys import Transaction, Banksys

        self.transaction_queue = list[tuple[Transaction, Terminal]]()
        """List of transactions (ordered by timestamp) that will take place over the cours of an episode."""
        self.transaction_index = 0
        """Index of the next transaction to be processed."""
        assert isinstance(system, Banksys), "System must be an instance of Banksys"
        self.system = system
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

    def step(self, action: Action) -> tuple[npt.NDArray[np.float32], float, bool]:
        """
        Perform a step in the environment, and returns the new state, the reward and whether the episode is done.
        """
        if self.t > self.t_max:
            raise ValueError("Cannot step past t_max: perform a reset() first.")
        self.t += 60 * (action.delay_hours + action.delay_days * 24)
        if self.t > self.t_max:
            return self.get_state(), 0.0, True

        data = StepData(action, self.t, self.current_card.id)
        is_fraud = self.system.classify(data)
        if is_fraud:
            reward = 0.0
        else:
            reward = action.amount
        return self.get_state(), reward, is_fraud

    def seed(self, seed_value: int):
        random.seed(seed_value)
