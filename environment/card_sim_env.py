from banksys import Terminal, Card, Banksys, Transaction
from .action import Action
from copy import deepcopy
import random
import numpy as np
from datetime import datetime, timedelta
from .step_data import StepData
import numpy.typing as npt


class CardSimEnv:
    def __init__(self, system: Banksys, start_date: datetime, attack_duration: timedelta, customer_location_is_known: bool = False):
        self.transaction_index = 0
        """Index of the next transaction to be processed."""
        assert isinstance(system, Banksys), "System must be an instance of Banksys"
        self.system = system
        self.t_start = start_date
        self.current_time = deepcopy(start_date)
        self.current_card = self.system.cards[0]
        self.t_max = start_date + attack_duration
        self.customer_location_is_known = customer_location_is_known

    def reset(self) -> npt.NDArray[np.float32]:
        self.current_card = random.choice(self.system.cards)
        return self.get_state()

    @property
    def hour(self) -> float:
        return self.current_time.hour / 24.0

    @property
    def day(self) -> int:
        return (self.current_time - self.t_start).days

    def get_state(self):
        remaining_time = (self.t_max - self.current_time).seconds / self.t_max.second
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
        if self.current_time > self.t_max:
            raise ValueError("Cannot step past t_max: perform a reset() first.")
        self.current_time += timedelta(action.delay_days, hours=action.delay_hours)
        if self.current_time > self.t_max:
            return self.get_state(), 0.0, True

        terminal_id = self.system.get_closest_terminal(self.current_card.customer_x, self.current_card.customer_y).id
        trx = Transaction(action.amount, self.current_time, terminal_id, self.current_card.id, action.is_online)
        is_fraud = self.system.classify(trx)
        if is_fraud:
            reward = 0.0
        else:
            reward = action.amount
        return self.get_state(), reward, is_fraud

    def seed(self, seed_value: int):
        random.seed(seed_value)
