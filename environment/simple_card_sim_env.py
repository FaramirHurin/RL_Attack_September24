import random
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Transaction

from .action import Action
from .card_registry import CardRegistry

if TYPE_CHECKING:
    from banksys import Banksys


class SimpleCardSimEnv(MARLEnv[ContinuousSpace]):
    def __init__(
        self,
        system: "Banksys",
        avg_card_block_delay: timedelta = timedelta(days=7),
        card_registry: Optional[CardRegistry] = None,
        *,
        customer_location_is_known: bool = False,
        normalize_location: bool = False,
    ):
        """
        Args:
            system: The Banksys object to be used for the simulation.
            card_block_delay: The average time delay before a card can be blocked.
            n_parallel: The number of parallel transactions to be processed.
            customer_location_is_known: Whether the customer's location is known.
        """
        self.normalize_location = normalize_location
        if customer_location_is_known:
            obs_shape = (6,)
        else:
            obs_shape = (4,)
        action_space = ContinuousSpace(
            low=np.array([0.01] + [0.0] * 5),
            high=np.array([100_000, 200, 200, 1, avg_card_block_delay.days, avg_card_block_delay.total_seconds() / 3600]),
            labels=["amount", "terminal_x", "terminal_y", "is_online", "delay_days", "delay_hours"],
        )
        super().__init__(
            1,
            action_space=action_space,
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )
        self.system = system
        self.t = deepcopy(system.attack_start)
        self.t_start = deepcopy(system.attack_start)
        if card_registry is None:
            card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.card_registry = card_registry
        self.customer_location_is_known = customer_location_is_known
        self.current_card = self.card_registry.release_card(self.t)
        # self.transactions = list[Transaction]()

    def reset(self):
        # self.system.rollback(self.transactions)
        # self.transactions = []
        # self.t = deepcopy(self.t_start)
        self.current_card = self.card_registry.release_card(self.t)
        obs = self.get_observation()
        state = self.get_state()
        return obs, state

    def get_observation(self):
        state = self.compute_state()
        return Observation(state, self.available_actions())

    def get_state(self):
        state = self.compute_state()
        return State(state)

    @property
    def observation_size(self):
        return self.observation_shape[0]

    def compute_state(self):
        time_ratio = self.card_registry.get_time_ratio(self.current_card, self.t)
        features = [time_ratio, self.current_card.is_credit, self.t.hour, self.t.day]
        if self.customer_location_is_known:
            x, y = self.current_card.customer_x, self.current_card.customer_y
            if self.normalize_location:
                x, y = x / 200, y / 200
            features += [x, y]
        return np.array(features, dtype=np.float32)

    def step(self, np_action: np.ndarray):
        action = Action.from_numpy(np_action)
        if self.normalize_location:
            action.terminal_x *= 200
            action.terminal_y *= 200
        self.t += action.timedelta
        assert self.t <= self.system.attack_end, f"Simulation time {self.t} exceeds attack end time {self.system.attack_end}"
        if self.card_registry.has_expired(self.current_card, self.t):
            self.card_registry.clear(self.current_card)
            done = True
            reward = 0.0
            trx = None
        else:
            terminal_id = self.system.get_closest_terminal(self.current_card.customer_x, self.current_card.customer_y).id
            trx = Transaction(action.amount, self.t, terminal_id, self.current_card.id, action.is_online, is_fraud=True)
            fraud_is_detected = self.system.process_transaction(trx) or self.current_card.balance < action.amount
            self.transactions.append(trx)
            if fraud_is_detected:
                reward = 0.0
            else:
                reward = action.amount
                self.current_card.balance -= action.amount
            done = fraud_is_detected
        state = self.compute_state()
        return Step(Observation(state, self.available_actions()), State(state), reward, done), trx

    def seed(self, seed_value: int):
        random.seed(seed_value)
