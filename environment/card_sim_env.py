import logging
import random
from copy import deepcopy
from datetime import timedelta, datetime
from .exceptions import AttackPeriodExpired
from typing import TYPE_CHECKING

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Card, Transaction

from .action import Action
from .card_registry import CardRegistry
from .priority_queue import PriorityQueue

if TYPE_CHECKING:
    from banksys import Banksys

def round_timedelta_to_minute(td: timedelta) -> timedelta:
    total_seconds = td.total_seconds()
    rounded_seconds = round(total_seconds / 60) * 60
    return timedelta(seconds=rounded_seconds)


class CardSimEnv(MARLEnv[ContinuousSpace]):
    def __init__(
        self,
        system: "Banksys",
        avg_card_block_delay: timedelta,
        *,
        customer_location_is_known: bool = False,
        normalize_location: bool = False,
    ):
        self.normalize_location = normalize_location
        if customer_location_is_known:
            obs_shape = (6,)
        else:
            obs_shape = (4,)
        action_space = ContinuousSpace(
            low=np.array([0.01] + [0.0] * 4),
            high=np.array([100_000, 200, 200, 1,  avg_card_block_delay.total_seconds() / 3600]), #avg_card_block_delay.days,
            labels=["amount", "terminal_x", "terminal_y", "is_online",  "delay_hours"], #"delay_days",
        )
        super().__init__(
            1,
            action_space=action_space,
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )
        self.system = system
        self.t = system.attack_start
        self.t_start = deepcopy(system.attack_start)
        self.card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.customer_location_is_known = customer_location_is_known
        self.action_buffer = PriorityQueue[tuple[Card, np.ndarray]]()
        logging.info(f"Attack possible from {self.system.attack_start} to {self.system.attack_end}")

    def reset(self):
        self.t = deepcopy(self.t_start)

    def spawn_card(self):
        card = self.card_registry.release_card(self.t)
        state = self.compute_state(card)
        return card, Observation(state, self.available_actions()), State(state)

    def buffer_action(self, np_action: np.ndarray, card: Card):
        action = Action.from_numpy(np_action)
        if action.delay_hours > 1:
            DEBUG = False
        delta = max(round_timedelta_to_minute(action.timedelta), timedelta(minutes=1))

        execution_time = self.t + delta
        assert execution_time >= self.t, "Action can not be executed in the past"
        self.action_buffer.push((card, np_action), execution_time)

    def get_observation(self, card: Card):
        state = self.compute_state(card)
        return Observation(state, self.available_actions())

    def get_state(self, card: Card):
        state = self.compute_state(card)
        return State(state)

    @property
    def observation_size(self):
        return self.observation_shape[0]

    def compute_state(self, card: Card):
        time_ratio = self.card_registry.get_time_ratio(card, self.t)
        features = [time_ratio, card.is_credit, self.t.hour / 24, self.t.day / 31]
        if self.customer_location_is_known:
            x, y = card.customer_x, card.customer_y
            if self.normalize_location:
                x, y = x / 200, y / 200
            features += [x, y]
        return np.array(features, dtype=np.float32)

    def step(self):
        """
        Performs the next action in the queue.
        """
        t, (card, np_action) = self.action_buffer.ppop()
        action = Action.from_numpy(np_action)
        assert t >= self.t, "Actions can not be executed in the past"
        if t >= self.system.attack_end:
            raise AttackPeriodExpired(
                f"The end date of the attack ({self.system.attack_end.isoformat()}) has been reached (current date: {t.isoformat()})"
            )
        self.t = t
        if self.normalize_location:
            action.terminal_x *= 200
            action.terminal_y *= 200
        if self.card_registry.has_expired(card, self.t):
            self.card_registry.clear(card)
            done = True
            reward = 0.0
        else:
            terminal_id = self.system.get_closest_terminal(card.customer_x, card.customer_y).id
            trx = Transaction(action.amount, self.t, terminal_id, card.id, action.is_online, is_fraud=True)
            fraud_is_detected = self.system.process_transaction(trx) or card.balance < action.amount
            if fraud_is_detected:
                reward = 0.0
            else:
                reward = action.amount
            done = fraud_is_detected
        state = self.compute_state(card)
        return card, Step(Observation(state, self.available_actions()), State(state), reward, done)

    def seed(self, seed_value: int):
        random.seed(seed_value)
