import logging
import random
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Card, Transaction
from exceptions import AttackPeriodExpired, InsufficientFundsError

from .action import Action
from .card_registry import CardRegistry
from .priority_queue import PriorityQueue

if TYPE_CHECKING:
    from banksys import Banksys


class CardSimEnv(MARLEnv[ContinuousSpace]):
    def __init__(
        self,
        system: "Banksys",
        avg_card_block_delay: timedelta,
        *,
        include_weekday: bool = True,
        customer_location_is_known: bool = False,
        normalize_location: bool = False,
    ):
        self.normalize_location = normalize_location
        obs_size = 4
        if customer_location_is_known:
            obs_size += 2
        if include_weekday:
            obs_size += 7

        action_space = ContinuousSpace(
            low=np.array([0.01] + [0.0] * 4),
            high=np.array([100_000, 200, 200, 1, avg_card_block_delay.total_seconds() / 3600]),  # avg_card_block_delay.days,
            labels=["amount", "terminal_x", "terminal_y", "is_online", "delay_hours"],  # "delay_days",
        )
        super().__init__(
            1,
            action_space=action_space,
            observation_shape=(obs_size,),
            state_shape=(obs_size,),
        )
        self.system = system
        self.card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.customer_location_is_known = customer_location_is_known
        self.include_weekday = include_weekday
        self.action_buffer = PriorityQueue[tuple[Card, np.ndarray]]()
        logging.info(f"Attack possible from {self.system.attack_start} to {self.system.attack_end}")

    def reset(self):
        self.card_registry.reset(self.system.cards)
        self.action_buffer.clear()
        return

    def spawn_card(self):
        card = self.card_registry.release_card(self.t)
        state = self.compute_state(card)
        return card, Observation(state, self.available_actions()), State(state)

    def buffer_action(self, np_action: np.ndarray, card: Card):
        action = Action.from_numpy(np_action)
        card.attempted_attacks += 1
        # delta = timedelta(hours=action.delay_hours) + timedelta(minutes=1)
        execution_time = self.t + timedelta(hours=action.delay_hours)
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
        time_ratio = self.card_registry.get_remaining_time_ratio(card, self.t)
        features = [card.attempted_attacks, time_ratio, card.is_credit, self.t.hour / 24]
        if self.include_weekday:
            one_hot_weekday = [0.0] * 7
            one_hot_weekday[self.t.weekday()] = 1.0
            features += one_hot_weekday
        if self.customer_location_is_known:
            x, y = card.x, card.y
            if self.normalize_location:
                x, y = x / 200, y / 200
            features += [x, y]
        return np.array(features, dtype=np.float32)

    @property
    def t(self):
        return self.system.current_time

    def step(self):
        """
        Performs the next action in the queue.
        """
        t, (card, np_action) = self.action_buffer.ppop()
        action = Action.from_numpy(np_action)
        if t >= self.system.attack_end:
            raise AttackPeriodExpired(
                f"The end date of the attack ({self.system.attack_end.isoformat()}) has been reached (current date: {t.isoformat()})"
            )
        info = dict[str, Any](t=t.isoformat())
        if self.normalize_location:
            action.terminal_x *= 200
            action.terminal_y *= 200
        if self.card_registry.has_expired(card, t):
            self.card_registry.clear(card)
            reward = 0.0
            done = True
            info["expired"] = True
            self.system.simulate_until(t)
        else:
            terminal_id = self.system.get_closest_terminal(card.x, card.y).id
            trx = Transaction(action.amount, t, terminal_id, card.id, action.is_online, is_fraud=True)
            try:
                self.system.process_transaction(trx)
                transaction_denied = False
                if trx.fraud_is_detected:
                    info |= self.system.clf.get_details().to_dicts()[0]
                    transaction_denied = True
            except InsufficientFundsError:
                info["insufficient_funds"] = True
                transaction_denied = True
            done = trx.fraud_is_detected
            reward = 0.0 if transaction_denied else trx.amount
        state = self.compute_state(card)
        return card, Step(Observation(state, self.available_actions()), State(state), reward, done, info=info), np_action

    def seed(self, seed_value: int):
        random.seed(seed_value)
