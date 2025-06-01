import logging
import random
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Card, Transaction

from .action import Action
from .card_registry import CardRegistry
from .exceptions import AttackPeriodExpired
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
        obs_size  = 4
        if customer_location_is_known:
            obs_size += 2
        if include_weekday:
            obs_size += 7

        action_space = ContinuousSpace(
            low=np.array([0.01] + [0.0] * 4),
            high=np.array([100_000, 200, 200, 1,  avg_card_block_delay.total_seconds() / 3600]), #avg_card_block_delay.days,
            labels=["amount", "terminal_x", "terminal_y", "is_online",  "delay_hours"], #"delay_days",
        )
        super().__init__(
            1,
            action_space=action_space,
            observation_shape=(obs_size,),
            state_shape=(obs_size,),
        )
        self.system = system
        self.t = system.attack_start
        self.t_start = deepcopy(system.attack_start)
        self.card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.customer_location_is_known = customer_location_is_known
        self.include_weekday = include_weekday
        self.action_buffer = PriorityQueue[tuple[Card, np.ndarray]]()
        logging.info(f"Attack possible from {self.system.attack_start} to {self.system.attack_end}")

    def reset(self):
        self.t = deepcopy(self.t_start)
        self.t = deepcopy(self.t_start)

    def spawn_card(self):
        card = self.card_registry.release_card(self.t)
        try:
            state = self.compute_state(card)
        except:
            card.attempted_attacks = 0
            state = self.compute_state(card)

        return card, Observation(state, self.available_actions()), State(state)

    def buffer_action(self, np_action: np.ndarray, card: Card):
        action = Action.from_numpy(np_action)
        card.attempted_attacks += 1

        # execution_time = action.hour + one minute to self.t
        delta = timedelta(hours=action.delay_hours) + timedelta(minutes=1)


        execution_time = card.current_time + delta
        card.set_current_time(execution_time)

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
        features = [card.attempted_attacks, time_ratio, card.is_credit, card.current_time.hour / 24] #, self.t.day / 31
        if self.include_weekday:
            one_hot_weekday = [0.0] * 7
            one_hot_weekday[self.t.weekday()] = 1.0
            features += one_hot_weekday
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
        if action.delay_hours == 0:
            debug = 0
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
            trx = None
        else:
            terminal_id = self.system.get_closest_terminal(card.customer_x, card.customer_y).id
            trx = Transaction(action.amount, self.t, terminal_id, card.id, action.is_online, is_fraud=True)
            fraud_is_detected, cause_of_detection = self.system.process_transaction(trx) or card.balance < action.amount
            state = self.compute_state(card)
            if fraud_is_detected or card.balance < action.amount:
                reward = 10 * ( action.delay_hours - 1 ) if action.delay_hours < 1 else 0
            else:
                reward = action.amount * ((state[0]) ** 0.5)
                card.remove_money(action.amount)
            done = fraud_is_detected

        state = self.compute_state(card)
        return card, Step(Observation(state, self.available_actions()), State(state), reward, done, info={"trx": trx})

    def seed(self, seed_value: int):
        random.seed(seed_value)
