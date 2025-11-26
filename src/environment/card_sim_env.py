import hashlib
import logging
import random
from dataclasses import astuple
from typing import TYPE_CHECKING, Any

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Payer, Terminal, Transaction
from exceptions import AttackPeriodExpired, InsufficientFundsError

from .action import Action
from .payer_registry import PayerRegistry
from .priority_queue import PriorityQueue

if TYPE_CHECKING:
    from banksys import Banksys
    from parameters.env_parameters import EnvParameters


class CardSimEnv(MARLEnv[ContinuousSpace]):
    def __init__(
        self,
        system: "Banksys",
        params: "EnvParameters",
    ):
        self.normalize_location = params.normalize_location
        obs_size = 6  # time_ratio, hour_of_day, total_stolen, n_frauds, latest_fraud_amount, sufficient_funds
        if params.know_client:  # x, y
            obs_size += 2
        if params.include_weekday:  # one-hot weekday
            obs_size += 7

        low = [0.01] + [0.0] * 4
        high = [1_000, 200, 200, 1, params.avg_card_block_delay.total_seconds() / 3600]
        labels = ["amount", "terminal_x", "terminal_y", "is_online", "delay_hours"]
        if params.can_choose_debit_credit:
            low += [0]
            high += [1]
            labels += ["is_credit"]
        super().__init__(
            1,
            action_space=ContinuousSpace(low, high, labels),
            observation_shape=(obs_size,),
            state_shape=(obs_size,),
        )
        self.attackable_terminals = random.sample(system.terminals, round(len(system.terminals) * params.terminal_fract))
        self.system = system
        self.payer_registry = PayerRegistry(system.payers, params.avg_card_block_delay)
        self.customer_location_is_known = params.customer_location_is_known
        self.include_weekday = params.include_weekday
        self.action_buffer = PriorityQueue[tuple[Payer, np.ndarray]]()
        logging.info(f"Attack possible from {self.system.attack_start} to {self.system.attack_end}")

    def reset(self):
        self.payer_registry.reset()
        self.action_buffer.clear()

    def spawn_card(self):
        card = self.payer_registry.release_payer(self.t)
        state = self.compute_state(card)
        return card, Observation(state, self.available_actions()), State(state)

    def buffer_action(self, np_action: np.ndarray, card: Payer):
        action = Action.from_numpy(np_action)
        execution_time = self.t + action.timedelta
        self.action_buffer.push((card, np_action), execution_time)

    def get_observation(self, card: Payer):
        state = self.compute_state(card)
        return Observation(state, self.available_actions())

    def get_state(self, card: Payer):
        state = self.compute_state(card)
        return State(state)

    @property
    def observation_size(self):
        return self.observation_shape[0]

    @property
    def isodate(self):
        return self.t.date().isoformat()

    def compute_state(self, payer: Payer):
        features = [self.t.hour / 24, *self.payer_registry.get_features(payer, self.t)]
        if self.include_weekday:
            one_hot_weekday = [0.0] * 7
            one_hot_weekday[self.t.weekday()] = 1.0
            features += one_hot_weekday
        if self.customer_location_is_known:
            x, y = payer.x, payer.y
            if self.normalize_location:
                x, y = x / 200, y / 200
            features += [x, y]
        return np.array(features, dtype=np.float32)

    def get_closest_terminal(self, x: float, y: float) -> Terminal:
        closest_terminal = None
        closest_distance = float("inf")
        for terminal in self.attackable_terminals:
            distance = (terminal.x - x) ** 2 + (terminal.y - y) ** 2
            if distance < closest_distance:
                closest_terminal = terminal
                closest_distance = distance
        assert closest_terminal is not None
        return closest_terminal

    @property
    def t(self):
        return self.system.current_time

    def step(self):
        """
        Performs the next action in the queue.
        """
        t, (payer, np_action) = self.action_buffer.ppop()
        action = Action.from_numpy(np_action)
        if t >= self.system.attack_end:
            raise AttackPeriodExpired(f"The end date of the attack ({self.system.attack_end.isoformat()}) has been reached")
        info = dict[str, Any](t=t.isoformat())
        if self.normalize_location:
            action.terminal_x *= 200
            action.terminal_y *= 200
        if self.payer_registry.has_expired(payer, t):
            self.payer_registry.clear(payer)
            reward = 0.0
            done = True
            info["expired"] = True
        else:
            self.system.simulate_until(t)
            trx = Transaction(
                amount=action.amount,
                timestamp=t,
                terminal_id=self.get_closest_terminal(payer.x, payer.y).id,
                payer_id=payer.id,
                is_online=action.is_online,
                is_credit=False,  # action.is_credit,
                is_fraud=True,
            )
            try:
                self.system.process_transaction(trx)
                if trx.fraud_is_detected:
                    info |= self.system.clf.get_details().to_dicts()[0]
                    reward = 0.0
                    self.payer_registry.clear(payer)
                else:
                    self.payer_registry.notify_transaction_processed(trx)
                    reward = trx.amount
            except InsufficientFundsError:
                info["insufficient_funds"] = True
                reward = 0.0
                self.payer_registry.notify_insufficient_funds(trx)
            done = trx.fraud_is_detected
        state = self.compute_state(payer)
        return payer, Step(Observation(state, self.available_actions()), State(state), reward, done, info=info), np_action

    def seed(self, seed_value: int):
        random.seed(seed_value)

    def sha256(self):
        return hashlib.sha256(str(astuple(self)).encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        h = self.sha256()
        return int(h, 16)
