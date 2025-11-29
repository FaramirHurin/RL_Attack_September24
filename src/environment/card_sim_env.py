import random
from typing import TYPE_CHECKING, Any

import numpy as np
from marlenv import ContinuousSpace, MARLEnv, Observation, State, Step

from banksys import Payer, Terminal, Transaction
from exceptions import AttackPeriodExpired, InsufficientFundsError
from utils import tb_log

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
        self.attackable_terminals = random.sample(system.terminals, round(len(system.terminals) * params.terminal_fract))
        self.system = system
        self.payer_registry = PayerRegistry(system.payers, params.avg_block_delay, system.attack_start)
        self.customer_location_is_known = params.customer_location_is_known
        self.include_weekday = params.include_weekday
        self.action_buffer = PriorityQueue[tuple[Payer, np.ndarray]]()
        self.scale_amount = params.scale_amount
        obs = self.compute_state(system.payers[0])
        low = [0.01] + [0.0] * 4
        high = [1_000, 200, 200, 1, params.avg_block_delay.total_seconds() / 3600]
        labels = ["amount", "terminal_x", "terminal_y", "is_online", "delay_hours"]
        if params.can_choose_debit_credit:
            low += [0]
            high += [1]
            labels += ["is_credit"]
        super().__init__(
            1,
            action_space=ContinuousSpace(low, high, labels),
            observation_shape=obs.shape,
            state_shape=obs.shape,
        )

    def reset(self):
        self.payer_registry.reset()
        self.action_buffer.clear()

    def spawn_payer(self):
        payer = self.payer_registry.release_payer(self.t)
        state = self.compute_state(payer)
        return payer, Observation(state, self.available_actions()), State(state)

    def buffer_action(self, np_action: np.ndarray, payer: Payer):
        action = Action.from_numpy(np_action)
        execution_time = self.t + action.timedelta
        self.action_buffer.push((payer, np_action), execution_time)

    def get_observation(self, payer: Payer):
        state = self.compute_state(payer)
        return Observation(state, self.available_actions())

    def get_state(self, payer: Payer):
        state = self.compute_state(payer)
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
            x, y = payer.x / 200, payer.y / 200
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

    @property
    def elapsed_time(self):
        return self.system.current_time - self.system.attack_start

    def step(self):
        """
        Performs the next action in the queue.
        """
        t, (payer, np_action) = self.action_buffer.ppop()
        action = Action.from_numpy(np_action).denormalized(self.scale_amount)
        if t >= self.system.attack_end:
            raise AttackPeriodExpired(f"The end date of the attack ({self.system.attack_end.isoformat()}) has been reached")
        tb_log("env/action", action.as_dict(), self.elapsed_time)
        info = dict[str, Any](t=t.isoformat())
        if self.payer_registry.has_expired(payer, t):
            self.payer_registry.clear(payer)
            reward = 0.0
            done = True
            info["expired"] = True
        else:
            trx = Transaction(
                amount=action.amount,
                timestamp=t,
                terminal_id=self.get_closest_terminal(action.terminal_x, action.terminal_y).id,
                payer_id=payer.id,
                is_online=action.is_online,
                is_credit=False,
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
        tb_log("env/reward", reward, self.elapsed_time)
        return payer, Step(Observation(state, self.available_actions()), State(state), reward, done, info=info), np_action
