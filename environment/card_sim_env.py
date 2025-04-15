from banksys import Banksys, Transaction
from .action import Action
from copy import deepcopy
import random
import numpy as np
from datetime import timedelta
from marlenv import Observation, Step, MARLEnv, ContinuousActionSpace, State


class CardSimEnv(MARLEnv[Action, ContinuousActionSpace]):
    def __init__(self, system: Banksys, attack_duration: timedelta, *, customer_location_is_known: bool = False):
        if customer_location_is_known:
            obs_shape = (6,)
        else:
            obs_shape = (4,)
        super().__init__(
            ContinuousActionSpace(
                1,
                low=[0] * 6,
                high=[100_000, 200, 200, 1, attack_duration.days, attack_duration.seconds / 3600],
                action_names=["amount", "terminal_x", "terminal_y", "is_online", "delay_days", "delay_hours"],
            ),
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )

        self.transaction_index = 0
        """Index of the next transaction to be processed."""
        assert isinstance(system, Banksys), "System must be an instance of Banksys"
        self.system = system
        self.t_start = system.earliest_attackable_moment
        self.current_time = deepcopy(self.t_start)
        self.current_card = self.system.cards[0]
        self.t_max = self.t_start + attack_duration
        self.customer_location_is_known = customer_location_is_known

    def reset(self):
        self.current_card = random.choice(self.system.cards)
        self.current_time = deepcopy(self.t_start)
        return self.get_observation(), self.get_state()

    def get_observation(self) -> Observation:
        state = self.compute_state()
        return Observation(state, self.available_actions())

    @property
    def transactions(self):
        return self.system.transactions

    @property
    def current_transaction(self):
        return self.transactions[self.transaction_index]

    @property
    def observation_size(self):
        return self.observation_shape[0]

    @property
    def hour(self) -> float:
        return self.current_time.hour / 24.0

    @property
    def day(self) -> int:
        return (self.current_time - self.t_start).days

    def compute_state(self):
        remaining_time = (self.t_max - self.current_time).seconds / self.t_max.second
        features = [remaining_time, self.current_card.is_credit, self.hour, self.day]
        if self.customer_location_is_known:
            features += [
                self.current_card.customer_x,
                self.current_card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def get_state(self):
        state = self.compute_state()
        return State(state)

    def step(self, action: Action):
        if self.current_time > self.t_max:
            raise ValueError("Cannot step past t_max: perform a reset() first.")

        np_action = action.to_numpy()
        clamped_np_action = self.action_space.clamp(np_action)
        action = Action.from_numpy(clamped_np_action)

        self.current_time += timedelta(action.delay_days, hours=action.delay_hours)
        if self.current_time < self.t_max:
            terminal_id = self.system.get_closest_terminal(self.current_card.customer_x, self.current_card.customer_y).id
            trx = Transaction(action.amount, self.current_time, terminal_id, self.current_card.id, action.is_online)
            is_fraud = self.system.classify(trx)
            if is_fraud:
                reward = 0.0
            else:
                reward = action.amount
            done = is_fraud
        else:
            done = True
            reward = 0.0
        self.transaction_index += 1
        state = self.compute_state()
        return Step(Observation(state, self.available_actions()), State(state), reward, done, False)

    def seed(self, seed_value: int):
        random.seed(seed_value)
