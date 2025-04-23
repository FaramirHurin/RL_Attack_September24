from banksys import Banksys, Transaction, Card
from .action import Action
import random
import numpy as np
from datetime import timedelta
from marlenv import Observation, Step, MARLEnv, State, ContinuousActionSpace
from datetime import datetime


class CardSimEnv(MARLEnv[Action, ContinuousActionSpace]):
    def __init__(self, system: Banksys, attack_duration: timedelta, n_parallel: int = 10, *, customer_location_is_known: bool = False):
        if customer_location_is_known:
            obs_shape = (6,)
        else:
            obs_shape = (4,)
        super().__init__(
            ContinuousActionSpace(
                1,
                low=[0.01] + [0.0] * 5,
                high=[100_000, 200, 200, 1, attack_duration.days, attack_duration.seconds / 3600],
                action_names=["amount", "terminal_x", "terminal_y", "is_online", "delay_days", "delay_hours"],
            ),
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )
        self.n_parallel = n_parallel
        """Index of the next transaction to be processed."""
        assert isinstance(system, Banksys), "System must be an instance of Banksys"
        self.system = system
        self.t_start = system.earliest_attackable_moment
        self.t_max = self.t_start + attack_duration
        self.attack_duration = attack_duration
        self.customer_location_is_known = customer_location_is_known
        self.prev_t = self.t_start

    def reset(self):
        self.prev_t = self.t_start
        obs = list[tuple[Card, Observation, State]]()
        available_actions = self.available_actions()
        for card in random.sample(self.system.cards, k=self.n_parallel):
            s = self.compute_state(self.t_start, card)
            obs.append((card, Observation(s, available_actions), State(s)))
        return obs

    def get_observation(self):
        raise NotImplementedError()

    def get_state(self, card_id: int):
        raise NotImplementedError()

    @property
    def observation_size(self):
        return self.observation_shape[0]

    def compute_state(self, t: datetime, card: Card):
        remaining_time = (self.t_max - t).seconds / self.t_max.second
        day = (t - self.t_start).days
        hour = t.hour / 24.0
        features = [remaining_time, card.is_credit, hour, day]
        if self.customer_location_is_known:
            features += [
                card.customer_x,
                card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def step(self, t: datetime, action: Action, card: Card):
        assert t > self.prev_t, (
            f"Can not go back in time with actions ! Previous action was taken at {self.prev_t} and current action is at {t}."
        )
        self.prev_t = t

        if t > self.t_max:
            done = True
            reward = 0.0
            trx = None
        # Get the next action to be performed and apply it to the BankSys
        else:
            terminal_id = self.system.get_closest_terminal(card.customer_x, card.customer_y).id
            trx = Transaction(action.amount, t, terminal_id, card.id, action.is_online)
            is_fraud = self.system.classify(trx)
            if is_fraud:
                reward = 0.0
            else:
                reward = action.amount
            done = is_fraud
        state = self.compute_state(t, card)
        return Step(Observation(state, self.available_actions()), State(state), reward, done, False), trx

    def seed(self, seed_value: int):
        random.seed(seed_value)
