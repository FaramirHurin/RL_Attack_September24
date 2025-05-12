from banksys import Banksys, Transaction
from .action import Action
import random
from copy import deepcopy
import numpy as np
from datetime import timedelta
from marlenv import Observation, Step, MARLEnv, State, ContinuousSpace, ActionSpace
from .card_registry import CardRegistry

class SimpleCardSimEnv(MARLEnv[ContinuousSpace]):
    def __init__(
        self,
        system: Banksys,
        avg_card_block_delay: timedelta = timedelta(days=7),
        *,
        customer_location_is_known: bool = False,
    ):
        """
        Args:
            system: The Banksys object to be used for the simulation.
            card_block_delay: The average time delay before a card can be blocked.
            n_parallel: The number of parallel transactions to be processed.
            customer_location_is_known: Whether the customer's location is known.
        """
        if customer_location_is_known:
            obs_shape = (6,)
        else:
            obs_shape = (4,)
        action_space = ActionSpace(1, ContinuousSpace( low= np.array([0.01] + [0.0] * 5),
                high=np.array([100_000, 200, 200, 1, avg_card_block_delay.days, avg_card_block_delay.total_seconds() / 3600]),
                labels=["amount", "terminal_x", "terminal_y", "is_online", "delay_days", "delay_hours"],
            ))
        super().__init__(
            action_space=action_space,
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )
        self.system = system
        self.saved_system = deepcopy(system)
        self.t = system.earliest_attackable_moment
        self.t_start = deepcopy(system.earliest_attackable_moment)
        self.card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.customer_location_is_known = customer_location_is_known
        self.current_card = self.card_registry.release_card(self.t)
        self.transactions = list[Transaction]()

    def reset(self):
        self.system.rollback(self.transactions)
        self.transactions = []
        self.t = self.t_start
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
            features += [
                self.current_card.customer_x,
                self.current_card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def step(self, np_action: np.ndarray):
        """
        Perform the given action at the given time.
        """
        action = Action.from_numpy(np_action)
        self.t += action.timedelta
        if self.card_registry.has_expired(self.current_card, self.t):
            self.card_registry.clear(self.current_card)
            done = True
            reward = 0.0
            trx = None
        else:
            terminal_id = self.system.get_closest_terminal(self.current_card.customer_x, self.current_card.customer_y).id
            trx = Transaction(action.amount, self.t, terminal_id, self.current_card.id, action.is_online)
            is_fraud = self.system.process_transaction(trx)
            self.transactions.append(trx)
            if is_fraud:
                reward = 0.0
            else:
                reward = action.amount
            done = is_fraud
        state = self.compute_state()
        return Step(Observation(state, self.available_actions()), State(state), reward, done, False), trx

    def seed(self, seed_value: int):
        random.seed(seed_value)
