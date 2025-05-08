from banksys import Banksys, Transaction, Card
from .action import Action
import random
import numpy as np
from datetime import timedelta
from marlenv import Observation, Step, MARLEnv, State, ContinuousSpace
from .card_registry import CardRegistry


class CardSimEnv(MARLEnv[ContinuousSpace]):
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
        super().__init__(
            1,
            ContinuousSpace(
                low=[0.01] + [0.0] * 5,
                high=[100_000, 200, 200, 1, avg_card_block_delay.days, avg_card_block_delay.total_seconds() / 3600],
                labels=["amount", "terminal_x", "terminal_y", "is_online", "delay_days", "delay_hours"],
            ),
            observation_shape=obs_shape,
            state_shape=obs_shape,
        )
        self.system = system
        self.t = system.earliest_attackable_moment
        """Current time in the simulation."""
        self.card_registry = CardRegistry(system.cards, avg_card_block_delay)
        self.customer_location_is_known = customer_location_is_known

    def reset(self, n_parallel: int = 10):
        cards = [self.steal_card() for _ in range(n_parallel)]
        obs = [self.get_observation(c) for c in cards]
        states = [self.get_state(c) for c in cards]
        return list(zip(cards, obs, states))

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
        features = [time_ratio, card.is_credit, self.t.hour, self.t.day]
        if self.customer_location_is_known:
            features += [
                card.customer_x,
                card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def steal_card(self):
        return self.card_registry.release_card(self.t)

    def step(self, action: Action, card: Card):
        """
        Perform the given action at the given time.
        """
        self.t += action.timedelta
        self.card_registry.update(self.t)
        if self.card_registry.has_expired(card, self.t):
            done = True
            reward = 0.0
            trx = None
        else:
            terminal_id = self.system.get_closest_terminal(card.customer_x, card.customer_y).id
            trx = Transaction(action.amount, self.t, terminal_id, card.id, action.is_online)
            is_fraud = self.system.process_transaction(trx)
            if is_fraud:
                reward = 0.0
            else:
                reward = action.amount
            done = is_fraud
        state = self.compute_state(card)
        return Step(Observation(state, self.available_actions()), State(state), reward, done, False), trx

    def seed(self, seed_value: int):
        random.seed(seed_value)
