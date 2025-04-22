from banksys import Banksys, Transaction, Card
from .action import Action
from copy import deepcopy
import random
from collections import OrderedDict
import numpy as np
from datetime import timedelta
from marlenv import Observation, Step, MARLEnv, ContinuousSpace, State, ContinuousActionSpace
#ContinuousSpace  ContinuousActionSpace

class CardSimEnv(MARLEnv[Action, ContinuousActionSpace]):
    def __init__(self, system: Banksys, attack_duration: timedelta, n_parallel: int = 10, *, customer_location_is_known: bool = False):
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
        self.n_parallel = n_parallel
        """Index of the next transaction to be processed."""
        assert isinstance(system, Banksys), "System must be an instance of Banksys"
        self.system = system
        self.t_start = system.earliest_attackable_moment
        self.current_time = deepcopy(self.t_start)
        self.current_cards = OrderedDict[int, Card]()
        self.buffered_actions = OrderedDict[int, Action]()
        """Maps card IDs to actions."""
        self.t_max = self.t_start + attack_duration
        self.attack_duration = attack_duration
        self.customer_location_is_known = customer_location_is_known

    def reset(self):
        self.current_cards.clear()
        self.buffered_actions.clear()
        for card in random.sample(self.system.cards, k=self.n_parallel):
            self.current_cards[card.id] = card
        self.current_time = deepcopy(self.t_start)
        obs = OrderedDict[int, Observation]()
        states = OrderedDict[int, State]()
        available_actions = self.available_actions()
        for card_id in self.current_cards.keys():
            s = self.compute_state(card_id)
            obs[card_id] = Observation(s, available_actions)
            states[card_id] = State(s)
        return obs, states

    def get_observation(self, card_id: int):
        state = self.compute_state(card_id)
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

    def compute_state(self, card_id: int):
        card = self.current_cards[card_id]
        remaining_time = (self.t_max - self.current_time).seconds / self.t_max.second
        features = [remaining_time, card.is_credit, self.hour, self.day]
        if self.customer_location_is_known:
            features += [
                card.customer_x,
                card.customer_y,
            ]
        return np.array(features, dtype=np.float32)

    def get_state(self, card_id: int):
        state = self.compute_state(card_id)
        return State(state)

    def first_step(self, actions: list[Action]):
        """
        Since we assume that the attacker steals n_parallel cards at once, the first step
        has to receive a list of actions, one for each card.

        The environment stores the actions of each card, computes which action has to be
        performed first, and then calls the step() for the first action.
        """
        min_id = 0
        t_min = self.t_start
        for card_id, action in zip(self.current_cards.keys(), actions):
            self.buffered_actions[card_id] = action
            t = self.t_start + action.timedelta
            if t < t_min:
                t_min = t
                min_id = card_id
        return self.step(actions[min_id], min_id)

    def get_next_action(self) -> tuple[Action, Card]:
        min_id = 0
        t_min = self.t_start
        for card_id, action in self.buffered_actions.items():
            t = self.t_start + action.timedelta
            if t < t_min:
                t_min = t
                min_id = card_id
        return self.buffered_actions[min_id], self.current_cards[min_id]

    def step(self, action: Action, card_id: int):
        """
        Buffer the given action for further processing, then compute the next action to be
        performed and apply it to the BankSys.

        The returned step contains the observation of the card that was processed, along with
        the its ID.
        """
        if self.current_time > self.t_max:
            raise ValueError("Cannot step past t_max: perform a reset() first.")

        np_action = action.to_numpy()
        clamped_np_action = self.action_space.clamp(np_action)
        action = Action.from_numpy(clamped_np_action)
        self.buffered_actions[card_id] = action

        # Get the next action to be performed and apply it to the BankSys
        action, card = self.get_next_action()
        self.current_time += timedelta(action.delay_days, hours=action.delay_hours)
        if self.current_time < self.t_max:
            terminal_id = self.system.get_closest_terminal(card.customer_x, card.customer_y).id
            trx = Transaction(action.amount, self.current_time, terminal_id, card.id, action.is_online)
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
        state = self.compute_state(card.id)
        return Step(Observation(state, self.available_actions()), State(state), reward, done, False), card.id

    def seed(self, seed_value: int):
        random.seed(seed_value)
