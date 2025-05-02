from environment.priority_queue import PriorityQueue
from .agents.ppo_new import PPO
from banksys import Card
from environment import Action
from marlenv import Episode, Observation, State, Step
from datetime import datetime
from Baselines.attack_generation import Delayed_Vae_Agent


class DelayedParallelAgent:
    def __init__(self, agent: PPO|Delayed_Vae_Agent):
        self.buffered_actions = PriorityQueue[tuple[Card, Action]]()
        self.episodes = dict[int, Episode]()
        self.agent = agent

    def reset(self, t: datetime, card_data: list[tuple[Card, Observation, State]]):
        self.buffered_actions.clear()
        self.episodes.clear()
        for card, obs, state in card_data:
            action = self.agent.choose_action(obs.data)
            if not sum(obs.data) == 0:
                DEBUG = True
            t_action = t + action.timedelta
            self.buffered_actions.push((card, action), t_action)
            self.episodes[card.id] = Episode.new(obs, state)

    def buffer_action_for(self, current_time: datetime, card: Card):
        e = self.episodes[card.id]
        obs = e.obs[-1]
        action = self.agent.choose_action(obs)
        self.buffered_actions.push((card, action), current_time + action.timedelta)

    def pop_next_action(self):
        return self.buffered_actions.ppop()

    def store_transition(self, t: datetime, card: Card, action: Action, step: Step):
        """
        When an action has been performed in the environment, we store the transition in the episode.
        """
        e = self.episodes[card.id]
        e.add(step, action)
        if not step.is_terminal:
            self.buffer_action_for(t, card)

    @property
    def is_done(self):
        return len(self.buffered_actions) == 0
