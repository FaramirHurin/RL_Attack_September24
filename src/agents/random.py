from marlenv import ContinuousSpace, Episode, Transition
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: ContinuousSpace):
        super().__init__()
        self.action_space = action_space

    def choose_action(self, *args, **kwargs):
        return self.action_space.sample(), None

    def update_episode(self, episode: Episode, step_num: int, episode_num: int, simulation_t: int):
        return

    def update_transition(self, transition: Transition, step: int, episode_num: int, simulation_t: int):
        return
