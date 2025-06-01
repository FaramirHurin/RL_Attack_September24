from marlenv import Transition
import numpy as np
from agents import Agent


class SimpleGenetic(Agent):
    def __init__(self):
        super().__init__()

    def choose_action(self, state: np.ndarray):
        return super().choose_action(state)

    def update_transition(self, t: Transition):
        return
