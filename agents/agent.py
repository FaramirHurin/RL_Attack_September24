from abc import ABC, abstractmethod
import numpy as np
from marlenv import Transition


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Choose an action based on the current state.
        """

    @abstractmethod
    def update(self, t: Transition):
        """
        Update the agent's policy based on the transition.
        """
