from abc import ABC, abstractmethod
from typing_extensions import Self
import numpy as np
from marlenv import Transition


class Agent(ABC):
    @abstractmethod
    def select_action(self, obs_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Select an action and returns the action selected and the log probability of the action.
        """
        ...

    @abstractmethod
    def update(self): ...
    @abstractmethod
    def to(self, device) -> Self: ...

    @abstractmethod
    def store(self, transition: Transition): ...
