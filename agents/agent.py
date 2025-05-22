from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from marlenv import Transition, Episode
import torch


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state: np.ndarray, hx: Optional[torch.Tensor] = None) -> tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Choose an action based on the current state.
        """

    @abstractmethod
    def update_transition(self, t: Transition, step: int):
        """
        Update the agent's policy based on the transition.
        """

    @abstractmethod
    def update_episode(self, episode: Episode, step_num: int, episode_num: int):
        """
        Update the agent's policy based on the episode.
        """
