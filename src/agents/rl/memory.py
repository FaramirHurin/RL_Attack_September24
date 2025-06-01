import torch
from marlenv import Episode, Transition

from .batch import EpisodeBatch


class Memory:
    def __init__(self):
        self._episodes = list[Episode]()
        self.size = 0

    @property
    def current_episode(self):
        if len(self._episodes) == 0:
            return None
        return self._episodes[-1]

    def add(self, transition: Transition):
        self.size += 1
        # Either the very first episode or the previous episode is finished,
        # so we create a new one.
        current_episode = self.current_episode
        if current_episode is None or current_episode.is_finished:
            current_episode = Episode.from_transitions([transition])
            self._episodes.append(current_episode)
        else:
            current_episode.add(transition)

    def clear(self):
        last_episode = self.current_episode
        if last_episode is None:
            pass
        elif last_episode.is_finished:
            self._episodes.clear()
        else:
            last_episode.invalidate_cached_properties()
            self._episodes = [last_episode]
        self.size = 0

    def to_batch(self, device: torch.device):
        """Convert the memory to a batch of episodes."""
        if self.size == 0:
            raise ValueError("Memory is empty")
        if len(self._episodes) == 0:
            assert self.current_episode is not None
            return EpisodeBatch([self.current_episode])
        return EpisodeBatch(self._episodes, device)
