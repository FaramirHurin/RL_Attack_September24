from agents.rl.batch import TransitionBatch, EpisodeBatch
import torch
import numpy as np
from marlenv import Episode, Transition, Observation, State, Step
from typing import Sequence
import random


def _make_episode(length: int, n_actions: int, step_reward: float):
    assert length > 0, "Cannot create an episode of length <= 0"
    transitions = list[Transition]()
    obs = Observation(data=np.array([0]), available_actions=np.full(n_actions, True))
    state = State(data=np.array([0]))
    for t in range(length):
        action = np.random.random(n_actions)
        step = Step(obs, state, step_reward, t == length - 1)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs = Observation(data=np.array([t]), available_actions=np.full(n_actions, True))
        state = State(data=np.array([t]))
    return Episode.from_transitions(transitions)


def _make_episode_batch(ep_lengths: Sequence[int] | np.ndarray, n_actions: int = 5, step_reward: float = 0.5):
    episodes = list[Episode]()
    for e, length in enumerate(ep_lengths):
        episodes.append(_make_episode(length, n_actions, step_reward))
    return EpisodeBatch(episodes)


def _make_transition_batch(size: int, n_actions: int = 5, reward: float = 0.5):
    transitions = list[Transition]()
    for t in range(size):
        transition = Transition(
            Observation(data=np.array([t]), available_actions=np.full(n_actions, True)),
            State(data=np.array([t])),
            np.random.random(n_actions),
            reward,
            t % 10 == 9,
            {},
            Observation(data=np.array([t + 1]), available_actions=np.full(n_actions, True)),
            State(data=np.array([t + 1])),
            False,
        )
        transitions.append(transition)
    return TransitionBatch(transitions)


def test_episode_minibatch():
    lengths = list(range(5, 10))
    batch = _make_episode_batch(lengths, step_reward=0.5)

    for _ in range(10):
        sampled_indices = random.choices(range(len(lengths)), k=3)
        minibatch = batch.get_minibatch(sampled_indices)
        for i, sampled_index in enumerate(sampled_indices):
            length = lengths[sampled_index]
            assert torch.all(minibatch.rewards[:length, i] == 0.5)
            assert torch.all(minibatch.rewards[length:, i] == 0.0)
            assert torch.all(minibatch.masks[:length, i] == 1)
            assert torch.all(minibatch.masks[length:, i] == 0)
            assert torch.all(~minibatch.dones[: length - 1, i])
            assert torch.all(minibatch.dones[length - 1 :, i])


def test_episode_batch_creation():
    BATCH_SIZE = 5
    MIN_EP_LENGTH = 10
    batch = _make_episode_batch(range(MIN_EP_LENGTH, MIN_EP_LENGTH + BATCH_SIZE), step_reward=0.5)
    assert len(batch) == BATCH_SIZE
    assert batch.size == BATCH_SIZE
    for t in range(BATCH_SIZE):
        assert torch.all(batch.masks[: MIN_EP_LENGTH + t, t] == 1)
        assert torch.all(batch.masks[MIN_EP_LENGTH + t :, t] == 0)
        assert torch.all(~batch.dones[: MIN_EP_LENGTH + t - 1, t])
        assert torch.all(batch.dones[MIN_EP_LENGTH + t - 1 :, t])
        assert torch.all(batch.rewards[: MIN_EP_LENGTH + t, t] == 0.5)
        assert torch.all(batch.rewards[MIN_EP_LENGTH + t :, t] == 0.0)
