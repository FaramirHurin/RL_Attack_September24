import torch
import numpy as np
from typing import Type
from marlenv import ContinuousSpace, Episode, Transition, Observation, State, Step
from agents.rl import LinearActorCritic, RecurrentActorCritic
from agents.rl.batch import EpisodeBatch


def _make_test_without_batch(network_class: Type[LinearActorCritic | RecurrentActorCritic], use_covariance_matrix=True):
    """Test that multiple calls to policy() produce identical results."""
    # Setup
    state_size = 10
    action_space = ContinuousSpace(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
    device = torch.device("cpu")

    # Create network
    network = network_class(state_size, action_space, device, use_covariance_matrix)
    network.eval()  # Set to eval mode to disable dropout/batchnorm if any

    # Create test data
    batch_size = 32
    states = torch.randn(batch_size, state_size, device=device)

    # First policy call
    with torch.no_grad():
        policy1, _ = network.policy(states)
        base_dist1 = policy1.base_dist
        if isinstance(base_dist1, torch.distributions.Independent):
            base_dist1 = base_dist1.base_dist
            assert isinstance(base_dist1, torch.distributions.Normal)
            means1: torch.Tensor = base_dist1.loc
            stds1 = base_dist1.scale
            cov1 = None
        elif isinstance(base_dist1, torch.distributions.MultivariateNormal):  # MultivariateNormal
            means1 = base_dist1.mean
            cov1 = base_dist1.covariance_matrix
            stds1 = None
        else:
            raise ValueError("Unknown distribution type returned by policy")

    # Second policy call (should be identical)
    with torch.no_grad():
        policy2, _ = network.policy(states)
        base_dist2 = policy2.base_dist

        if isinstance(base_dist2, torch.distributions.Independent):  # Independent distribution
            assert stds1 is not None, "stds1 should not be None for Independent distribution"
            base_dist2 = base_dist2.base_dist
            assert isinstance(base_dist2, torch.distributions.Normal)
            means2 = base_dist2.loc
            stds2 = base_dist2.scale
            assert torch.equal(means1, means2)
            assert torch.equal(stds1, stds2)

        elif isinstance(base_dist2, torch.distributions.MultivariateNormal):  # MultivariateNormal
            assert cov1 is not None, "cov1 should not be None for MultivariateNormal distribution"
            means2 = base_dist2.mean
            cov2 = base_dist2.covariance_matrix
            assert torch.equal(means1, means2)
            assert torch.equal(cov1, cov2)


def _make_episode(length: int, state_size: int, n_actions: int):
    """Create a synthetic episode for testing."""
    assert length > 0, "Cannot create an episode of length <= 0"
    transitions = []
    obs_data = np.random.randn(state_size).astype(np.float32)
    obs = Observation(data=obs_data, available_actions=np.full(n_actions, True))
    state = State(data=obs_data)
    for t in range(length):
        action = np.random.random(n_actions)
        step = Step(obs, state, reward=0.5, done=(t == length - 1))
        transitions.append(Transition.from_step(obs, state, action, step))
        obs_data = np.random.randn(state_size).astype(np.float32)
        obs = Observation(data=obs_data, available_actions=np.full(n_actions, True))
        state = State(data=obs_data)
    return Episode.from_transitions(transitions)


def _make_test_with_minibatch(network_class: Type[RecurrentActorCritic | LinearActorCritic], use_covariance_matrix=True):
    """Test that policies computed on full batch match policies on minibatch at sampled indices."""
    # Setup
    state_size = 10
    n_actions = 2
    action_space = ContinuousSpace(low=np.array([0.0] * n_actions), high=np.array([1.0] * n_actions))
    device = torch.device("cpu")

    # Create network
    network = network_class(state_size, action_space, device, use_covariance_matrix)

    # Create episode batch with varying lengths
    n_episodes = 16
    episode_lengths = np.random.randint(5, 15, size=n_episodes)
    episodes = [_make_episode(length, state_size, n_actions) for length in episode_lengths]
    batch = EpisodeBatch(episodes, device=device)

    # Compute policy on full batch
    with torch.no_grad():
        full_policy, _ = network.policy(batch.obs)
        full_log_probs = full_policy.log_prob(batch.actions)
        full_base_dist = full_policy.base_dist

        if isinstance(full_base_dist, torch.distributions.Independent):
            full_base_dist = full_base_dist.base_dist
            assert isinstance(full_base_dist, torch.distributions.Normal)
            full_locs = full_base_dist.loc
            full_scales = full_base_dist.scale
        elif isinstance(full_base_dist, torch.distributions.MultivariateNormal):
            full_locs = full_base_dist.mean
            full_cov = full_base_dist.covariance_matrix
        else:
            raise ValueError("Unknown distribution type returned by policy")

    # Sample minibatch and compute policy
    minibatch_size = 8
    assert minibatch_size < n_episodes, "Test is wrongly written"
    for _ in range(10):
        sampled_indices = np.random.choice(n_episodes, minibatch_size, replace=False)
        minibatch = batch.get_minibatch(sampled_indices)
        sampled_indices = (slice(None), sampled_indices)
        mini_policy, _ = network.policy(minibatch.obs)
        mini_base_dist = mini_policy.base_dist

        if isinstance(mini_base_dist, torch.distributions.Independent):
            mini_base_dist = mini_base_dist.base_dist
            assert isinstance(mini_base_dist, torch.distributions.Normal)
            assert torch.equal(mini_base_dist.loc, full_locs[sampled_indices])
            assert torch.equal(mini_base_dist.scale, full_scales[sampled_indices])
        elif isinstance(mini_base_dist, torch.distributions.MultivariateNormal):
            assert torch.equal(mini_base_dist.mean, full_locs[sampled_indices])
            assert torch.equal(mini_base_dist.covariance_matrix, full_cov[sampled_indices])
        else:
            raise ValueError("Unknown distribution type returned by policy")
        mini_log_probs = mini_policy.log_prob(minibatch.actions)
        assert torch.equal(mini_log_probs, full_log_probs[sampled_indices])


def test_policy_consistency_linear_indenpendent():
    _make_test_without_batch(LinearActorCritic, use_covariance_matrix=True)


def test_policy_consistency_linear_multivariate():
    _make_test_without_batch(LinearActorCritic, use_covariance_matrix=False)


def test_policy_consistency_recurrent_independent():
    _make_test_without_batch(RecurrentActorCritic, use_covariance_matrix=True)


def test_policy_consistency_recurrent_multivariate():
    _make_test_without_batch(RecurrentActorCritic, use_covariance_matrix=False)


def test_policy_consistency_linear_minibatch_independent():
    _make_test_with_minibatch(LinearActorCritic, use_covariance_matrix=True)


def test_policy_consistency_linear_minibatch_multivariate():
    _make_test_with_minibatch(LinearActorCritic, use_covariance_matrix=False)


def test_policy_consistency_recurrent_minibatch_independent():
    _make_test_with_minibatch(RecurrentActorCritic, use_covariance_matrix=True)


def test_policy_consistency_recurrent_minibatch_multivariate():
    _make_test_with_minibatch(RecurrentActorCritic, use_covariance_matrix=False)
