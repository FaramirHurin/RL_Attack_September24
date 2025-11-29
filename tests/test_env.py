from environment import CardSimEnv, Action
from banksys import Payer
from datetime import timedelta
from parameters import EnvParameters
from copy import deepcopy
import numpy as np

from .mocks import mock_banksys, MockClassificationSystem

# NOTE: mock_banksys is a fixture that provides a mocked Banksys instance with a mock classification system. It is defined in conftest.py.


def test_spawn_card():
    bs = mock_banksys()
    params = EnvParameters(avg_card_block_delay=timedelta(days=1))
    env = CardSimEnv(bs, params)
    payer, _, _ = env.spawn_payer()
    assert len(env.payer_registry.expected_expirations) == 1
    assert isinstance(payer, Payer)


def test_obs_size():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    obs_size = env.observation_size
    _, obs, *_ = env.spawn_payer()
    assert len(obs.data) == obs_size


def test_observation():
    bs = mock_banksys()
    clf = clf = MockClassificationSystem()
    bs.clf = clf
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))

    payer, obs, _ = env.spawn_payer()
    payer.balance = 1000
    # Manually set the actual expiration to the expected one for determinism
    env.payer_registry.actual_expirations[payer] = env.payer_registry.expected_expirations[payer]
    hour_ratio, time_remaining, prev_fraud_time, total_stolen, balance_upper_bound, *days, x, y = obs.data
    assert prev_fraud_time == -1.0  # No previous frauds
    assert time_remaining == 1.0
    assert hour_ratio == env.t.hour / 24
    assert balance_upper_bound == -1.0  # Not blocked yet, so no upper bound
    assert total_stolen == 0.0

    action = Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1)
    env.buffer_action(action.to_numpy(), payer)
    payer, step, _ = env.step()
    assert payer.balance == 990

    hour_ratio, time_remaining, prev_fraud_time, total_stolen, balance_upper_bound, *days, x, y = step.obs.data
    assert prev_fraud_time == 1 / 24
    assert time_remaining == 23 / 24
    assert hour_ratio == env.t.hour / 24
    assert total_stolen == 10.0 / 100.0
    assert balance_upper_bound == -1.0

    action = Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1)
    env.buffer_action(action.to_numpy(), payer)
    bs.simulate_until(env.t + action.timedelta)
    clf.set_next_predictions(True)  # Next transaction will be detected as a fraud
    payer, step, _ = env.step()
    assert payer.balance == 990
    assert step.done


def test_card_blocked_zero_reward():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    card, _, _ = env.spawn_payer()
    card.balance = 5
    env.buffer_action(Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1).to_numpy(), card)
    card, step, _ = env.step()
    assert step.reward.item() == 0.0, "Reward should be zero when card is blocked due to insufficient balance"
    assert not step.done

    hour_ratio, time_remaining, prev_fraud_time, total_stolen, balance_upper_bound, *days, x, y = step.obs.data
    assert hour_ratio == env.t.hour / 24
    assert time_remaining == 23 / 24
    assert prev_fraud_time == -1.0
    assert total_stolen == 0.0
    assert balance_upper_bound == 10.0 / 100.0


def test_time_going():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    # Timeline
    # Card 1  Spawn  --- 1h buffer action ----------------------- 3h action --
    # Card 2  Spawn  -------------------------1h30 action --------------------

    card1 = env.spawn_payer()[0]
    card2 = env.spawn_payer()[0]

    t_0 = deepcopy(env.t)

    action1 = Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1)
    env.buffer_action(action1.to_numpy(), card1)

    action2 = Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1.5)
    env.buffer_action(action2.to_numpy(), card2)

    card, step, np_action = env.step()
    assert card == card1
    assert env.t == t_0 + timedelta(hours=action1.delay_hours)
    assert step.reward.item() == action1.amount
    assert np.array_equal(np_action, action1.to_numpy())

    action3 = Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=2)
    env.buffer_action(action3.to_numpy(), card1)

    card, step, np_action = env.step()
    assert card == card2
    assert env.t == t_0 + timedelta(hours=action2.delay_hours)
    assert step.reward.item() == action2.amount
    assert np.array_equal(np_action, action2.to_numpy())

    card, step, np_action = env.step()
    assert card == card1
    assert env.t == t_0 + timedelta(hours=action1.delay_hours + action3.delay_hours)
    assert step.reward.item() == action3.amount
    assert np.array_equal(np_action, action3.to_numpy())


def test_bufferred_action_is_stepped():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters())
    payer = env.spawn_payer()[0]
    np_actions = [
        np.array([5, 30, 30, 1, 0.25], dtype=np.float32),
        np.array([50, 10, 10, 1, 0.5], dtype=np.float32),
        np.array([20, 20, 20, 0, 1.0], dtype=np.float32),
    ]
    actions = [Action.from_numpy(a) for a in np_actions]
    # Not added in chronological order
    env.buffer_action(np_actions[1], payer)
    env.buffer_action(np_actions[2], payer)
    env.buffer_action(np_actions[0], payer)

    for action, np_action in zip(actions, np_actions):
        stepped_payer, _, stepped_np_action = env.step()
        assert action == Action.from_numpy(stepped_np_action)
        assert stepped_payer == payer
        assert np.array_equal(stepped_np_action, np_action)
