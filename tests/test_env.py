from environment import CardSimEnv, Action
from banksys import Payer
from datetime import timedelta
from parameters import EnvParameters
from copy import deepcopy
import numpy as np

from .mocks import mock_banksys

# NOTE: mock_banksys is a fixture that provides a mocked Banksys instance with a mock classification system. It is defined in conftest.py.


def test_spawn_card():
    bs = mock_banksys()
    params = EnvParameters(avg_card_block_delay=timedelta(days=1))
    env = CardSimEnv(bs, params)
    payer, _, _ = env.spawn_card()
    assert len(env.payer_registry.expected_expirations) == 1
    assert isinstance(payer, Payer)


def test_obs_size():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    obs_size = env.observation_size
    _, obs, *_ = env.spawn_card()
    assert len(obs.data) == obs_size


def test_observation():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))

    payer, obs, _ = env.spawn_card()
    payer.balance = 1000
    # Manually set the actual expiration to the expected one for determinism
    env.payer_registry.actual_expirations[payer] = env.payer_registry.expected_expirations[payer]
    hour_ratio, time_remaining, total_stolen, n_attacks, balance_upper_bound, *days, x, y = obs.data
    assert n_attacks == 0
    assert time_remaining == 1.0
    assert hour_ratio == env.t.hour / 24
    assert balance_upper_bound == -1.0  # Not blocked yet, so no upper bound
    assert total_stolen == 0.0

    env.buffer_action(Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1).to_numpy(), payer)
    payer, step, _ = env.step()
    assert payer.balance == 990

    hour_ratio, time_remaining, total_stolen, n_attacks, balance_upper_bound, *days, x, y = step.obs.data
    assert n_attacks == 1
    assert time_remaining == 23 / 24
    assert hour_ratio == env.t.hour / 24
    assert total_stolen == 10.0 / 100.0
    assert balance_upper_bound == -1.0


def test_card_blocked_zero_reward():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    card, _, _ = env.spawn_card()
    card.balance = 5
    env.buffer_action(Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1).to_numpy(), card)
    card, step, _ = env.step()
    assert step.reward.item() == 0.0, "Reward should be zero when card is blocked due to insufficient balance"
    assert not step.done

    hour_ratio, time_remaining, total_stolen, n_attacks, balance_upper_bound, *days, x, y = step.obs.data
    assert hour_ratio == env.t.hour / 24
    assert time_remaining == 23 / 24
    assert total_stolen == 0.0
    assert n_attacks == 1
    assert balance_upper_bound == 10.0 / 100.0


def test_time_going():
    bs = mock_banksys()
    env = CardSimEnv(bs, EnvParameters(avg_card_block_delay=timedelta(days=1)))
    # Timeline
    # Card 1  Spawn  --- 1h buffer action ----------------------- 3h action --
    # Card 2  Spawn  -------------------------1h30 action --------------------

    card1 = env.spawn_card()[0]
    card2 = env.spawn_card()[0]

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
