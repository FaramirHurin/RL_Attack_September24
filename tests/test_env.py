from environment import CardSimEnv, Action
from banksys import Card, Banksys
from datetime import timedelta


from .mocks import mock_banksys

# NOTE: mock_banksys is a fixture that provides a mocked Banksys instance with a mock classification system. It is defined in conftest.py.


def test_spawn_card():
    bs = mock_banksys()
    env = CardSimEnv(bs, timedelta(days=1))
    card, _, _ = env.spawn_card()
    assert len(env.card_registry.expected_expirations) == 1
    assert isinstance(card, Card)


def test_observation():
    bs = mock_banksys()
    env = CardSimEnv(bs, timedelta(days=1))

    card, obs, _ = env.spawn_card()
    card.balance = 1000
    # Manually set the actual expiration to the expected one for determinism
    env.card_registry.actual_expirations[card] = env.card_registry.expected_expirations[card]
    n_attacks, time_remaining, is_credit, hour_ratio, *_ = obs.data
    assert n_attacks == 0
    assert time_remaining == 1.0
    assert bool(is_credit) == card.is_credit
    assert hour_ratio == env.t.hour / 24

    env.buffer_action(Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1).to_numpy(), card)
    card, step = env.step()
    assert card.balance == 990

    n_attacks, time_remaining, is_credit, hour_ratio, *_ = step.obs.data
    assert n_attacks == 1
    assert time_remaining == 23 / 24
    assert bool(is_credit) == card.is_credit
    assert hour_ratio == env.t.hour / 24


def test_card_blocked_zero_reward():
    bs = mock_banksys()
    env = CardSimEnv(bs, timedelta(days=1))
    card, _, _ = env.spawn_card()
    card.balance = 5
    env.buffer_action(Action(amount=10, terminal_x=0, terminal_y=0, is_online=True, delay_hours=1).to_numpy(), card)
    card, step = env.step()
    assert step.reward.item() == 0.0, "Reward should be zero when card is blocked due to insufficient balance"
    assert not step.done

    n_attacks, time_remaining, is_credit, hour_ratio, *_ = step.obs.data
    assert n_attacks == 1
    assert time_remaining == 23 / 24
    assert bool(is_credit) == card.is_credit
    assert hour_ratio == env.t.hour / 24
