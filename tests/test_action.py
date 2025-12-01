from environment.action import Action, IS_ONLINE_INDEX, AMOUNT_INDEX, TERMINAL_X_INDEX, TERMINAL_Y_INDEX, DELAY_HOURS_INDEX, FIELDS_INDEX
import numpy as np
import pytest


def test_action_to_numpy():
    action = Action(amount=100, terminal_x=1, terminal_y=2, is_online=True, delay_hours=3)
    numpy_action = action.to_numpy()

    assert len(numpy_action) == 5
    amount, terminal_x, terminal_y, is_online, delay_hours = numpy_action
    assert amount == 100
    assert terminal_x == 1
    assert terminal_y == 2
    assert is_online == 1  # True is represented as 1
    assert delay_hours == 3


def test_action_from_numpy():
    numpy_action = np.array([100, 1, 2, 1, 3], dtype=np.float32)  # is_online is represented as 0 (False)
    action = Action.from_numpy(numpy_action)

    assert action.amount == 100.0
    assert action.terminal_x == 1.0
    assert action.terminal_y == 2.0
    assert action.is_online is True
    assert action.delay_hours == 3.0


def test_action_indices():
    for _ in range(1_000):
        numpy_action = np.random.random(len(FIELDS_INDEX)).astype(np.float32) + [0.01, 0.0, 0.0, 0.0, 0.0]  # Ensure amount is at least 0.01
        action = Action.from_numpy(numpy_action)
        assert round(numpy_action[AMOUNT_INDEX], 2) == action.amount
        assert pytest.approx(numpy_action[TERMINAL_X_INDEX]) == action.terminal_x
        assert pytest.approx(numpy_action[TERMINAL_Y_INDEX]) == action.terminal_y
        assert bool(numpy_action[IS_ONLINE_INDEX] > 0.5) == action.is_online
        assert pytest.approx(numpy_action[DELAY_HOURS_INDEX]) == action.delay_hours


def test_from_np_deterministic():
    np_action = np.random.rand(5).astype(np.float32)
    action1 = Action.from_numpy(np_action)
    action2 = Action.from_numpy(np_action)
    assert action1 == action2


def test_action_conversions():
    original_action = Action(amount=50, terminal_x=10, terminal_y=20, is_online=False, delay_hours=2.5)
    numpy_action = np.zeros(len(FIELDS_INDEX), dtype=np.float32)
    numpy_action[AMOUNT_INDEX] = original_action.amount
    numpy_action[TERMINAL_X_INDEX] = original_action.terminal_x
    numpy_action[TERMINAL_Y_INDEX] = original_action.terminal_y
    numpy_action[IS_ONLINE_INDEX] = 1.0 if original_action.is_online else 0.0
    numpy_action[DELAY_HOURS_INDEX] = original_action.delay_hours
    converted_action = Action.from_numpy(numpy_action)

    assert original_action.amount == converted_action.amount
    assert original_action.terminal_x == converted_action.terminal_x
    assert original_action.terminal_y == converted_action.terminal_y
    assert original_action.is_online == converted_action.is_online
    assert original_action.delay_hours == converted_action.delay_hours
