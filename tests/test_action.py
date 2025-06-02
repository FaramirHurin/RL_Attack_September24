from environment import Action
import numpy as np


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

    assert action.amount == 100
    assert action.terminal_x == 1
    assert action.terminal_y == 2
    assert action.is_online is True  # True is represented as 1
    assert action.delay_hours == 3


def test_action_conversions():
    original_action = Action(amount=50, terminal_x=10, terminal_y=20, is_online=False, delay_hours=2.5)
    numpy_action = original_action.to_numpy()
    converted_action = Action.from_numpy(numpy_action)

    assert original_action.amount == converted_action.amount
    assert original_action.terminal_x == converted_action.terminal_x
    assert original_action.terminal_y == converted_action.terminal_y
    assert original_action.is_online == converted_action.is_online
    assert original_action.delay_hours == converted_action.delay_hours
