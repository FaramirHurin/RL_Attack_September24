from banksys import Card, Transaction
from datetime import datetime
from exceptions import InsufficientFundsError


def test_balance():
    c = Card(0, 0, 0, 20, True)
    t = Transaction(20, datetime.now(), 0, 0, False, True)
    c.add(t, update_balance=False)

    assert c.balance == 20
    t2 = Transaction(10, datetime.now(), 0, 0, False, True)
    c.add(t2, update_balance=True)
    assert c.balance == 10

    t3 = Transaction(15, datetime.now(), 0, 0, False, True)
    try:
        c.add(t3, update_balance=True)
        assert False, "Expected InsufficientFundsError"
    except InsufficientFundsError:
        pass
