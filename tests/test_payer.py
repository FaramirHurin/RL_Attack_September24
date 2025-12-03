from banksys import Payer, Transaction
from datetime import datetime, timedelta
from exceptions import InsufficientFundsError

AGG_WINDOWS = (timedelta(hours=12),)


def test_balance():
    c = Payer(0, 0, 0, 20, AGG_WINDOWS)
    t = Transaction(20, datetime.now(), 0, 0, False, True, predicted_label=False)
    c.add(t, update_balance=False)

    assert c.balance == 20
    t2 = Transaction(10, datetime.now(), 0, 0, False, True, predicted_label=False)
    c.add(t2, update_balance=True)
    assert c.balance == 10

    t3 = Transaction(15, datetime.now(), 0, 0, False, True, predicted_label=False)
    try:
        c.add(t3, update_balance=True)
        assert False, "Expected InsufficientFundsError"
    except InsufficientFundsError:
        pass


def test_features():
    agg = (timedelta(days=1), timedelta(days=7))
    payer = Payer(0, 0, 0, 500, agg)
    features = payer.compute_features(datetime(2023, 10, 10))
    for k, v in features.items():
        assert v == 0.0, f"Expected 0.0 for {k}, got {v}"

    trx = Transaction(100, datetime(2023, 1, 9), 0, 0, True, False, predicted_label=False)
    payer.add(trx, update_balance=False)
    features = payer.compute_features(trx.timestamp)
    assert features[Payer.colname("count", timedelta(days=1))] == 1.0
    assert features[Payer.colname("avg", timedelta(days=1))] == 100.0
    assert features[Payer.colname("count", timedelta(days=7))] == 1.0
    assert features[Payer.colname("avg", timedelta(days=7))] == 100.0

    trx = Transaction(120, datetime(2023, 1, 14, hour=17), 0, 0, True, False, predicted_label=False)
    payer.add(trx, update_balance=False)
    features = payer.compute_features(trx.timestamp)
    assert features[Payer.colname("count", timedelta(days=7))] == 2
    assert features[Payer.colname("avg", timedelta(days=7))] == 110.0
    assert features[Payer.colname("count", timedelta(days=1))] == 1
    assert features[Payer.colname("avg", timedelta(days=1))] == 120.0

    trx = Transaction(180, datetime(2023, 1, 18, hour=18), 0, 0, True, False, predicted_label=False)
    payer.add(trx, update_balance=False)
    features = payer.compute_features(trx.timestamp)
    assert features[Payer.colname("count", timedelta(days=7))] == 2.0
    assert features[Payer.colname("avg", timedelta(days=7))] == 150.0
