from banksys import TransactionWindow, Transaction
from datetime import datetime, timedelta


def test_update():
    window = TransactionWindow()
    assert len(window) == 0

    window.add(Transaction(100, datetime(2023, 1, 1), 0, 0, True, False))
    assert len(window) == 1

    window.update(datetime(2023, 1, 1, hour=10), timedelta(days=1))
    assert len(window) == 1

    window.update(datetime(2023, 1, 2, hour=1), timedelta(days=1))
    assert len(window) == 0

    window.add(Transaction(100, datetime(2023, 1, 1), 0, 0, True, False))
    window.add(Transaction(100, datetime(2023, 1, 2), 0, 0, True, False))
    window.add(Transaction(100, datetime(2023, 1, 3), 0, 0, True, False))
    window.update(datetime(2023, 1, 4), timedelta(days=7))
    assert len(window) == 3

    window.update(datetime(2023, 1, 8, hour=1), timedelta(days=7))
    assert len(window) == 2


def test_count_and_mean_and_risk():
    agg = (timedelta(days=1), timedelta(days=7))
    window = TransactionWindow()
    features = window.count_and_mean(agg, datetime(2023, 1, 10))
    for k, v in features.items():
        assert v == 0.0, f"Expected 0.0 for {k}, got {v}"

    features = window.count_and_risk(agg, datetime(2023, 1, 10))
    for k, v in features.items():
        assert v == 0.0, f"Expected 0.0 for {k}, got {v}"

    window.add(Transaction(100, datetime(2023, 1, 9), 0, 0, True, False, predicted_label=False))
    features = window.count_and_mean((timedelta(days=7),), datetime(2023, 1, 14))
    assert features[f"card_n_trx_last_{timedelta(7)}"] == 1
    assert features[f"card_mean_amount_last_{timedelta(7)}"] == 100.0

    features = window.count_and_risk((timedelta(days=7),), datetime(2023, 1, 14))
    assert features[f"terminal_n_trx_last_{timedelta(7)}"] == 1
    assert features[f"terminal_risk_last_{timedelta(7)}"] == 0.0

    window.add(Transaction(120, datetime(2023, 1, 14, hour=17), 0, 0, True, False, predicted_label=True))
    features = window.count_and_mean(agg, datetime(2023, 1, 15, hour=16))
    assert features[f"card_n_trx_last_{timedelta(1)}"] == 1
    assert features[f"card_n_trx_last_{timedelta(7)}"] == 2
    assert features[f"card_mean_amount_last_{timedelta(1)}"] == 120
    assert features[f"card_mean_amount_last_{timedelta(7)}"] == 110

    features = window.count_and_risk((timedelta(days=7),), datetime(2023, 1, 15, hour=18))
    assert features[f"terminal_n_trx_last_{timedelta(7)}"] == 2
    assert features[f"terminal_risk_last_{timedelta(7)}"] == 0.5

    features = window.count_and_risk((timedelta(days=7),), datetime(2023, 1, 18, hour=18))
    assert features[f"terminal_n_trx_last_{timedelta(7)}"] == 1
    assert features[f"terminal_risk_last_{timedelta(7)}"] == 1.0
