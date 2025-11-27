from banksys import Transaction, Terminal
from banksys.terminal import PREFIX_N_TRX, PREFIX_RISK
from datetime import datetime, timedelta


def test_terminal_features():
    agg = (timedelta(days=1), timedelta(days=7))
    terminal = Terminal(0, 0, 0, agg)
    features = terminal.compute_features(datetime(2023, 10, 10))
    for k, v in features.items():
        assert v == 0.0, f"Expected 0.0 for {k}, got {v}"

    trx = Transaction(100, datetime(2023, 1, 9), 0, 0, True, False, predicted_label=False)
    terminal.add(trx)
    features = terminal.compute_features(trx.timestamp)
    assert features[f"{PREFIX_N_TRX}{timedelta(days=1)}"] == 1.0
    assert features[f"{PREFIX_RISK}{timedelta(days=1)}"] == 0.0
    assert features[f"{PREFIX_N_TRX}{timedelta(days=7)}"] == 1.0
    assert features[f"{PREFIX_RISK}{timedelta(days=7)}"] == 0.0

    trx = Transaction(120, datetime(2023, 1, 14, hour=17), 0, 0, True, False, predicted_label=True)
    terminal.add(trx)
    features = terminal.compute_features(trx.timestamp)
    assert features[f"{PREFIX_N_TRX}{timedelta(days=7)}"] == 2
    assert features[f"{PREFIX_RISK}{timedelta(days=7)}"] == 0.5
    assert features[f"{PREFIX_RISK}{timedelta(days=1)}"] == 1
    assert features[f"{PREFIX_N_TRX}{timedelta(days=1)}"] == 1.0

    trx = Transaction(120, datetime(2023, 1, 18, hour=18), 0, 0, True, False, predicted_label=True)
    terminal.add(trx)
    features = terminal.compute_features(trx.timestamp)
    assert features[f"{PREFIX_N_TRX}{timedelta(days=7)}"] == 2.0
    assert features[f"{PREFIX_RISK}{timedelta(days=7)}"] == 1.0
