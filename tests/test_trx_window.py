from banksys import TransactionWindow, Transaction
from datetime import datetime, timedelta
import random


def trx_with_timestamp(t: datetime, is_fraud=False):
    return Transaction(random.random() * 10, t, 0, 0, True, is_fraud)


def test_window_count_within_window():
    t_end = datetime(2022, 5, 5)
    transactions = [
        trx_with_timestamp(t_end - timedelta(weeks=2)),
        trx_with_timestamp(t_end - timedelta(hours=2)),
        trx_with_timestamp(t_end - timedelta(minutes=2)),
        trx_with_timestamp(t_end),
    ]
    window = TransactionWindow([timedelta(seconds=10), timedelta(hours=1), timedelta(days=1), timedelta(weeks=1), timedelta(weeks=4)])
    for trx in transactions:
        window.add(trx)

    counts = window.compute_counts_by_window(t_end)
    assert counts == [1, 2, 3, 3, 4]


def test_terminal_count_within_window_empty():
    agg_windows = [timedelta(seconds=10), timedelta(hours=1), timedelta(days=1), timedelta(weeks=1), timedelta(weeks=4)]
    window = TransactionWindow(agg_windows)
    counts = window.compute_counts_by_window(datetime(2027, 5, 5))
    assert len(counts) == len(agg_windows)
    assert all(c == 0 for c in counts)


def test_add():
    window = TransactionWindow([timedelta(days=1), timedelta(hours=1)])
    assert len(window) == 0

    window.add(trx_with_timestamp(datetime(2023, 1, 1)), update=True)
    assert len(window) == 1
    window.add(trx_with_timestamp(datetime(2023, 1, 1, hour=2)), update=True)
    window.add(trx_with_timestamp(datetime(2023, 1, 1, hour=15)), update=True)
    window.add(trx_with_timestamp(datetime(2023, 1, 1, hour=18)), update=True)
    assert len(window) == 4

    window.add(trx_with_timestamp(datetime(2023, 1, 2, hour=16)), update=True)
    assert len(window) == 2


def test_count_and_mean():
    agg = (timedelta(days=1), timedelta(days=7))
    window = TransactionWindow(agg)
    amounts, counts = window.compute_avg_amount_and_count_by_window(datetime(2023, 1, 10))
    assert all(a == 0 for a in amounts)
    assert all(c == 0 for c in counts)

    window.add(Transaction(40, datetime(2023, 1, 7), 0, 0, True, False, predicted_label=False))
    window.add(Transaction(100, datetime(2023, 1, 9), 0, 0, True, False, predicted_label=False))
    amounts, counts = window.compute_avg_amount_and_count_by_window(datetime(2023, 1, 9, hour=12))
    assert amounts == [100, 70]
    assert counts == [1, 2]

    t = datetime(2023, 1, 12)
    window.update(t)
    amounts, counts = window.compute_avg_amount_and_count_by_window(t)
    assert amounts == [0, 70]
    assert counts == [0, 2]

    window.add(Transaction(130, datetime(2023, 1, 13, hour=17), 0, 0, True, False, predicted_label=True))
    amounts, counts = window.compute_avg_amount_and_count_by_window(datetime(2023, 1, 13, hour=21))
    assert amounts == [130, 90]
    assert counts == [1, 3]

    t = datetime(2023, 1, 15, hour=18)
    window.update(t)
    amounts, counts = window.compute_avg_amount_and_count_by_window(t)
    assert amounts == [0, 115]
    assert counts == [0, 2]

    t = datetime(2023, 1, 25, hour=18)
    window.update(t)
    amounts, counts = window.compute_avg_amount_and_count_by_window(t)
    assert amounts == [0, 0]
    assert counts == [0, 0]
