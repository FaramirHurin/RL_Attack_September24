from banksys.classification import RuleBasedClassifier, StatisticalClassifier
from banksys import Banksys, Transaction
from datetime import timedelta
import polars as pl
import numpy as np

from parameters import Parameters, CardSimParameters


def test_rules():
    dts = [timedelta(hours=1), timedelta(days=1), timedelta(weeks=1)]
    df = pl.DataFrame({f"card_n_trx_last_{dt}": list(range(10)) for dt in dts})

    for dt in dts:
        max_amount = 2
        clf = RuleBasedClassifier({dt: max_amount})
        labels = clf.predict(df)
        assert len(labels) == 10
        for i in range(10):
            if i <= max_amount:
                assert not labels[i]
            else:
                assert labels[i]


def test_rules_details():
    df = pl.DataFrame(
        {
            f"card_n_trx_last_{timedelta(hours=1)}": list(range(10)),
            f"card_n_trx_last_{timedelta(days=1)}": list(range(10)),
            f"card_n_trx_last_{timedelta(weeks=1)}": list(range(10)),
        }
    )
    clf = RuleBasedClassifier(
        {
            timedelta(hours=1): 5,
            timedelta(days=1): 100,
            timedelta(weeks=1): 100,
        }
    )

    labels = clf.predict(df)
    assert not np.all(labels[:6])  # First 6 values should not be outliers
    assert np.all(labels[6:])  # Last 4 values exceed the daily rule

    details = clf.get_details()
    cause_hourly = details[f"Rule: card_n_trx_last_{timedelta(hours=1)} < 5"]
    assert not np.all(cause_hourly[:6])  # First 6 values should not be outliers
    assert np.all(cause_hourly[6:])  # Last 4 values should be outliers

    cause_daily = details[f"Rule: card_n_trx_last_{timedelta(days=1)} < 100"]
    assert not np.all(cause_daily)  # No rule is violated

    cause_weekly = details[f"Rule: card_n_trx_last_{timedelta(weeks=1)} < 100"]
    assert not np.all(cause_weekly)  # No rule is violated


def test_statistical():
    df = pl.DataFrame({"amount": np.arange(100) + 1})
    clf = StatisticalClassifier({"amount": (0.1, 0.9)})
    clf.fit(df)

    assert clf.quantiles_values["amount"] == (11, 90.0)

    labels = clf.predict(df)
    assert np.all(labels[:10])  # First 10 values should be outliers
    assert not np.all(labels[10:90])  # Middle 80 values should not be outliers
    assert np.all(labels[-10:])  # Last 10 values should be outliers


def test_statistical_bounds_accepted():
    df = pl.DataFrame({"amount": np.arange(100) + 1})
    clf = StatisticalClassifier({"amount": (0, 1)})
    clf.fit(df)

    assert clf.quantiles_values["amount"] == (1, 100)

    labels = clf.predict(df)
    assert not np.any(labels)
    s = np.sum(labels)
    assert s == 0, f"Expected no outliers, but found {s} outliers."
