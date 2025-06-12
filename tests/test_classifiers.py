from banksys.classification import RuleBasedClassifier, StatisticalClassifier
from banksys import Banksys, Transaction
from datetime import timedelta
import polars as pl
import numpy as np
from plots import Experiment, Run
from sklearn.ensemble import IsolationForest

from parameters import Parameters, CardSimParameters, ClassificationParameters


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

def test_Isolation_Forest():
    params = ClassificationParameters(
            use_anomaly=True,
            n_trees=139,
            balance_factor=0.06473635736763925,
            contamination="auto",
            training_duration=timedelta(days=150),
            quantiles={
                "amount": (0.0, 0.9976319783361984),
                "terminal_risk_last_1 day, 0:00:00": (0.0, 0.9999572867664103),
            },
            rules={
                timedelta(hours=1): 8,
                timedelta(days=1): 19,
                timedelta(weeks=1): 32,
            },
        )
    sys = Banksys.load("../src/cache/banksys/10000-payers/365-days/start-2023-01-01")
    X = sys.clf.dataset["Transactions"]
    y = sys.clf.dataset["Labels"]

    X_train, X_test = X[:8000], X[8000:]
    y_train, y_test = y[:8000], y[8000:]

    # Fit the Isolation Forest model
    clf = IsolationForest()
    clf.fit(X_train)

    y_predict = clf.predict(X_test)

    assert y_predict.shape == y_test.shape
    assert np.mean(y_predict == -1) < 0.1, "Not too many outliers predicted"
    assert np.mean(y_predict == 1) > 0.005, "Not too many normal transactions predicted"


def test_Isolation_Forest_time():
    params = ClassificationParameters(
        use_anomaly=True,
        n_trees=139,
        balance_factor=0.06473635736763925,
        contamination="auto",
        training_duration=timedelta(days=150),
        quantiles={
            "amount": (0.0, 0.9976319783361984),
            "terminal_risk_last_1 day, 0:00:00": (0.0, 0.9999572867664103),
        },
        rules={
            timedelta(hours=1): 8,
            timedelta(days=1): 19,
            timedelta(weeks=1): 32,
        },
    )
    sys = Banksys.load("../src/cache/banksys/10000-payers/365-days/start-2023-01-01")
    X = sys.clf.dataset["Transactions"]
    y = sys.clf.dataset["Labels"]

    iso = IsolationForest()
    iso.fit(X)

    start_date = sys.clf.current_time

    X_test_list = sys.simulate_until(
        start_date + timedelta(days=10),
    )

    for elem in X_test_list:
        labels = iso.predict(elem)
        assert np.mean(labels == -1) < 0.2, "Not too many outliers predicted at elem" + str(elem)
        # assert np.mean(labels == 1) > 0.005, "Not too many normal transactions predicted"
