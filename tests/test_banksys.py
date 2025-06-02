from banksys import Banksys, Transaction, Card, Terminal
import polars as pl
from datetime import datetime
from parameters import Parameters, CardSimParameters, ClassificationParameters

from datetime import timedelta
from .mocks import mock_banksys


def test_invalid_dates():
    params = Parameters(
        cardsim=CardSimParameters(n_days=50, n_payers=100),
        clf_params=ClassificationParameters(training_duration=timedelta(days=30)),
        aggregation_windows=(timedelta(days=30),),
    )  # Not enough data for the classification system
    transactions, cards, terminals = params.cardsim.get_simulation_data()
    try:
        Banksys(
            transactions,
            cards,
            terminals,
            params.aggregation_windows,
            params.clf_params,
            params.terminal_fract,
        )
        assert False, "Expected ValueError for insufficient data"
    except AssertionError:
        pass


def test_balance_and_date():
    transactions = [
        # Warmup
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=False),  # 0
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 1
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 2
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 3
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 4
        Transaction(90, datetime(2023, 1, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 5
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 6
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, card_id=0, is_online=False, is_fraud=False),  # 7
        # Training data
        Transaction(170, datetime(2023, 2, 1), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 8
        Transaction(160, datetime(2023, 2, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 9
        # Test transaction (to prevent the system from crashing because there are no transactions to process)
        Transaction(140, datetime(2023, 3, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 10
    ]
    trx_df = pl.DataFrame(transactions)
    system = Banksys(
        trx_df,
        pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)]),
        pl.DataFrame([Terminal(0, 75, 95), Terminal(1, 17, 56)]),
        aggregation_windows=(timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        clf_params=ClassificationParameters(training_duration=timedelta(days=30), balance_factor=1),
        attackable_terminal_factor=1.0,
        fp_rate=0,
        fn_rate=0,
    )
    trx = transactions[-1]
    system.cards[trx.card_id].balance = 500
    system.process_transaction(trx)
    assert system.cards[trx.card_id].balance == 500 - trx.amount, "Balance should be updated after transaction"


def test_make_features():
    cards = pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)])
    terminals = pl.DataFrame([Terminal(0, 75, 95), Terminal(1, 17, 56)])

    transactions = [
        # Warmup
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=False),  # 0
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 1
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 2
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 3
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 4
        Transaction(90, datetime(2023, 1, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 5
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 6
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, card_id=0, is_online=False, is_fraud=False),  # 7
        # Training data
        Transaction(170, datetime(2023, 2, 1), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 8
        Transaction(160, datetime(2023, 2, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 9
        # Test transaction (to prevent the system from crashing because there are no transactions to process)
        Transaction(140, datetime(2023, 3, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 10
    ]
    trx_df = pl.DataFrame(transactions)
    system = Banksys(
        trx_df,
        cards,
        terminals,
        aggregation_windows=(timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        clf_params=ClassificationParameters(training_duration=timedelta(days=30), balance_factor=1),
        attackable_terminal_factor=1.0,
        fp_rate=0,
        fn_rate=0,
    )
    print(trx_df[:-1])
    # Create features for all transactions except the last one that has not been processed yet
    features = system.make_features(system._transactions_df[:-1])
    assert features.height == len(transactions) - 1
    for i, trx in enumerate(transactions[:-1]):
        assert trx.amount == features[i, "amount"]
        assert trx.is_online == features[i, "is_online"]
        assert trx.terminal_id == features[i, "terminal_id"]
        assert trx.card_id == features[i, "card_id"]

    def check_card_features(transactions: list[Transaction], features: pl.DataFrame):
        for i, trx in enumerate(transactions):
            for dt in system.aggregation_windows:
                preceding_transactions = [t for t in transactions[:i] if (t.timestamp + dt) >= trx.timestamp]
                count = len(preceding_transactions)
                mean = sum(t.amount for t in preceding_transactions) / count if count > 0 else 0.0
                fc = features[i, f"card_n_trx_last_{dt}"]
                fm = features[i, f"card_mean_amount_last_{dt}"]
                assert count == fc, f"Expected {count} transactions, got {fc} for card {trx.card_id} at index {i}"
                assert mean == fm, f"Expected mean {mean}, got {fm} for card {trx.card_id} at index {i}"

    card_0_trx = [t for t in transactions[:-1] if t.card_id == 0]
    card_1_trx = [t for t in transactions[:-1] if t.card_id == 1]
    check_card_features(card_0_trx, features.filter(pl.col("card_id") == 0))
    check_card_features(card_1_trx, features.filter(pl.col("card_id") == 1))

    def check_terminal_features(transactions: list[Transaction], features: pl.DataFrame):
        for i, trx in enumerate(transactions):
            for dt in system.aggregation_windows:
                preceding_transactions = [t for t in transactions[:i] if (t.timestamp + dt) >= trx.timestamp]
                count = len(preceding_transactions)
                risk = sum(t.is_fraud for t in preceding_transactions) / count if count > 0 else 0.0
                ft = features[i, f"terminal_n_trx_last_{dt}"]
                fr = features[i, f"terminal_risk_last_{dt}"]
                assert count == ft, f"Expected {count} transactions, got {ft} for terminal {trx.terminal_id} at index {i}"
                assert risk == fr, f"Expected risk {risk}, got {fr} for terminal {trx.terminal_id} at index {i}"

    terminal_0_trx = [t for t in transactions[:-1] if t.terminal_id == 0]
    terminal_1_trx = [t for t in transactions[:-1] if t.terminal_id == 1]
    check_terminal_features(terminal_0_trx, features.filter(pl.col("terminal_id") == 0))
    check_terminal_features(terminal_1_trx, features.filter(pl.col("terminal_id") == 1))
