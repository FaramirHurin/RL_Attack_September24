import copy
import os
import shutil
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


def test_simulate_until():
    """
    Test that the system indeed simulated until the given date
    """
    bs = mock_banksys()
    assert bs.next_trx.timestamp >= bs.attack_start

    bs.simulate_until(bs.attack_start + bs.max_aggregation_duration / 2)
    assert bs.next_trx.timestamp >= bs.attack_start + bs.max_aggregation_duration / 2


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
        Transaction(140, datetime(2023, 3, 11), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 10
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
    trx = transactions[-2]
    system.cards[trx.card_id].balance = 500
    system.process_transaction(trx)
    assert system.cards[trx.card_id].balance == 500 - trx.amount, "Balance should be updated after transaction"


def test_n_transacations_per_card():
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
        Transaction(140, datetime(2023, 3, 15), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 10
    ]
    trx_df = pl.DataFrame(transactions)
    system = Banksys(
        trx_df,
        pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)]),
        pl.DataFrame([Terminal(0, 75, 95), Terminal(1, 17, 56)]),
        aggregation_windows=(timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30)),
        clf_params=ClassificationParameters(training_duration=timedelta(days=30), balance_factor=1),
    )

    trx = Transaction(120, datetime(2023, 3, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=True)  # 10
    card = system.cards[trx.card_id]
    past_transactions = card.transactions.get_window().copy()
    system.process_transaction(trx, update_balance=True)
    future_transactions = card.transactions.get_window().copy()

    assert trx in future_transactions and trx not in past_transactions, "Transaction should be added to the card's transaction window"

    # Assert all transactions in window are in future transactions
    for t in past_transactions:
        if t.timestamp >= trx.timestamp - timedelta(days=30):
            assert t in future_transactions, "All transactions in the window should be in the future transactions"


def test_make_features():
    cards = pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)])
    terminals = pl.DataFrame([Terminal(0, 75, 95), Terminal(1, 17, 56)])

    transactions = [
        # Training data
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=False),
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(390, datetime(2023, 1, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        # Actual agregation
        Transaction(170, datetime(2023, 2, 14), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(160, datetime(2023, 2, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(190, datetime(2023, 3, 2, hour=23, minute=59), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        # Transaction far in the future to allow for an attack
        Transaction(190, datetime(2024, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
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

    def make_check(trx: Transaction):
        card_transactions = [t for t in transactions if t.card_id == trx.card_id]
        term_transactions = [t for t in transactions if t.terminal_id == trx.terminal_id]
        card_trx_per_agg = dict[timedelta, list[Transaction]]()
        term_trx_per_agg = dict[timedelta, list[Transaction]]()
        for delta in system.aggregation_windows:
            card_trx_per_agg[delta] = [t for t in card_transactions if trx.timestamp - delta <= t.timestamp < trx.timestamp]
            term_trx_per_agg[delta] = [t for t in term_transactions if trx.timestamp - delta <= t.timestamp < trx.timestamp]

        features = system.process_transaction(trx, update_balance=True)
        assert features.pop("amount") == trx.amount
        assert features.pop("is_online") == trx.is_online
        assert features.pop("hour") == trx.timestamp.hour
        days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        weekday_num = trx.timestamp.weekday()
        for day in days:
            if day == days[weekday_num]:
                assert features.pop(day) == 1
            else:
                assert features.pop(day) == 0

        for delta in system.aggregation_windows:
            n_trx_per_card = len(card_trx_per_agg[delta])
            mean_amount = sum(t.amount for t in card_trx_per_agg[delta]) / n_trx_per_card if n_trx_per_card > 0 else 0
            n_trx_per_term = len(term_trx_per_agg[delta])
            risk_score = sum(t.is_fraud for t in term_trx_per_agg[delta]) / n_trx_per_term if n_trx_per_term > 0 else 0
            assert features.pop(f"card_n_trx_last_{delta}") == n_trx_per_card
            assert features.pop(f"card_mean_amount_last_{delta}") == mean_amount
            assert features.pop(f"terminal_n_trx_last_{delta}") == n_trx_per_term
            assert features.pop(f"terminal_risk_last_{delta}") == risk_score
        assert len(features) == 0, f"All features should be tested but {features.keys()} remain untested"

    make_check(Transaction(180, datetime(2023, 3, 3), terminal_id=0, card_id=0, is_online=True, is_fraud=False))


def test_save_load():
    bs = mock_banksys()
    # end_date = bs.attack_start + bs.max_aggregation_duration / 2
    directory = os.path.join("cache", f"{datetime.now().isoformat().replace(':', '-')}")
    try:
        bs.save(directory)
        trx = bs.next_trx
        next_trx = next(bs.trx_iterator)

        bs2 = Banksys.load(directory)
        trx2 = bs2.next_trx
        next_trx2 = next(bs2.trx_iterator)

        assert trx == trx2, "The first transaction should be the same after loading the Banksys instance"
        assert next_trx == next_trx2, "The next transaction should be the same after loading the Banksys instance"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_aggregated_features():
    cards = pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)])
    terminals = pl.DataFrame([Terminal(index, 75, 95) for index in range(20)])

    transactions = [
        # Training data
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=False),
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(390, datetime(2023, 1, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        # Actual agregation
        Transaction(170, datetime(2023, 2, 14), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(160, datetime(2023, 2, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(190, datetime(2023, 3, 2, hour=23, minute=59), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        # Transaction far in the future to allow for an attack
        Transaction(190, datetime(2024, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
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
    trx_0 = Transaction(190, datetime(2023, 8, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=True)
    features = system.make_transaction_features(trx_0)
    aggr_1_day_0 = features.pop(f"card_n_trx_last_{timedelta(weeks=1)}")
    assert aggr_1_day_0 == 0, "There should be no transactions in the last week before processing the first transaction"
    system.process_transaction(trx_0, update_balance=False)

    for index in range(4):
        day = index + 1
        trx_1 = Transaction(200, datetime(2023, 8, 1 + day), terminal_id=index, card_id=0, is_online=False, is_fraud=True)
        features_1 = system.make_transaction_features(trx_1)
        aggr_1_day = features_1.pop(f"card_n_trx_last_{timedelta(weeks=1)}")
        system.process_transaction(trx_1, update_balance=False)
        assert aggr_1_day == aggr_1_day_0 + 1, (
            "The number of transactions in the last day should be incremented by 1 after processing a new transaction"
        )
        aggr_1_day_0 = copy.copy(aggr_1_day)


def test_balance_when_predicted_fraudulent():
    """
    We should decide what the behaviour should be when a transaction is predicted to be fraudulent
    but there is not enough balance on the card. Should we still record it to the card's transaction history?
    And to the terminal's transaction history?
    """
    assert False
