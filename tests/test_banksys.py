import os
import random
import shutil
from datetime import datetime, timedelta

import polars as pl

from banksys import Banksys, Payer, Terminal, Transaction
from parameters import CardSimParameters, ClassificationParameters, Parameters

from .mocks import MockClassificationSystem, mock_banksys


def make_trx(
    t: datetime,
    *,
    amount: float | None = None,
    payer: int = 0,
    terminal: int = 0,
    is_fraud: bool | None = None,
    is_online: bool | None = None,
    is_credit: bool = False,
):
    if amount is None:
        amount = random.random() * 100
    if is_fraud is None:
        is_fraud = random.random() > 0.5
    if is_online is None:
        is_online = random.random() > 0.5
    return Transaction(
        amount=amount,
        timestamp=t,
        terminal_id=terminal,
        payer_id=payer,
        is_fraud=is_fraud,
        is_online=is_online,
        is_credit=is_credit,
    )


def test_invalid_dates():
    params = Parameters(
        cardsim=CardSimParameters(n_days=50, n_payers=100),
        clf_params=ClassificationParameters(
            float_training_duration_s=timedelta(days=30),
            float_aggregation_windows_s=(timedelta(days=30),),
        ),
    )  # Not enough data for the classification system
    transactions, cards, terminals = params.cardsim.load_simulation_data(params.dataset_dir)
    try:
        Banksys(
            transactions,
            cards,
            terminals,
            params.clf_params,
        )
        assert False, "Expected ValueError for insufficient data"
    except AssertionError:
        pass


def test_fast_forward():
    """
    Test that the system indeed simulated until the given date
    """
    params = Parameters(cardsim=CardSimParameters(n_days=100, n_payers=100))
    trx, cards, terminals = params.cardsim.load_simulation_data(params.dataset_dir)
    bs = Banksys(trx, cards, terminals, params.clf_params)
    bs.clf = MockClassificationSystem()
    assert bs.next_trx.timestamp >= bs.attack_start

    max_window = params.clf_params.longest_window

    bs._fast_forward(bs.attack_start + max_window / 2, compute_features=False)
    assert bs.next_trx.timestamp >= bs.attack_start + max_window / 2


def test_balance_and_date():
    AGG_WINDOWS = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30))
    transactions = [
        # Warmup
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, payer_id=0, is_online=False, is_fraud=False),  # 0
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 1
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 2
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 3
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 4
        Transaction(90, datetime(2023, 1, 15), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 5
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 6
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, payer_id=0, is_online=False, is_fraud=False),  # 7
        # Training data
        Transaction(170, datetime(2023, 2, 1), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 8
        Transaction(160, datetime(2023, 2, 5), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 9
        # Test transaction (to prevent the system from crashing because there are no transactions to process)
        Transaction(140, datetime(2023, 3, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=True),  # 10
        Transaction(140, datetime(2023, 3, 11), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 10
    ]
    trx_df = pl.DataFrame(transactions)
    system = Banksys(
        trx_df,
        pl.DataFrame([Payer(0, 10, 25, 500, AGG_WINDOWS), Payer(1, 20, 30, 1000, AGG_WINDOWS)]),
        pl.DataFrame([Terminal(0, 75, 95, AGG_WINDOWS), Terminal(1, 17, 56, AGG_WINDOWS)]),
        params=ClassificationParameters(
            float_training_duration_s=timedelta(days=30),
            balance_factor=1,
            fp_rate=0.0,
            float_aggregation_windows_s=AGG_WINDOWS,
        ),
    )
    system.clf = MockClassificationSystem()
    trx = transactions[-2]
    system.payers[trx.payer_id].balance = 500.0
    system.process_transaction(trx)
    assert system.payers[trx.payer_id].balance == 500 - trx.amount, "Balance should be updated after transaction"


def test_n_transacations_per_card():
    AGG_WINDOWS = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30))
    transactions = [
        # Warmup
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, payer_id=0, is_online=False, is_fraud=False),  # 0
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 1
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 2
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 3
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 4
        Transaction(90, datetime(2023, 1, 15), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 5
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 6
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, payer_id=0, is_online=False, is_fraud=False),  # 7
        # Training data
        Transaction(170, datetime(2023, 2, 1), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 8
        Transaction(160, datetime(2023, 2, 5), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),  # 9
        # Test transaction (to prevent the system from crashing because there are no transactions to process)
        Transaction(140, datetime(2023, 3, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 10
        Transaction(140, datetime(2023, 3, 15), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),  # 10
    ]
    trx_df = pl.DataFrame(transactions)
    system = Banksys(
        trx_df,
        pl.DataFrame([Payer(0, 10, 25, 500, AGG_WINDOWS), Payer(1, 20, 30, 1000, AGG_WINDOWS)]),
        pl.DataFrame([Terminal(0, 75, 95, AGG_WINDOWS), Terminal(1, 17, 56, AGG_WINDOWS)]),
        params=ClassificationParameters(float_training_duration_s=timedelta(days=30), balance_factor=1),
    )
    system.clf = MockClassificationSystem()

    trx = Transaction(120, datetime(2023, 3, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=True)  # 10
    payer = system.payers[trx.payer_id]
    past_transactions = payer._window.transactions.copy()
    system.process_transaction(trx)
    future_transactions = payer._window.transactions.copy()

    assert trx in future_transactions and trx not in past_transactions, "Transaction should be added to the card's transaction window"

    # Assert all transactions in window are in future transactions
    for t in past_transactions:
        if t.timestamp >= trx.timestamp - timedelta(days=30):
            assert t in future_transactions, "All transactions in the window should be in the future transactions"


def test_make_features():
    AGG_WINDOWS = (timedelta(hours=1), timedelta(days=1), timedelta(days=7), timedelta(days=30))
    cards = pl.DataFrame([Payer(0, 10, 25, 500, AGG_WINDOWS), Payer(1, 20, 30, 1000, AGG_WINDOWS)])
    terminals = pl.DataFrame([Terminal(0, 75, 95, AGG_WINDOWS), Terminal(1, 17, 56, AGG_WINDOWS)])
    # End of the warmup on the 2023-01-10 + 30 days = 2023-01-31
    warmup_trx = [make_trx(datetime(2023, 1, 1))]
    # End of the training on the 2023-01-31 + 30 days = 2023-03-02
    train_trx = [
        make_trx(datetime(2023, 1, 31, hour=10), is_fraud=False),
        make_trx(datetime(2023, 2, 14), amount=170, payer=1, terminal=1, is_fraud=False),
        make_trx(datetime(2023, 2, 15), amount=160, payer=0, terminal=1, is_fraud=True),
        make_trx(datetime(2023, 3, 1, hour=23, minute=50), amount=190, payer=0, terminal=0, is_fraud=False),
    ]
    # Actual transactions to start from the 2023-03-02 00:00:00
    attack_trx = [make_trx(t=datetime(2023, 3, 2, hour=0, minute=10), is_fraud=True, payer=0, terminal=0)]
    transactions = [
        *warmup_trx,
        *train_trx,
        *attack_trx,
        # Transaction far in the future to avoid crashing
        make_trx(datetime(2124, 1, 1)),
    ]
    trx_df = pl.DataFrame(transactions)
    clf = MockClassificationSystem([t.is_fraud for t in warmup_trx] + [t.is_fraud for t in train_trx])
    system = Banksys(
        trx_df,
        cards,
        terminals,
        params=ClassificationParameters(
            balance_factor=1,
            float_aggregation_windows_s=AGG_WINDOWS,
            float_training_duration_s=timedelta(days=30),
        ),
        clf=clf,
    )

    def verify(trx: Transaction):
        features = system.process_transaction(trx)
        trx.predicted_label = True
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

        for delta in AGG_WINDOWS:
            payer_transactions = list[Transaction]()
            for t in train_trx:
                if t.is_fraud:  # Frauds are not added to the payer's history
                    continue
                if t.payer_id != trx.payer_id:
                    continue
                if t.timestamp >= trx.timestamp - delta:
                    payer_transactions.append(t)
            assert features.pop(Payer.colname("count", delta)) == len(payer_transactions)
            if len(payer_transactions) == 0:
                avg_amount = 0.0
            else:
                avg_amount = sum(t.amount for t in payer_transactions) / len(payer_transactions)
            assert features.pop(Payer.colname("avg", delta)) == avg_amount

            term_transactions = list[Transaction]()
            for t in train_trx:
                if t.terminal_id == trx.terminal_id and t.timestamp >= trx.timestamp - delta:
                    term_transactions.append(t)
            assert features.pop(Terminal.colname("count", delta)) == len(term_transactions)
            if len(term_transactions) == 0:
                risk_score = 0.0
            else:
                risk_score = sum(t.fraud_is_detected for t in term_transactions) / len(term_transactions)
            assert features.pop(Terminal.colname("risk", delta)) == risk_score

        assert len(features) == 0, f"All features should be tested but {features.keys()} remain untested"

    for trx in attack_trx:
        verify(trx)


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
    AGG_WINDOWS = (timedelta(weeks=1),)
    payers = pl.DataFrame([Payer(0, 10, 25, balance=990, agg_windows=AGG_WINDOWS), Payer(1, 20, 30, 1000, AGG_WINDOWS)])
    terminals = pl.DataFrame([Terminal(index, 75, 95, AGG_WINDOWS) for index in range(20)])

    transactions = [
        # Training data
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, payer_id=0, is_online=False, is_fraud=False),
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),
        Transaction(390, datetime(2023, 1, 15), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
        # Actual agregation
        Transaction(170, datetime(2023, 2, 14), terminal_id=1, payer_id=1, is_online=True, is_fraud=False),
        Transaction(160, datetime(2023, 2, 15), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
        Transaction(190, datetime(2023, 3, 2, hour=23, minute=59), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
        # Transaction far in the future to allow for an attack
        Transaction(190, datetime(2024, 1, 1), terminal_id=0, payer_id=0, is_online=False, is_fraud=True),
    ]
    trx_df = pl.DataFrame(transactions)
    clf = MockClassificationSystem()
    system = Banksys(
        trx_df,
        payers,
        terminals,
        params=ClassificationParameters(
            float_training_duration_s=timedelta(days=30),
            balance_factor=1,
            float_aggregation_windows_s=AGG_WINDOWS,
        ),
        clf=clf,
    )

    system.payers[0].balance = 10_000
    KEY_PAYER = Payer.colname("count", timedelta(weeks=1))
    KEY_TERM = Terminal.colname("count", timedelta(weeks=1))
    START_DATE = datetime(2023, 8, 1)
    for delta_days in range(6):
        hour = random.randint(0, 23)
        is_online = random.random() > 0.5
        features = system.process_transaction(
            Transaction(
                10,
                START_DATE + timedelta(days=delta_days, hours=hour),
                terminal_id=0,
                payer_id=0,
                is_online=is_online,
                is_fraud=True,
            )
        )
        assert features[KEY_PAYER] == delta_days, "At day N, the number of weekly transactions for the payer should be N"
        assert features[KEY_TERM] == delta_days, "At day N, the number of weekly transactions for the terminal should be N"
        assert features["hour"] == hour, "The hour feature should match the transaction hour"
        assert features["is_online"] == is_online, "The is_online feature should match the transaction is_online value"
