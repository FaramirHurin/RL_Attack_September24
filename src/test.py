from datetime import datetime, timedelta
from banksys import Transaction, Card, Terminal, Banksys
import polars as pl

from parameters import ClassificationParameters

if __name__ == "__main__":
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
        Transaction(170, datetime(2023, 2, 14), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 8
        Transaction(160, datetime(2023, 2, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 9
        Transaction(160, datetime(2023, 3, 2, hour=23, minute=59), terminal_id=0, card_id=0, is_online=False, is_fraud=True),  # 10
        # Test transaction used for later aggregation
        Transaction(140, datetime(2023, 3, 3), terminal_id=1, card_id=0, is_online=True, is_fraud=False),  # 11
        Transaction(140, datetime(2023, 3, 4), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 12
        Transaction(140, datetime(2023, 3, 5), terminal_id=1, card_id=1, is_online=True, is_fraud=False),  # 13
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

    test_trx = transactions[10:]
    for trx in test_trx:
        features = system.make_transaction_features(trx)
        assert trx.amount == features[0, "amount"]
        assert trx.is_online == features[0, "is_online"]
        assert trx.terminal_id == features[0, "terminal_id"]
        assert trx.card_id == features[0, "card_id"]

        for dt in system.aggregation_windows:
            prev_trx = [t for t in transactions if t.timestamp < trx.timestamp and t.timestamp >= trx.timestamp - dt]
            count = len(prev_trx)
            mean_amount = sum(t.amount for t in prev_trx) / count if count > 0 else 0
            feat_count = features[0, f"card_n_trx_last_{dt}"]
            feat_amount = features[0, f"card_mean_amount_last_{dt}"]
            assert count == feat_count
            assert mean_amount == feat_amount
