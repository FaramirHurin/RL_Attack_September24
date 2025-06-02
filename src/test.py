from datetime import datetime, timedelta
from banksys import Transaction, Card, Terminal, Banksys
import polars as pl

from parameters import ClassificationParameters

if __name__ == "__main__":
    cards = pl.DataFrame([Card(0, 10, 25, 500), Card(1, 20, 30, 1000)])
    terminals = pl.DataFrame([Terminal(0, 75, 95), Terminal(1, 17, 56)])

    transactions = [
        Transaction(100, datetime(2023, 1, 1), terminal_id=0, card_id=0, is_online=False, is_fraud=False),
        Transaction(200, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(150, datetime(2023, 1, 2), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(120, datetime(2023, 1, 5), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(180, datetime(2023, 1, 10), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(390, datetime(2023, 1, 15), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
        Transaction(210, datetime(2023, 1, 20), terminal_id=1, card_id=1, is_online=True, is_fraud=False),
        Transaction(130, datetime(2023, 1, 30), terminal_id=0, card_id=0, is_online=False, is_fraud=True),
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
        system.process_until(trx.timestamp)

        card_transactions = [t for t in transactions if t.card_id == trx.card_id]
        term_transactions = [t for t in transactions if t.terminal_id == trx.terminal_id]
        card_trx_per_agg = dict[timedelta, list[Transaction]]()
        term_trx_per_agg = dict[timedelta, list[Transaction]]()
        for delta in system.aggregation_windows:
            card_trx_per_agg[delta] = [t for t in card_transactions if trx.timestamp - delta <= t.timestamp < trx.timestamp]
            term_trx_per_agg[delta] = [t for t in term_transactions if trx.timestamp - delta <= t.timestamp < trx.timestamp]

        _, features = system.process_transaction(trx, update_balance=True)
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
