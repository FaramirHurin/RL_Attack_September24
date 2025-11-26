from environment.payer_registry import PayerRegistry
from banksys import Payer
from datetime import timedelta, datetime


def test_time_ratio():
    payers = [
        Payer(0, 0, 0, 20),
        Payer(1, 15, 36, 40),
        Payer(2, 31, 74, 23),
        Payer(3, 87, 65, 18),
    ]
    registry = PayerRegistry(payers, timedelta(days=1))
    t = datetime(2023, 1, 1)
    card = registry.release_payer(t)
    assert registry.get_remaining_time_ratio(card, t) == 1.0
    assert registry.get_remaining_time_ratio(card, t + timedelta(hours=12)) == 0.5
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=1)) == 0.0
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=1, hours=12)) == -0.5
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=2)) == -1
