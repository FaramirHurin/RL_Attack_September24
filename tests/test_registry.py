from environment.payer_registry import PayerRegistry
from banksys import Payer
from datetime import timedelta, datetime

AGG_WINDOWS = (timedelta(hours=12),)


def test_time_ratio():
    payers = [
        Payer(0, 0, 0, 20, AGG_WINDOWS),
        Payer(1, 15, 36, 40, AGG_WINDOWS),
        Payer(2, 31, 74, 23, AGG_WINDOWS),
        Payer(3, 87, 65, 18, AGG_WINDOWS),
    ]
    registry = PayerRegistry(payers, timedelta(days=1), datetime(2023, 1, 1))
    t = datetime(2023, 1, 1)
    card = registry.release_payer(t)
    assert registry._get_remaining_time_ratio(card, t) == 1.0
    assert registry._get_remaining_time_ratio(card, t + timedelta(hours=12)) == 0.5
    assert registry._get_remaining_time_ratio(card, t + timedelta(days=1)) == 0.0
    assert registry._get_remaining_time_ratio(card, t + timedelta(days=1, hours=12)) == -0.5
    assert registry._get_remaining_time_ratio(card, t + timedelta(days=2)) == -1


def test_no_doubles():
    payers = [Payer(i, 0, 0, 20, AGG_WINDOWS) for i in range(2)]
    registry = PayerRegistry(payers, timedelta(days=7), datetime(2023, 1, 1))
    t1 = datetime(2023, 1, 2)
    t2 = datetime(2023, 1, 2)
    for _ in range(1_000):
        registry.reset()
        payer1 = registry.release_payer(t1)
        assert registry.n_currently_released == 1
        payer2 = registry.release_payer(t2)
        assert registry.n_currently_released == 2
        assert payer1 != payer2
