from environment.card_registry import CardRegistry
from banksys import Card
from datetime import timedelta, datetime


def test_time_ratio():
    cards = [
        Card(0, 0, 0, 20, True),
        Card(1, 15, 36, 40, False),
        Card(2, 31, 74, 23, True),
        Card(3, 87, 65, 18, False),
    ]
    registry = CardRegistry(cards, timedelta(days=1))
    t = datetime(2023, 1, 1)
    card = registry.release_card(t)
    assert registry.get_remaining_time_ratio(card, t) == 1.0
    assert registry.get_remaining_time_ratio(card, t + timedelta(hours=12)) == 0.5
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=1)) == 0.0
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=1, hours=12)) == -0.5
    assert registry.get_remaining_time_ratio(card, t + timedelta(days=2)) == -1
