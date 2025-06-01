from datetime import datetime, timedelta
from banksys import Card
import random


class CardRegistry:
    def __init__(self, cards: list[Card], avg_card_block_delay: timedelta):
        self.cards = cards.copy()
        self.expected_expirations = dict[Card, datetime]()
        self.actual_expirations = dict[Card, datetime]()
        self.release_dates = dict[Card, datetime]()
        self.avg_card_block_delay = avg_card_block_delay
        self.expected_lifespan = avg_card_block_delay.total_seconds()
        self.sigma = self.expected_lifespan / 5

    def release_card(self, t: datetime):
        """
        Release a random (not blocked) card and set the expiration date according to the current time.
        """
        index = random.randint(0, len(self.cards) - 1)
        card = self.cards.pop(index)
        expected_expiration = t + self.avg_card_block_delay
        self.release_dates[card] = t
        self.expected_expirations[card] = expected_expiration
        expiration_seconds = random.normalvariate(mu=self.expected_lifespan, sigma=self.sigma)
        while expiration_seconds < 0:
            expiration_seconds = random.normalvariate(mu=self.expected_lifespan, sigma=self.sigma)
        self.actual_expirations[card] = t + timedelta(seconds=expiration_seconds)
        return card

    def get_expiration(self, card: Card):
        return self.actual_expirations[card]

    def has_expired(self, card: Card, t: datetime):
        return self.expected_expirations[card] < t

    def clear(self, card: Card):
        self.expected_expirations.pop(card, None)
        self.actual_expirations.pop(card, None)
        self.release_dates.pop(card, None)

    def get_time_ratio(self, card: Card, t: datetime):
        if card not in self.expected_expirations:
            return 1.0
        expected_expiration = self.expected_expirations[card]
        remaining = expected_expiration - t
        remaining_hours = remaining.total_seconds() / 3600 / 168
        # TODO Verify, I am chnging to remaining hours
        # elapsed_seconds = self.expected_lifespan - remaining_seconds
        return remaining_hours  #  elapsed_seconds # / self.expected_lifespan
