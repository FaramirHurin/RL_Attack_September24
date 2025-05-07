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
        self.mu = avg_card_block_delay.total_seconds()
        self.sigma = self.mu / 5

    def release_card(self, t: datetime):
        """
        Release a random (valid) card and set the expiration date according to the current time.
        """
        index = random.randint(0, len(self.cards) - 1)
        card = self.cards.pop(index)

        expected_expiration = t + self.avg_card_block_delay
        self.release_dates[card] = t
        self.expected_expirations[card] = expected_expiration
        expiration_seconds = random.normalvariate(mu=self.mu, sigma=self.sigma)
        while expiration_seconds < 0:
            expiration_seconds = random.normalvariate(mu=self.mu, sigma=self.sigma)
        self.actual_expirations[card] = t + timedelta(seconds=expiration_seconds)
        return card

    def update(self, t: datetime):
        """
        Update the registry to revoke cards that have expired.
        """
        to_revoke = []
        for card_id, expiration in self.expected_expirations.items():
            if expiration < t:
                to_revoke.append(card_id)
        for card_id in to_revoke:
            self.expected_expirations.pop(card_id)
            self.actual_expirations.pop(card_id)
        return len(to_revoke)

    def has_expired(self, card: Card, t: datetime):
        if card in self.expected_expirations:
            return self.expected_expirations[card] < t
        return False

    def get_time_ratio(self, card: Card, t: datetime):
        elapsed_seconds = (t - self.release_dates[card]).total_seconds()
        return elapsed_seconds / self.mu
