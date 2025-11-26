from datetime import datetime, timedelta
from banksys import Payer, Transaction
import random


class PayerRegistry:
    def __init__(self, payers: list[Payer], avg_block_delay: timedelta):
        self.all_payers = {payer.id: payer for payer in payers}
        self.payers = payers.copy()
        self.expected_expirations = dict[Payer, datetime]()
        self.actual_expirations = dict[Payer, datetime]()
        self.release_dates = dict[Payer, datetime]()
        self.avg_block_delay = avg_block_delay
        self.expected_lifespan = avg_block_delay.total_seconds()
        self.sigma = self.expected_lifespan / 5
        self.previous_frauds = dict[Payer, list[Transaction]]()
        self.sufficient_funds = dict[Payer, bool]()
        self.balance_upper_bound = dict[Payer, float]()

    def release_payer(self, t: datetime):
        """
        Release a random (not blocked) payer and set the expiration date according to the current time.
        """
        index = random.randint(0, len(self.payers) - 1)
        payer = self.payers.pop(index)
        expected_expiration = t + self.avg_block_delay
        self.release_dates[payer] = t
        self.expected_expirations[payer] = expected_expiration
        expiration_seconds = random.normalvariate(mu=self.expected_lifespan, sigma=self.sigma)
        while expiration_seconds < 0:
            expiration_seconds = random.normalvariate(mu=self.expected_lifespan, sigma=self.sigma)
        self.actual_expirations[payer] = t + timedelta(seconds=expiration_seconds)
        return payer

    def get_expiration(self, payer: Payer):
        return self.actual_expirations[payer]

    def has_expired(self, payer: Payer, t: datetime):
        return self.expected_expirations[payer] < t

    def reset(self):
        self.payers = list(self.all_payers.values())
        self.expected_expirations.clear()
        self.actual_expirations.clear()
        self.release_dates.clear()
        self.previous_frauds.clear()
        self.sufficient_funds.clear()
        self.balance_upper_bound.clear()

    def clear(self, payer: Payer):
        self.expected_expirations.pop(payer, None)
        self.actual_expirations.pop(payer, None)
        self.release_dates.pop(payer, None)
        self.previous_frauds.pop(payer, None)
        self.sufficient_funds.pop(payer, None)
        self.balance_upper_bound.pop(payer, None)

    def get_features(self, payer: Payer, t: datetime):
        successful_frauds = self.previous_frauds.get(payer, [])
        balance_upper_bound = self.balance_upper_bound.get(payer, None)
        n_attempts = len(successful_frauds)
        if balance_upper_bound is not None:
            n_attempts += 1
        if len(successful_frauds) == 0:
            total_stolen = 0.0
        else:
            total_stolen = sum(trx.amount for trx in successful_frauds)
        return [
            self._get_remaining_time_ratio(payer, t),
            total_stolen / 100.0,
            float(n_attempts),
            balance_upper_bound / 100.0 if balance_upper_bound is not None else -1.0,
        ]

    def _get_remaining_time_ratio(self, payer: Payer, t: datetime):
        if payer not in self.expected_expirations:
            return 1.0
        expected_expiration = self.expected_expirations[payer]
        remaining = expected_expiration - t
        elapsed_seconds = self.expected_lifespan - remaining.total_seconds()
        return 1 - (elapsed_seconds / self.expected_lifespan)

    def notify_transaction_processed(self, trx: Transaction):
        payer = self.all_payers[trx.payer_id]
        if payer not in self.previous_frauds:
            self.previous_frauds[payer] = [trx]
        else:
            self.previous_frauds[payer].append(trx)
        if payer in self.balance_upper_bound:
            self.balance_upper_bound[payer] -= trx.amount

    def notify_insufficient_funds(self, trx: Transaction):
        payer = self.all_payers[trx.payer_id]
        self.sufficient_funds[payer] = False
        self.balance_upper_bound[payer] = trx.amount
