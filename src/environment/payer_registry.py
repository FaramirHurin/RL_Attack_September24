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
        self.expected_expirations.clear()
        self.actual_expirations.clear()
        self.release_dates.clear()
        self.payers = list(self.all_payers.values())
        self.previous_frauds.clear()

    def clear(self, payer: Payer):
        self.expected_expirations.pop(payer, None)
        self.actual_expirations.pop(payer, None)
        self.release_dates.pop(payer, None)
        self.previous_frauds.pop(payer, None)

    def get_features(self, payer: Payer):
        frauds = self.previous_frauds.get(payer, [])
        n_frauds = len(frauds)
        if n_frauds == 0:
            sufficient_funds = -1.0
            total_stolen = 0.0
            latest_fraud_amount = 0.0
        else:
            sufficient_funds = float(self.sufficient_funds.get(payer, True))
            total_stolen = sum(trx.amount for trx in frauds) / 100.0
            latest_fraud_amount = frauds[-1].amount / 100.0
        return [total_stolen, float(n_frauds), latest_fraud_amount, sufficient_funds]

    def get_remaining_time_ratio(self, payer: Payer, t: datetime):
        if payer not in self.expected_expirations:
            return 1.0
        expected_expiration = self.expected_expirations[payer]
        remaining = expected_expiration - t
        elapsed_seconds = self.expected_lifespan - remaining.total_seconds()
        return 1 - (elapsed_seconds / self.expected_lifespan)

    def notify_transaction_processed(self, trx: Transaction):
        payer = self.all_payers[trx.payer_id]
        frauds = self.previous_frauds.get(payer, [])
        frauds.append(trx)
        self.previous_frauds[payer] = frauds

    def notify_insufficient_funds(self, trx: Transaction):
        payer = self.all_payers[trx.payer_id]
        self.sufficient_funds[payer] = False
