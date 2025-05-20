from .transaction import Transaction
from datetime import datetime
from typing import Optional


class OrderedTransactionsRegistry:
    def __init__(self, transactions: Optional[list[Transaction]] = None):
        if transactions is None:
            transactions = []
        else:
            transactions = sorted(transactions, key=lambda t: t.timestamp)
        self.transactions = transactions

    def add_transaction(self, transaction: Transaction):
        index = self._find_index(transaction.timestamp)
        self.transactions.insert(index, transaction)

    def get_before(self, timestamp: datetime) -> list[Transaction]:
        """
        Get all transactions before the given timestamp.
        """
        index = self._find_index(timestamp)
        return self.transactions[:index]

    def get_after(self, timestamp: datetime) -> list[Transaction]:
        """
        Get all transactions after the given timestamp.
        """
        index = self._find_index(timestamp)
        return self.transactions[index:]

    def get_between(self, start: datetime, end: datetime) -> list[Transaction]:
        """
        Get all transactions between the given start and end timestamps.
        """
        start_index = self._find_index(start)
        end_index = self._find_index(end)
        return self.transactions[start_index:end_index]

    def remove(self, transaction: Transaction):
        """
        Remove the transaction from the list.
        """
        index = self._find_index(transaction.timestamp)
        return self.transactions.pop(index)

    def _find_index(self, timestamp: datetime) -> int:
        """
        Find the index where the transaction should be inserted to keep the list sorted by timestamp.
        """
        low = 0
        high = len(self.transactions)
        while low < high:
            mid = (low + high) // 2
            if self.transactions[mid].timestamp < timestamp:
                low = mid + 1
            else:
                high = mid
        return low

    def __getitem__(self, index: int) -> Transaction:
        return self.transactions[index]

    def __len__(self) -> int:
        return len(self.transactions)

    def __iter__(self):
        return iter(self.transactions)
