from .transaction import Transaction
from datetime import datetime


class HasOrderedTransactions:
    def __init__(self):
        self.transactions = list[Transaction]()

    def add_transaction(self, transaction: Transaction):
        index = self._find_index(transaction.timestamp)
        self.transactions.insert(index, transaction)

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
