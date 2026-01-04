from banksys.transaction import Transaction
import polars as pl
from typing import Any


class TransactionIterator:
    def __init__(self, transactions: pl.DataFrame):
        self._iterator = transactions.iter_rows(named=True)
        self._cache: Transaction | None = None
        self._current: Transaction | None = None
        self._prev: Transaction | None = None
        self._df = transactions

    def peek(self):
        if self._cache is None:
            self._cache = Transaction(**next(self._iterator))
        return self._cache

    @property
    def prev(self):
        return self._prev

    def next(self):
        self._prev = self._current
        if self._cache is not None:
            self._current = self._cache
        else:
            self._current = Transaction(**next(self._iterator))
        self._cache = None
        return self._current

    def __next__(self):
        return self.next()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_iterator"]
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        if self._current is not None:
            remaining_trx = self._df.filter(pl.col("timestamp") > self._current.timestamp)
        else:
            remaining_trx = self._df
        self._iterator = remaining_trx.iter_rows(named=True)
