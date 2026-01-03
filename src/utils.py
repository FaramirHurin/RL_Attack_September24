import logging
from dataclasses import Field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any
from collections import deque

import numpy as np
import polars as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from banksys.transaction import Transaction

writer = None
prev_t = 0


def init_tb_logger(log_dir: str | None = None):
    global writer
    writer = SummaryWriter(log_dir)


def fields2schema(fields: list[Field]) -> dict:
    schema_dict = {}
    for field in fields:
        schema_dict[field.name] = field2pl_type(field)
    return schema_dict


def field2pl_type(field: Field) -> Any:
    if field.type is int:
        return pl.Int32
    elif field.type is float:
        return pl.Float32
    elif field.type in (bool, bool | None):
        return pl.Boolean
    elif field.type is str:
        return pl.String
    elif field.type is datetime:
        return pl.Datetime
    raise ValueError(f"Unsupported field type: {field.type}")


def get_device_by_seed(seed: int) -> torch.device:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id = seed % n_gpus
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def serialize_unknown(data):
    match data:
        case timedelta():
            return data.total_seconds()
    raise NotImplementedError(f"Unsupported serialization for type: {type(data)}. Value={data}")


@lru_cache(None)
def _warn_once(msg: str):
    logging.warning(msg)


def tb_log(tag: str, value: float | dict | np.floating, step: int | timedelta | None = None):
    if writer is None:
        _warn_once("TensorBoard writer is not initialized.")
        return
    global prev_t
    if isinstance(step, timedelta):
        step = int(step.total_seconds())
    elif step is None:
        step = prev_t
    prev_t = step
    match value:
        case float() | int() | bool() | np.floating() | np.integer():
            writer.add_scalar(tag, value, step)
        case str():
            writer.add_text(tag, value, step)
        case dict():
            for k, v in value.items():
                tb_log(f"{tag}/{k}", v, step)
        case other:
            raise ValueError(f"Unsupported type for tb_log: {type(other)}")


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
