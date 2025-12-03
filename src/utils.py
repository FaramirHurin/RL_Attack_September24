from typing import Any
import polars as pl
from datetime import datetime, timedelta
from dataclasses import Field
import logging
from functools import lru_cache
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


writer = None


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


def tb_log(tag: str, value: float | dict | np.floating, step: int | timedelta):
    if writer is None:
        _warn_once("TensorBoard writer is not initialized.")
        return
    if isinstance(step, timedelta):
        step = int(step.total_seconds())
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
