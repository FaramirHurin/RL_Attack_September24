from typing import Any
import polars as pl
from datetime import datetime, timedelta
from dataclasses import Field
import torch
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()


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


def tb_log(tag: str, value, step: int | timedelta):
    if isinstance(step, timedelta):
        step = int(step.total_seconds())
    match value:
        case dict():
            for k, v in value.items():
                writer.add_scalar(f"{tag}/{k}", v, step)
        case _:
            writer.add_scalar(tag, value, step)
