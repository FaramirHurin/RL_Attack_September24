# Import isolation forest
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import polars as pl


class RuleBasedClassifier:
    def __init__(
        self,
        max_values: dict[timedelta, float],
    ):
        self.max_values = max_values
        self.rule_values = max_values

    def predict(self, df: pl.DataFrame) -> npt.NDArray[np.bool]:
        labels = np.full(len(df), False, dtype=np.bool)
        for td, max_value in self.max_values.items():
            colname = f"card_n_trx_last_{td}"
            if colname not in df.columns:
                raise ValueError(f"DataFrame does not contain column for {td}.")
            y = df[colname] > max_value
            labels = labels | y.to_numpy().astype(np.bool)
        return labels
