from datetime import timedelta

import numpy as np
import polars as pl

from banksys.payer import PREFIX_COUNT


class RuleBasedClassifier:
    def __init__(self, max_values: dict[timedelta, int]):
        self._column_values = {f"{PREFIX_COUNT}{td}": max_value for td, max_value in max_values.items()}
        self._last_result = dict[str, np.ndarray]()

    def predict(self, df: pl.DataFrame):
        # We assume that rules are based on features that are already computed in the DataFrame
        labels = np.full(df.height, False, dtype=np.bool)
        for colname, max_value in self._column_values.items():
            y = df[colname].to_numpy() > max_value
            self._last_result[f"Rule: {colname} < {max_value}"] = y
            labels = labels | y
        return labels

    def get_details(self):
        """
        Returns the details of the last prediction.
        """
        return self._last_result
