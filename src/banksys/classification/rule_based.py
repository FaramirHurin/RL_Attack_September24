from datetime import timedelta

import numpy as np
import polars as pl


class RuleBasedClassifier:
    def __init__(self, max_values: dict[timedelta, float]):
        self._column_values = {f"card_n_trx_last_{td}": max_value for td, max_value in max_values.items()}
        self._last_result = dict[str, np.ndarray]()

    def predict(self, df: pl.DataFrame):
        # We assume that rules are based on features that are already computed in the DataFrame
        labels = np.full(len(df), False, dtype=np.bool)
        for colname, max_value in self._column_values.items():
            y = df[colname] > max_value
            y = y.to_numpy().astype(np.bool)
            self._last_result[f"Rule: {colname} < {max_value}"] = y
            labels = labels | y
        return labels

    def get_details(self):
        """
        Returns the details of the last prediction.
        """
        return self._last_result
