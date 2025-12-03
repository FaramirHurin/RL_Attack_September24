from datetime import timedelta
import numpy as np
import polars as pl

from banksys.payer import Payer


class RuleBasedClassifier:
    def __init__(self, max_values: dict[timedelta, int]):
        self._column_values = {f"{Payer.colname('count', td)}": max_value for td, max_value in max_values.items()}
        self._rule_names = [self.rule_name(td) for td in max_values.keys()]
        self._last_result = dict[str, np.ndarray]()

    def predict(self, df: pl.DataFrame):
        # We assume that rules are based on features that are already computed in the DataFrame
        labels = np.full(df.height, False, dtype=np.bool)
        for (colname, max_value), rule_name in zip(self._column_values.items(), self._rule_names):
            y = df[colname].to_numpy() > max_value
            self._last_result[rule_name] = y
            labels = labels | y
        return labels

    def get_details(self):
        """
        Returns the details of the last prediction.
        """
        return self._last_result

    def rule_name(self, window: timedelta) -> str:
        colname = Payer.colname("count", window)
        return f"{colname} < {self._column_values[colname]}"
