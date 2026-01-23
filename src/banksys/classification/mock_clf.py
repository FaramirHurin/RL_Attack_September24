import numpy as np
import polars as pl


class MockClf:
    def __init__(self, name: str):
        self.name = name
        self.last_result = np.zeros(0, dtype=bool)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        self.last_result = np.zeros(df.height, dtype=bool)
        return self.last_result

    def fit(self, df: pl.DataFrame, labels: np.ndarray):
        return

    def get_details(self):
        return {self.name: self.last_result}
