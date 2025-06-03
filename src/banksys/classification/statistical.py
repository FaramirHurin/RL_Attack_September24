import numpy as np
import numpy.typing as npt
import polars as pl


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """

    def __init__(self, col_quantiles: dict[str, tuple[float, float]]):
        self.quantiles = col_quantiles
        self.quantiles_values = dict[str, tuple[float, float]]()  # Dictionary to hold quantiles for each considered

    def fit(self, df: pl.DataFrame):
        # Compute the quantiles lower and upper bounds for each column
        for col, (low, high) in self.quantiles.items():
            low_quantile = df[col].quantile(low)
            high_quantile = df[col].quantile(high)
            assert low_quantile is not None
            assert high_quantile is not None
            self.quantiles_values[col] = (low_quantile, high_quantile)

    def predict(self, df: pl.DataFrame) -> npt.NDArray[np.bool]:
        mask = np.full(df.height, False, dtype=np.bool)
        for col, (low, high) in self.quantiles_values.items():
            is_out_of_bounds = ~df[col].is_between(low, high).to_numpy().astype(np.bool)
            mask |= is_out_of_bounds
        return mask
