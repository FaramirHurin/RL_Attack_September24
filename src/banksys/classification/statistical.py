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
        self._last_predictions = dict[str, npt.NDArray[np.bool]]()  # Store last predictions for each column
        self._details_cols = {col: f"{col} > {high:.2f} or < {low:.2f}" for col, (low, high) in col_quantiles.items()}

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
            is_out_of_bounds = ~df[col].is_between(low, high, "both").to_numpy().astype(np.bool)
            self._last_predictions[col] = is_out_of_bounds
            mask |= is_out_of_bounds
        return mask

    def get_details(self) -> dict[str, npt.NDArray[np.bool]]:
        """
        Returns a dictionary with the last predictions for each column.
        The keys are the column names and the values are the boolean arrays indicating outliers.
        """
        res = {}
        for col, preds in self._last_predictions.items():
            detail_col = self._details_cols[col]
            res[detail_col] = preds
        return res
