import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import overload
from ..transaction import Transaction


class StatisticalClassifier:
    """
    Classifier that classifies outliers as frauds.
    """

    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles = quantiles
        self.quantiles_df = None

    def fit(self, transactions_df: pd.DataFrame):
        assert all(f in transactions_df.columns for f in self.considered_features)
        self.quantiles_df = transactions_df[self.considered_features].quantile(q=self.quantiles)

    def predict_transaction(self, transaction: Transaction) -> bool:
        assert self.quantiles_df is not None, "The classifier has not been fitted yet."
        for feature in self.considered_features:
            value = getattr(transaction, feature)
            mmin, mmax = self.quantiles_df[feature]
            if not mmin <= value <= mmax:
                return True
        return False

    def predict_dataframe(self, df: pd.DataFrame) -> npt.NDArray[np.bool]:
        assert self.quantiles_df is not None, "The classifier has not been fitted yet."
        return df[self.considered_features].isin(self.quantiles_df).all(axis=1).to_numpy()

    @overload
    def predict(self, transaction: Transaction, /) -> bool: ...
    @overload
    def predict(self, df: pd.DataFrame, /) -> npt.NDArray[np.bool]: ...

    def predict(self, df_or_transaction, /):
        match df_or_transaction:
            case Transaction() as transaction:
                return self.predict_transaction(transaction)
            case pd.DataFrame() as df:
                return self.predict_dataframe(df)
        raise ValueError("Invalid input type. Expected Transaction or DataFrame.")
