from polars import DataFrame
from banksys import ClassificationSystem
import polars as pl
import numpy as np
from banksys import Banksys
from parameters import Parameters, CardSimParameters


class MockClassificationSystem(ClassificationSystem):
    def __init__(self, next_predictions: list[bool] | None = None):
        if next_predictions is None:
            next_predictions = []
        self.next_predictions = next_predictions.copy()

    def predict(self, df: DataFrame):
        predictions = self.next_predictions[: df.height]
        self.next_predictions = self.next_predictions[df.height :]
        predictions += [False] * (df.height - len(predictions))
        return np.array(predictions, dtype=bool)

    def predict_with_cause(self, df: DataFrame):
        return self.predict(df), pl.DataFrame({"Mock detection": [False]})

    def set_next_predictions(self, *p: bool):
        self.next_predictions = list(p)

    def fit(self, transactions: pl.DataFrame, is_fraud: np.ndarray):
        return


def mock_banksys():
    params = Parameters(cardsim=CardSimParameters(n_days=100, n_payers=100))
    trx, cards, terminals = params.cardsim.get_simulation_data(params.dataset_dir)
    bs = Banksys(trx, cards, terminals, params.clf_params)
    bs.clf = MockClassificationSystem()
    return bs
