from numpy._typing import NDArray
from polars import DataFrame
from banksys import ClassificationSystem
import polars as pl
import numpy as np
from banksys import Banksys
from parameters import Parameters, CardSimParameters
import os


class MockClassificationSystem(ClassificationSystem):
    def __init__(self):
        pass

    def predict(self, df: DataFrame):
        return np.full(df.height, False, dtype=bool)

    def predict_with_cause(self, df: DataFrame):
        return self.predict(df), pl.DataFrame({"Mock detection": [False]})


def mock_banksys(use_cache: bool = True, save: bool = True):
    file_path = os.path.join("cache", "mock-banksys")
    if use_cache:
        try:
            return Banksys.load(file_path)
        except FileNotFoundError:
            pass
    params = Parameters(cardsim=CardSimParameters(n_days=100, n_payers=100))
    trx, cards, terminals = params.cardsim.get_simulation_data()
    bs = Banksys(trx, cards, terminals, params.aggregation_windows, params.clf_params, params.terminal_fract)
    bs.clf = MockClassificationSystem()
    if save:
        bs.save(os.path.join("cache", "mock-banksys"))
    return bs
