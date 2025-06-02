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
        return np.full(df.height, False, dtype=bool), pl.DataFrame({"Mock detection": [False]})


def mock_banksys():
    file_path = os.path.join("cache", "mock-banksys")
    try:
        return Banksys.load(file_path)
    except FileNotFoundError:
        pass
    params = Parameters(cardsim=CardSimParameters(n_days=100, n_payers=100))
    trx, cards, terminals = params.cardsim.get_simulation_data()
    bs = Banksys(trx, cards, terminals, params.aggregation_windows, params.clf_params, params.terminal_fract)
    bs.clf = MockClassificationSystem()
    bs.save(os.path.join("cache", "mock-banksys"))
    return bs
