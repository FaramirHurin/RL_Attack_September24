from copy import deepcopy
from datetime import timedelta
import logging
import time
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
from banksys import Banksys
import polars as pl
import dotenv
from sklearn.metrics import f1_score
from parameters import ClassificationParameters, CardSimParameters
from multiprocessing.pool import Pool

USE_ANOMALY = False
WITH_MODIFICATION = False
N_REPEATS = 10
POOL_SIZE = 20
CACHE_ROOT = "cache"


def e2(clf: ClassificationParameters, seed: int):
    cardsim = CardSimParameters.paper_params(with_modification=WITH_MODIFICATION)
    trx, payers, terminals = cardsim.load_simulation_data(os.path.join(CACHE_ROOT, f"seed-{seed}"))
    banksys = Banksys(trx, payers, terminals, clf, silent=False)
    t_start = banksys.current_time
    t_end = t_start + timedelta(days=30)
    labels = banksys._transactions.filter(pl.col("timestamp").is_between(t_start, t_end))["is_fraud"].to_numpy()
    features = pl.DataFrame(banksys._fast_forward(t_end, compute_features=True, show_progress=False), schema=banksys.schema)
    predicted = banksys.clf.predict(features)
    f1 = f1_score(labels, predicted)
    return float(f1)


def e1(clf: ClassificationParameters) -> float:
    with Pool(POOL_SIZE) as pool:
        results = pool.starmap(e2, [(deepcopy(clf), seed) for seed in range(N_REPEATS)])
    return sum(results) / len(results)


def objective(trial: optuna.Trial):
    clf_params = ClassificationParameters.suggest(trial, use_anomaly=USE_ANOMALY)
    return e1(clf_params)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    study = optuna.create_study(
        storage=JournalStorage(JournalFileBackend(file_path="journal.log")),
        study_name=f"clf-tuning-anomaly-{USE_ANOMALY}",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, n_jobs=1)
