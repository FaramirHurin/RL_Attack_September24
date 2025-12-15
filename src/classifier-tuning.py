from datetime import timedelta
import logging
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
from banksys import Banksys
import polars as pl
import dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from parameters import ClassificationParameters, CardSimParameters
from multiprocessing.pool import Pool

USE_ANOMALY = True
WITH_MODIFICATION = True
N_REPEATS = 10
POOL_SIZE = 20
CACHE_ROOT = "cache"


def compute_f1(clf: ClassificationParameters, seed: int):
    cardsim = CardSimParameters.paper_params(with_modification=WITH_MODIFICATION)
    trx, payers, terminals = cardsim.load_simulation_data(os.path.join(CACHE_ROOT, f"seed-{seed}"))
    banksys = Banksys(trx, payers, terminals, clf, silent=True)
    t_start = banksys.current_time
    t_end = t_start + timedelta(days=30)
    labels = banksys._transactions.filter(pl.col("timestamp").is_between(t_start, t_end))["is_fraud"].to_numpy()
    features = pl.DataFrame(banksys._fast_forward(t_end, compute_features=True, show_progress=False), schema=banksys.schema)
    predicted = banksys.clf.predict(features)
    f1 = float(f1_score(labels, predicted))
    metrics = {
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "f1": f1,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted)),
    }
    return f1, metrics


def objective(trial: optuna.Trial):
    clf = ClassificationParameters.suggest(trial, use_anomaly=USE_ANOMALY)
    with Pool(POOL_SIZE) as pool:
        results = pool.starmap(compute_f1, [(clf, seed) for seed in range(N_REPEATS)])
    f1s, metrics = zip(*results)
    avg_f1 = sum(f1s) / len(f1s)
    trial.set_user_attr("metrics", metrics)
    return avg_f1


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    study = optuna.create_study(
        storage=JournalStorage(JournalFileBackend(file_path="clf-tuning.journal")),
        study_name=f"anomaly-{USE_ANOMALY}-modification-{WITH_MODIFICATION}",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200, n_jobs=4)
