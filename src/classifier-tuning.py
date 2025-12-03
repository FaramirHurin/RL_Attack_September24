from datetime import timedelta
import logging
import numpy as np
import optuna
import os
from banksys import Banksys
import polars as pl
import dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from parameters import ClassificationParameters, CardSimParameters
import random

USE_ANOMALY = False
WITH_MODIFICATION = False
N_REPEATS = 10
CACHE_ROOT = "cache"


def setup():
    for seed in range(N_REPEATS):
        random.seed(seed)
        np.random.seed(seed)
        cardsim = CardSimParameters.paper_params(with_modification=WITH_MODIFICATION)
        cardsim.load_simulation_data(os.path.join(CACHE_ROOT, f"seed-{seed}"))


def experiment(trial: optuna.Trial):
    total = 0.0
    for seed in range(N_REPEATS):
        random.seed(seed)
        np.random.seed(seed)
        cardsim = CardSimParameters.paper_params(with_modification=WITH_MODIFICATION)
        clf_params = ClassificationParameters.suggest(trial, use_anomaly=USE_ANOMALY)

        trx, payers, terminals = cardsim.load_simulation_data(os.path.join(CACHE_ROOT, f"seed-{seed}"))
        banksys = Banksys(trx, payers, terminals, clf_params)
        t_start = banksys.current_time
        t_end = t_start + timedelta(days=10)
        labels = banksys._transactions.filter(pl.col("timestamp").is_between(t_start, t_end))["is_fraud"].to_numpy()
        features = pl.DataFrame(banksys._fast_forward(t_end, compute_features=True, show_progress=True), schema=banksys.schema)
        predicted = banksys.clf.predict(features)

        cm = confusion_matrix(labels, predicted)
        logging.info(f"{cm}")
        f1 = f1_score(labels, predicted)
        accuracy = accuracy_score(labels, predicted)
        precision = precision_score(labels, predicted)
        recall = recall_score(labels, predicted)
        metrics = {
            "confusion_matrix": cm.tolist(),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }
        classification_report(labels, predicted)
        logging.info(f"Trial number: {trial.number} - Metrics: {metrics}")
        total += float(f1)
    return total / N_REPEATS


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    setup()
    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name=f"clf-tuning-anomaly-{USE_ANOMALY}",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment, n_trials=200, n_jobs=5)
