from datetime import timedelta
import logging
import numpy as np
import optuna
import os
from banksys import Banksys
import polars as pl
import dotenv
from banksys import ClassificationSystem
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from parameters import ClassificationParameters, Parameters, CardSimParameters


CARDSIM_PARAMS = CardSimParameters(n_days=150, n_payers=10_000)
TEST_DURATION = timedelta(days=30)


def setup():
    params = Parameters(cardsim=CARDSIM_PARAMS)
    banksys = params.create_banksys(use_cache=False, silent=False, fit=False)
    # Perform the fit by hand
    banksys.fast_forward(banksys.training_start)
    features = banksys.fast_forward(banksys.attack_start)
    train_x = pl.DataFrame(features)
    train_y = banksys.training_set["is_fraud"].to_numpy().astype(np.bool)
    banksys.save("after-warmup")

    test_y = banksys._transactions_df.filter(pl.col("timestamp").is_between(banksys.attack_start, banksys.attack_start + TEST_DURATION))[
        "is_fraud"
    ]
    return train_x, train_y, test_y.to_numpy().astype(np.bool)


def experiment(trial: optuna.Trial, train_x: pl.DataFrame, train_y: np.ndarray, test_y: np.ndarray):
    try:
        params = ClassificationParameters.suggest(trial, timedelta(days=30))
        banksys = Banksys.load("after-warmup")
        banksys.clf = ClassificationSystem(params)
        banksys.clf.fit(train_x, train_y)

        dfs = banksys.simulate_until(banksys.attack_start + timedelta(days=30))
        test_x = pl.concat(dfs)
        predicted = banksys.clf.predict(test_x)

        details = banksys.clf.get_details().describe()
        with pl.Config(tbl_cols=-1):
            logging.info(details)
        metrics = {}
        cm = confusion_matrix(test_y, predicted)
        logging.info(f"{cm}")
        f1 = f1_score(test_y, predicted)
        accuracy = accuracy_score(test_y, predicted)
        precision = precision_score(test_y, predicted)
        recall = recall_score(test_y, predicted)
        metrics = {
            "confusion_matrix": cm.tolist(),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }
        classification_report(test_y, predicted)
        logging.info(f"Trial number: {trial.number} - Metrics: {metrics}")
        return float(f1)
    except Exception as e:
        logging.error(f"Trial number: {trial.number} failed with error: {e}")
        return 0.0


def main():
    train_x, train_y, test_y = setup()
    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="clf-tuning-fp0.01-fn0.01",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(lambda trial: experiment(trial, train_x, train_y, test_y), n_trials=200, n_jobs=5)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
