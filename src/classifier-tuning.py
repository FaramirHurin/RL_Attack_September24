from datetime import timedelta
import logging
import numpy as np
import optuna
import os
import polars as pl
import dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from parameters import ClassificationParameters, Parameters, CardSimParameters


def experiment(trial: optuna.Trial):
    try:
        params = Parameters(
            cardsim=CardSimParameters(n_days=150, n_payers=10_000),
            clf_params=ClassificationParameters.suggest(trial, timedelta(days=30)),
        )
        trial_dir = f"trial_{trial.number}"
        banksys = params.create_banksys(use_cache=False, silent=False)
        logging.info("Producing test set via simulation")
        df_list = banksys.simulate_until(banksys.attack_start + timedelta(days=30))
        logging.info("Done")
        features = pl.concat(df_list)
        os.makedirs(trial_dir, exist_ok=True)
        df = banksys._transactions_df.filter(pl.col("timestamp").is_between(banksys.attack_start, banksys.current_time))
        features.write_csv(os.path.join(trial_dir, "features.csv"))
        df.write_csv(os.path.join(trial_dir, "transactions.csv"))

        predicted = banksys.clf.predict(features)
        details = banksys.clf.get_details()
        print(details)
        print(details.describe())
        pl.DataFrame({"is_fraud": predicted}).write_csv(os.path.join(trial_dir, "predicted.csv"))
        assert features["amount"].equals(df["amount"])
        truth = df["is_fraud"].to_numpy().astype(np.bool)

        metrics = {}
        cm = confusion_matrix(truth, predicted)
        logging.info(f"{cm}")
        f1 = f1_score(truth, predicted)
        accuracy = accuracy_score(truth, predicted)
        precision = precision_score(truth, predicted)
        recall = recall_score(truth, predicted)
        metrics = {
            "confusion_matrix": cm.tolist(),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }
        logging.info(f"Trial number: {trial.number}, parameters: {params.clf_params} with metrics: {metrics}")
        return float(f1)
    except Exception as e:
        logging.error(f"Trial number: {trial.number} failed with error: {e}")
        return 0.0


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="clf-tuning",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment, n_trials=200, n_jobs=1)
