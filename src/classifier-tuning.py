from datetime import timedelta
import logging
import numpy as np
import optuna
import os
import polars as pl
import dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from parameters import ClassificationParameters, Parameters, CardSimParameters


def experiment(trial: optuna.Trial):
    try:
        params = Parameters(
            cardsim=CardSimParameters(n_days=365, n_payers=10_000),
            clf_params=ClassificationParameters(
                training_duration=timedelta(days=90),
                n_trees=trial.suggest_int("n_trees", 20, 200),
                contamination=trial.suggest_float("contamination", 0, 0.05),
                balance_factor=trial.suggest_float("balance_factor", 0.02, 0.25),
                quantiles={
                    "amount": (
                        trial.suggest_float("quantiles_amount_low", 0.0, 0.1),
                        trial.suggest_float("quantiles_amount_high", 0.9, 1.0),
                    ),
                },
                use_anomaly=True,  # trial.suggest_categorical("use_anomaly", [True, False]),
                rules={
                    timedelta(hours=1): trial.suggest_int("max_trx_hour", 2, 10),
                    timedelta(days=1): trial.suggest_int("max_trx_day", 2, 20),
                    timedelta(weeks=1): trial.suggest_int("max_trx_week", 15, 50),
                },
            ),
        )
        banksys = params.create_banksys(use_cache=True, silent=True)
        logging.info("Producing test set via simulation")
        df_list = banksys.simulate_until(banksys.attack_start + timedelta(days=30))
        features = pl.concat(df_list)
        predicted = banksys.clf.predict(features)
        df = banksys._transactions_df.filter(pl.col("timestamp").is_between(banksys.attack_start, banksys.current_time))
        assert features["amount"].equals(df["amount"])
        truth = df["is_fraud"].to_numpy().astype(np.bool)

        metrics = {}
        f1 = f1_score(truth, predicted)
        accuracy = accuracy_score(truth, predicted)
        precision = precision_score(truth, predicted)
        recall = recall_score(truth, predicted)
        metrics = {
            "f1": float(f1),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }
        logging.info(f"Trial number: {trial.number}, parameters: {params.clf_params} with metrics: {metrics}")
        return float(f1)
    except Exception as e:
        logging.error(f"Trial number: {trial.number} failed with error: {e}")
        return 0


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs-debug.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="clf-tuning",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment, n_trials=100, n_jobs=5)
