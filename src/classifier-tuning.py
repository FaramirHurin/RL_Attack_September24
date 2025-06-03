from datetime import timedelta
import logging
import numpy as np
import optuna
import os
import polars as pl
from tqdm import tqdm
import dotenv
from sklearn.metrics import f1_score
from parameters import ClassificationParameters, Parameters, CardSimParameters
from banksys import Transaction


PARAMS = Parameters(
    cardsim=CardSimParameters(n_days=365, n_payers=10_000),
    clf_params=ClassificationParameters(training_duration=timedelta(days=90)),
)


def get_params(with_rules: bool, trial: optuna.Trial):
    params = ClassificationParameters(
        n_trees=trial.suggest_int("n_trees", 20, 200),
        contamination=trial.suggest_float("contamination", 0.0001, 0.05),
        balance_factor=trial.suggest_float("balance_factor", 0.001, 0.25),
        quantiles_values=[
            trial.suggest_float("quantiles_low", 0.0, 0.1),
            trial.suggest_float("quantiles_high", 0.9, 1.0),
        ],
        use_anomaly=trial.suggest_categorical("use_anomaly", [True, False]),
    )
    if with_rules:
        params._rules = {
            timedelta(hours=1).total_seconds(): trial.suggest_int("max_trx_hour", 2, 20),
            timedelta(days=1).total_seconds(): trial.suggest_int("max_trx_day", 2, 50),
            timedelta(weeks=1).total_seconds(): trial.suggest_int("max_trx_week", 15, 300),
        }
    else:
        params._rules = {}
    return params


def experiment(params: ClassificationParameters):
    params.training_duration = PARAMS.clf_params.training_duration  # Ensure the training duration is set
    PARAMS.clf_params = params
    banksys = PARAMS.create_banksys(use_cache=True)
    f = banksys.simulate_until(banksys.attack_start + timedelta(days=50))
    features = pl.concat(f)
    predicted = banksys.clf.predict(features)
    df = banksys._transactions_df.filter(pl.col("timestamp").is_between(banksys.attack_start, banksys.current_time))
    assert features["amount"].equals(df["amount"])
    truth = df["is_fraud"].to_numpy().astype(np.bool)

    f1 = f1_score(truth, predicted)
    # recall = recall_score(test_y, predictions)
    # precision = precision_score(test_y, predictions)
    # confusion = confusion_matrix(test_y, predictions)
    return float(f1)


def experiment_with_rules(trial: optuna.Trial):
    params = get_params(with_rules=True, trial=trial)
    return experiment(params)


def experiment_without_rules(trial: optuna.Trial):
    params = get_params(with_rules=False, trial=trial)
    return experiment(params)


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
        study_name="with-rules",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment_with_rules, n_trials=100, n_jobs=10)

    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="without-rules",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment_without_rules, n_trials=100, n_jobs=20)
