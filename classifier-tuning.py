import logging
import optuna
import os
import dotenv
from sklearn.metrics import f1_score
from parameters import ClassificationParameters, Parameters, CardSimParameters
from banksys import Banksys, ClassificationSystem, TransactionsRegistry


def get_test_set():
    logging.info("Generating test set...")
    params = Parameters(
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters(),
    )
    cards, terminals, transactions = params.cardsim.get_simulation_data()
    banksys = Banksys(
        cards=cards,
        terminals=terminals,
        aggregation_windows=params.aggregation_windows,
        attackable_terminal_factor=params.terminal_fract,
        clf_params=params.clf_params,
    )
    logging.info("Fitting banksys")
    registry = TransactionsRegistry(transactions)
    train_x, train_y = banksys.fit(registry)
    test_x, test_y = banksys.generate_test_set(registry)
    logging.info("Test set generated")
    return banksys, train_x, train_y, test_x, test_y


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
        params.rules = {
            "max_trx_hour": trial.suggest_int("max_trx_hour", 2, 20),
            "max_trx_day": trial.suggest_int("max_trx_day", 2, 50),
            "max_trx_week": trial.suggest_int("max_trx_week", 15, 300),
        }
    else:
        params.rules = {}
    return params


def experiment(params: ClassificationParameters):
    clf = ClassificationSystem(banksys, params)
    clf.fit(train_x, train_y)
    # Predict & compute metrics
    predictions = clf.predict(test_x)
    # accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
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
    banksys, train_x, train_y, test_x, test_y = get_test_set()

    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="with-rules",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment_with_rules, n_trials=100, n_jobs=1)

    study = optuna.create_study(
        storage="sqlite:///classifier-tuning.db",
        study_name="without-rules",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment_without_rules, n_trials=100, n_jobs=1)
