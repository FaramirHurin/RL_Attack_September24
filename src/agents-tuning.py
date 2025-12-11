import logging
import os

import dotenv
import optuna

from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, EnvParameters
from experiment import Experiment
from main import run_parallel

POOL_SIZE = 2
N_RUNS = 8
USE_ANOMALY = False
WITH_MODIFICATION = False


def experiment(trial: optuna.Trial) -> float:
    params = Parameters(
        agent=PPOParameters.suggest_rppo(trial),
        clf_params=ClassificationParameters.paper_params(USE_ANOMALY, WITH_MODIFICATION),
        cardsim=CardSimParameters.paper_params(with_modification=WITH_MODIFICATION),
        env_params=EnvParameters(),
    )
    exp = Experiment.create(params)
    runs = run_parallel(exp, n_jobs=POOL_SIZE, n_repetitions=N_RUNS)
    total = sum(r.total_amount for r in runs)
    objective = total / N_RUNS
    logging.info(f"Trial {trial.number} avg objective: {objective}")
    return objective


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    study = optuna.create_study(
        storage="sqlite:///agents-tuning.db",
        study_name=f"rppo-anomaly={USE_ANOMALY}-with_modification={WITH_MODIFICATION}",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment, n_trials=80, n_jobs=1)
