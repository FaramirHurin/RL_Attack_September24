import multiprocessing as mp
from multiprocessing.pool import AsyncResult
import dotenv
import optuna
import os
from datetime import datetime, timedelta
from typing import Callable
import pandas as pd

from runner import Runner
import logging
from parameters import CardSimParameters, Parameters, PPOParameters, ClassificationParameters, VAEParameters


N_PARALLEL = 8
TIMEOUT = timedelta(minutes=20)


def run(p: Parameters, trial_num: int):
    try:
        runner = Runner(p, quiet=p.seed_value != 0)
        run = runner.run()
        return run.total_amount
    except Exception as e:
        logging.error(f"Trial {trial_num}: Error occurred while running experiment with seed {p.seed_value}: {e}")
    return 0.0


def experiment(trial: optuna.Trial, fn: Callable[[optuna.Trial], PPOParameters | VAEParameters]) -> float:
    params = Parameters(
        agent=fn(trial),
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters.paper_params(),
        seed_value=0,
    )
    submission_times = list[datetime]()
    handles = list[AsyncResult]()
    with mp.Pool(N_PARALLEL) as pool:
        for p in params.repeat(N_PARALLEL):
            handles.append(pool.apply_async(run, (p, trial.number)))
            submission_times.append(datetime.now())
        amounts = []
        for i, handle in enumerate(handles):
            remaining = TIMEOUT - (datetime.now() - submission_times[i])
            try:
                result = handle.get(timeout=remaining.total_seconds())
                logging.info(f"Trial {trial.number} run {i} completed with result {result:.2f}")
                amounts.append(result)
            except mp.TimeoutError:
                logging.error(f"Trial {trial.number} timed out after {TIMEOUT}.")

    if len(amounts) == 0:
        logging.critical(f"Trial {trial.number} failed to complete any run.")
        return 0.0
    objective = sum(amounts) / len(amounts)
    logging.critical(f"Trial {trial.number} objective: {objective}")
    return objective


def make_tuning(n_trials: int, name: str, fn: Callable[[optuna.Trial], PPOParameters | VAEParameters], n_jobs: int):
    study = optuna.create_study(
        storage="sqlite:///agent-tuning.db",
        study_name=name,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(lambda t: experiment(t, fn), n_trials=n_trials, n_jobs=n_jobs)
    logging.critical(f"Best trial: {study.best_trial.number} with value {study.best_value} and params {study.best_params}")
    df: pd.DataFrame = study.trials_dataframe()
    # Save the DataFrame to a CSV file
    df.to_csv(f"tuning-{name}-{datetime.now()}.csv", index=False)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(filename="logs.txt", filemode="a", level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    p = Parameters(
        PPOParameters(),
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters.paper_params(),
        save=False,
    )
    if not p.banksys_is_in_cache():
        p.create_banksys(save=True)

    make_tuning(150, "ppo", lambda t: PPOParameters.suggest(False, t), 4)
    make_tuning(100, "rppo", lambda t: PPOParameters.suggest(True, t), 4)
    make_tuning(100, "vae", VAEParameters.suggest, 4)
