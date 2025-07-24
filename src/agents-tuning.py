import logging
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from multiprocessing.pool import AsyncResult
from typing import Callable

import dotenv
import optuna
import torch

from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, VAEParameters
from plots import Experiment, Run
from runner import Runner

N_PARALLEL = 2
# TIMEOUT = timedelta(minutes=25)
CLF_PARAMS = ClassificationParameters.paper_params(True)
CARDSIM_PARAMS = CardSimParameters.paper_params()


def run(p: Parameters, trial_num: int):
    try:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            # We assign the run to the device based on its absolute run number, i.e.
            # trial.number * N_PARALLEL + seed_value.
            device_num = (trial_num * N_PARALLEL + p.seed_value) % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_num}")
        runner = Runner(p, quiet=True, device=device)
        episodes = runner.run()
        return Run.create(p, episodes)
    except Exception as e:
        logging.error(f"Trial {trial_num}: Error occurred while running experiment with seed {p.seed_value}: {e}", exc_info=True)


def experiment(trial: optuna.Trial, fn: Callable[[optuna.Trial], PPOParameters | VAEParameters]) -> float:
    params = Parameters(
        agent=fn(trial),
        clf_params=CLF_PARAMS,
        cardsim=CARDSIM_PARAMS,
        seed_value=0,
        include_weekday=True,
        save=True,
        n_episodes=1000,
    )
    exp = Experiment.create(params)

    pool = mp.Pool(N_PARALLEL)
    run_args = [(p, trial.number) for p in exp.repeat(N_PARALLEL)]
    runs = pool.starmap(run, run_args)
    total = 0
    for r in runs:
        if r is not None:
            total += r.total_amount
            logging.info(f"Trial {trial.number} run completed with result {r.total_amount:.2f}")
        else:
            logging.error(f"Trial {trial.number} run failed.")

    objective = total / len(runs)
    logging.critical(f"Trial {trial.number} objective: {objective}")
    return objective


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    p = Parameters(
        clf_params=CLF_PARAMS,
        cardsim=CARDSIM_PARAMS,
        save=False,
    )
    if not p.banksys_is_in_cache():
        logging.info("Creating banksys...")
        b = p.create_banksys()
        b.save(p.banksys_dir)

    # try:
    #     study = optuna.create_study(
    #         storage="sqlite:///agents-tuning.db",
    #         study_name="ppo",
    #         direction=optuna.study.StudyDirection.MAXIMIZE,
    #         load_if_exists=True,
    #     )
    #     study.optimize(lambda t: experiment(t, PPOParameters.suggest_ppo), n_trials=100, n_jobs=6)
    # except Exception as e:
    #     logging.error(f"Error during PPO study optimization: {e}", exc_info=True)

    # try:
    #     study = optuna.create_study(
    #         storage="sqlite:///agents-tuning.db",
    #         study_name="rppo",
    #         direction=optuna.study.StudyDirection.MAXIMIZE,
    #         load_if_exists=True,
    #     )
    #     study.optimize(lambda t: experiment(t, PPOParameters.suggest_rppo), n_trials=40, n_jobs=6)
    # except Exception as e:
    #     logging.error(f"Error during RPPO study optimization: {e}", exc_info=True)

    try:
        study = optuna.create_study(
            storage="sqlite:///agents-tuning.db",
            study_name="vae",
            direction=optuna.study.StudyDirection.MAXIMIZE,
            load_if_exists=True,
        )
        study.optimize(lambda t: experiment(t, VAEParameters.suggest), n_trials=100, n_jobs=2)
    except Exception as e:
        logging.error(f"Error during VAE study optimization: {e}", exc_info=True)

    logging.info("All trials completed.")
