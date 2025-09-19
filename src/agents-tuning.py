import logging
import os
from multiprocessing.pool import Pool, AsyncResult

import dotenv
import optuna
import torch

from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, VAEParameters
from plots import Experiment, Run
from runner import Runner

N_JOBS = 5
POOL_SIZE = 4
N_RUNS = 4
USE_ANOMALY = True


def run(p: Parameters, trial_num: int):
    logging.info(f"Starting trial {trial_num} with seed {p.seed_value}...")
    try:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            # We assign the run to the device based on its absolute run number, i.e.
            # trial.number * N_PARALLEL + seed_value.
            device_num = (trial_num + p.seed_value) % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_num}")
        runner = Runner(p, quiet=True, device=device)
        logging.info(f"Running trial {trial_num} with seed {p.seed_value}...")
        episodes = runner.run()
        return Run.create(p, episodes)
    except Exception as e:
        logging.error(f"Trial {trial_num}: Error occurred while running experiment with seed {p.seed_value}: {e}", exc_info=True)


def experiment(trial: optuna.Trial) -> float:
    params = Parameters(
        agent=PPOParameters.suggest_rppo(trial),
        clf_params=ClassificationParameters.paper_params(USE_ANOMALY),
        cardsim=CardSimParameters.paper_params(),
        save=False,
        n_episodes=4000,
        seed_value=0,
    )
    exp = Experiment.create(params)
    total = 0.0
    with Pool(POOL_SIZE) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p in exp.repeat(N_RUNS):
            logging.info(f"Submitting trial {trial.number} run with seed {p.seed_value}...")
            handles.append(pool.apply_async(run, (p, trial.number)))
        for h in handles:
            r = h.get()
            if r is None:
                logging.error(f"Trial {trial.number} run failed.")
            else:
                total += r.total_amount
                logging.info(f"Trial {trial.number} run completed with result {r.total_amount:.2f}")
    objective = total / N_RUNS
    logging.info(f"Trial {trial.number} avg objective: {objective}")
    return objective


def main():
    global USE_ANOMALY
    p = Parameters(
        clf_params=ClassificationParameters.paper_params(USE_ANOMALY),
        cardsim=CardSimParameters.paper_params(),
        save=False,
    )
    if not p.banksys_is_in_cache():
        logging.info("Creating banksys...")
        b = p.create_banksys()
        b.save(p.banksys_dir)

    USE_ANOMALY = False
    study = optuna.create_study(
        storage="sqlite:///agents-tuning.db",
        study_name=f"rppo-use-anomaly={USE_ANOMALY}",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(experiment, n_trials=80, n_jobs=N_JOBS)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
