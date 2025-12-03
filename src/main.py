import logging
import os
from multiprocessing.pool import AsyncResult, Pool
from typing import Literal

import dotenv
from marlenv.utils import Schedule

import utils
from experiment import Experiment, Run
from parameters import CardSimParameters, ClassificationParameters, EnvParameters, Parameters, PPOParameters, VAEParameters
from runner import Runner


def run(p: Parameters, rundir: str):
    logging.info(f"Starting run with seed {p.seed}...")
    p.seed_random()
    try:
        runner = Runner(p, quiet=False)
        episodes = runner.run()
        return Run.create(rundir, p, episodes)
    except Exception as e:
        logging.error(f"Run with seed {p.seed}: Error occurred while running experiment: {e}", exc_info=True)


def run_parallel(exp: Experiment, n_jobs: int = 8, n_repetitions: int = 32):
    runs = list[Run]()
    with Pool(n_jobs) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p, rundir in exp.repeat(n_repetitions):
            logging.info(f"Submitting run with seed {p.seed}...")
            handles.append(pool.apply_async(run, (p, rundir)))
        for h in handles:
            r = h.get()
            if r is not None:
                runs.append(r)
                logging.info(f"Run with seed {r.params.seed} completed with result {r.total_amount:.2f}")
    return runs


def main(
    algorithm: Literal["vae", "ppo", "rppo"],
    anomaly: bool,
    n_repetitions: int = 1,
    ulb_data: bool = False,
    with_modification: bool = False,
    initial_seed: int = 0,
    n_jobs: int = 1,
):
    if algorithm == "vae":
        agent = VAEParameters.best_vae(anomaly)
    elif algorithm == "rppo":
        agent = PPOParameters.best_rppo(anomaly)
    elif algorithm == "ppo":
        agent = PPOParameters.best_ppo(anomaly)
        c2 = Schedule.constant(0.0)
        agent.entropy_c2 = c2
        agent.normalize_rewards = False
        agent.normalize_advantages = True
        # utils.init_tb_logger(f"runs/entropy-c2={c2.value}")
        utils.init_tb_logger()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(with_modification=with_modification, ulb_data=ulb_data),
        clf_params=ClassificationParameters(use_anomaly=anomaly, fp_rate=0.01, fn_rate=0.01),
        env_params=EnvParameters(pool_size=50, n_episodes=6000),
        seed=initial_seed,
        invalidate_banksys_cache=False,
    )
    exp = Experiment.create(params)
    if n_jobs == 1:
        return [run(p, rundir) for p, rundir in exp.repeat(n_repetitions)]
    return run_parallel(exp, n_jobs=n_jobs, n_repetitions=n_repetitions)


if __name__ == "__main__":
    # Le problème c'est que le score de risque augmente jusqu'à atteindre 1.0 dans les terminaux de paiement.
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "info").upper()  # info
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        main(algorithm="ppo", anomaly=True, initial_seed=7, with_modification=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
