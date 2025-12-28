import logging
import os
from multiprocessing.pool import AsyncResult, Pool
from typing import Literal
from datetime import timedelta

import dotenv
from tap import Tap

from experiment import Experiment, Run
from parameters import (
    CardSimParameters,
    ClassificationParameters,
    EnvParameters,
    Parameters,
    PPOParameters,
    VAEParameters,
)
from runner import Runner


class Arguments(Tap):
    agent: Literal["vae", "ppo", "rppo"] = "ppo"
    "Algorithm to use for the agent"
    anomaly: bool = False
    "Whether to use anomaly detection"
    n_repetitions: int = 1
    "Number of repetitions for the experiment"
    with_modification: bool = False
    "Whether to use modification in the environment"
    initial_seed: int = 0
    "Initial random seed"
    n_jobs: int = 1
    "Number of parallel jobs to run"
    ulb_data: bool = False
    "Whether to use ULB data"
    logdir: str | None = None
    """Directory to store the logs of the experiment"""
    retrain_interval_days: int | None = None
    "Interval in days to retrain the classifier"


def run(p: Parameters, rundir: str, quiet: bool = False) -> Run | None:
    # utils.init_tb_logger(os.path.join("runs", f"{p.agent_name}-{datetime.now().isoformat().replace(':', '-')}"))
    logging.info(f"Starting run with seed {p.seed}...")
    p.seed_random()
    try:
        runner = Runner(p, quiet=quiet)
        episodes = runner.run()
        return Run.create(rundir, p, episodes)
    except Exception as e:
        logging.error(
            f"Run with seed {p.seed}: Error occurred while running experiment: {e}",
            exc_info=True,
        )


def run_parallel(exp: Experiment, n_jobs: int = 8, n_repetitions: int = 32):
    runs = list[Run]()
    with Pool(n_jobs) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p, rundir in exp.repeat(n_repetitions):
            logging.info(f"Submitting run with seed {p.seed}...")
            handles.append(pool.apply_async(run, (p, rundir, True)))
        for h in handles:
            r = h.get()
            if r is not None:
                runs.append(r)
                logging.info(f"Run with seed {r.params.seed} completed with result {r.total_amount:.2f}")
    return runs


def main(args: Arguments):
    if args.agent == "vae":
        agent = VAEParameters.best_vae(args.anomaly, args.with_modification)
    elif args.agent == "rppo":
        agent = PPOParameters.best_rppo(args.anomaly, args.with_modification)
    elif args.agent == "ppo":
        agent = PPOParameters.best_ppo(args.anomaly, args.with_modification)
    else:
        raise ValueError(f"Unknown algorithm: {args.agent}")
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(with_modification=args.with_modification, ulb_data=args.ulb_data),
        clf_params=ClassificationParameters.paper_params(with_anomaly=args.anomaly, with_modification=args.with_modification),
        env_params=EnvParameters(pool_size=50, n_episodes=6000),
        seed=args.initial_seed,
    )
    if args.retrain_interval_days is not None:
        params.clf_params.retrain_interval = timedelta(days=args.retrain_interval_days)
    exp = Experiment.create(params, logdir=args.logdir)
    if args.n_jobs == 1:
        return [run(p, rundir) for p, rundir in exp.repeat(args.n_repetitions)]
    return run_parallel(exp, n_jobs=args.n_jobs, n_repetitions=args.n_repetitions)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        # args = Arguments().parse_args()
        for algorithm in ("ppo", "rppo", "vae"):
            for anomaly in (False, True):
                for modification in (False, True):
                    args = Arguments().parse_args()
                    args.agent = algorithm
                    args.anomaly = anomaly
                    args.with_modification = modification
                    args.n_repetitions = 30
                    args.initial_seed = 100  # Seed different from the tuning
                    args.n_jobs = 1
                    args.ulb_data = False
                    args.logdir = os.path.join("logs", f"{algorithm}-retrained")
                    if anomaly:
                        args.logdir += "-with-anomaly"
                    else:
                        args.logdir += "-no-anomaly"
                    if modification:
                        args.logdir += "-with-modification"
                    else:
                        args.logdir += "-no-modification"
                    if os.path.exists(args.logdir):
                        logging.info(f"Logdir {args.logdir} already exists. Skipping...")
                        continue
                    logging.info(f"Starting experiment with arguments: {args}")
                    main(args)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
