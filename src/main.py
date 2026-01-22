import logging
import os
import time
from datetime import timedelta, datetime
from multiprocessing.pool import AsyncResult, Pool
from typing import Literal

import dotenv
from tap import Tap

from experiment import Experiment, Run
from parameters import (
    CardSimParameters,
    ClassificationParameters,
    EnvParameters,
    Parameters,
    PPOParameters,
    RandomParameters,
    VAEParameters,
)
from runner import Runner


class Arguments(Tap):
    agent: Literal["vae", "ppo", "rppo", "random"] = "ppo"
    "Algorithm to use for the agent"
    anomaly: bool = False
    "Whether to use anomaly detection"
    n_repetitions: int = 1
    "Number of repetitions for the experiment"
    modification: bool = False
    "Whether to use modification in the environment"
    initial_seed: int = 100
    "Initial random seed"
    n_jobs: int = 1
    "Number of parallel jobs to run"
    ulb_data: bool = False
    "Whether to use ULB data"
    retrain_interval: int | None = None
    "Interval to retrain the classifier (in days)"
    only_clipped_surrogate: bool = False
    noise: bool = False
    "Whether to add noise to the CardSim data"
    know_client: bool = False
    override: bool = False
    "Whether to override existing logs"

    @property
    def logdir(self):
        logdir = os.path.join("logs", self.agent)
        if self.anomaly:
            logdir += "-with-anomaly"
        else:
            logdir += "-no-anomaly"
        if self.modification:
            logdir += "-with-modification"
        else:
            logdir += "-no-modification"
        if self.retrain_interval is not None:
            logdir += f"-retrain-{self.retrain_interval}d"
        if self.only_clipped_surrogate:
            logdir += "-only-clipped-surrogate"
        if self.know_client and self.agent != "random":
            # It has no effect on random agent
            logdir += "-known-client"
        return logdir


def run(p: Parameters, rundir: str, quiet: bool = False, override: bool = False) -> Run | None:
    # utils.init_tb_logger(os.path.join("runs", f"{p.agent_name}-{datetime.now().isoformat().replace(':', '-')}"))
    logging.info(f"Starting run with seed {p.seed}...")
    if not override:
        try:
            run = Run.load(rundir)
            if run.params == p:
                logging.info(f"Run with seed {p.seed} already exists at {rundir}, skipping...")
                return run
            logging.info(f"Run directory {rundir} exists but parameters differ, re-running...")
        except FileNotFoundError:
            pass
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


def run_parallel(exp: Experiment, n_jobs: int = 8, n_repetitions: int = 32, override: bool = False):
    runs = list[Run]()
    with Pool(n_jobs) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p, rundir in exp.repeat(n_repetitions):
            logging.info(f"Submitting run with seed {p.seed}...")
            handles.append(pool.apply_async(run, (p, rundir, True, override)))
        logging.info(f"Waiting for {len(handles)} runs to complete...")
        start = datetime.now()
        while len(handles) > 0:
            ready = [(i, h) for i, h in enumerate(handles) if h.ready()]
            for i, h in reversed(ready):
                r = h.get()
                if r is not None:
                    runs.append(r)
                    logging.info(f"Run with seed {r.params.seed} completed with result {r.total_amount:.2f}")
                handles.pop(i)
            if len(ready) > 0:
                n_finished = n_repetitions - len(handles)
                n_remaining = len(handles)
                avg_time = (datetime.now() - start) / n_finished
                remaining = n_remaining * avg_time
                logging.info(f"[{n_finished}/{n_repetitions}] runs complete -- ETA {remaining}")
            time.sleep(1)
    return runs


def main(args: Arguments):
    if args.agent == "vae":
        agent = VAEParameters.best_vae(
            args.anomaly,
            args.modification,
            args.know_client,
        )
    elif args.agent == "rppo":
        agent = PPOParameters.best_rppo(
            args.anomaly,
            args.modification,
            args.know_client,
            only_clipped_surrogate=args.only_clipped_surrogate,
        )
    elif args.agent == "ppo":
        agent = PPOParameters.best_ppo(
            args.anomaly,
            args.modification,
            args.know_client,
            args.only_clipped_surrogate,
        )
    elif args.agent == "random":
        agent = RandomParameters()  # Random agent does not need parameters
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    if args.retrain_interval is not None:
        retrain_interval = timedelta(days=args.retrain_interval)
    else:
        retrain_interval = None
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(with_modification=args.modification, ulb_data=args.ulb_data, with_noise=args.noise),
        clf_params=ClassificationParameters.paper_params(
            with_anomaly=args.anomaly,
            with_modification=args.modification,
            retrain_interval=retrain_interval,
        ),
        env_params=EnvParameters(customer_location_is_known=args.know_client),
        seed=args.initial_seed,
    )
    exp = Experiment.create(params, logdir=args.logdir)
    if args.n_jobs == 1:
        return [run(p, rundir, override=args.override) for p, rundir in exp.repeat(args.n_repetitions)]
    return run_parallel(exp, n_jobs=args.n_jobs, n_repetitions=args.n_repetitions, override=args.override)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        args = Arguments().parse_args()
        logging.info(f"Starting experiment with arguments: {args}")
        main(args)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
