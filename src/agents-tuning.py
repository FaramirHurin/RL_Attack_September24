import logging
import os
from typing import Literal

import dotenv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from tap import Tap

from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, EnvParameters, VAEParameters
from experiment import Experiment
from main import run_parallel


class Args(Tap):
    agent: Literal["vae", "ppo", "rppo"]
    "Algorithm to use for the agent"
    anomaly: bool = False
    "Whether to use anomaly detection"
    modification: bool = False
    n_episodes: int = 4000
    pool_size: int = 8
    n_runs: int = 8
    n_trials: int = 150


def experiment(trial: optuna.Trial, args: Args) -> float:
    match args.agent:
        case "rppo":
            agent = PPOParameters.suggest_rppo(trial)
        case "ppo":
            agent = PPOParameters.suggest_ppo(trial)
        case "vae":
            agent = VAEParameters.suggest(trial)
        case other:
            raise ValueError(f"Unknown agent type: {other}")
    params = Parameters(
        agent=agent,
        clf_params=ClassificationParameters.paper_params(args.anomaly, args.modification),
        cardsim=CardSimParameters.paper_params(with_modification=args.modification),
        env_params=EnvParameters(n_episodes=args.n_episodes),
    )
    exp = Experiment.create(params, f"logs/tuning/{args.agent}/trial-{trial.number}")
    runs = run_parallel(exp, n_jobs=args.pool_size, n_repetitions=args.n_runs)
    amounts = [r.total_amount for r in runs]
    trial.set_user_attr("amounts", amounts)
    trial.set_user_attr("#episodes", [r.n_episodes for r in runs])
    trial.set_user_attr("logdir", exp.logdir)
    total = sum(amounts)
    objective = total / args.n_runs
    logging.info(f"Trial {trial.number} avg objective: {objective}")
    return objective


def load_study(file: str, study_name: str):
    return optuna.create_study(
        storage=JournalStorage(JournalFileBackend(file_path=file)),
        study_name=study_name,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = Args().parse_args()
    study_name = f"{args.agent.upper()}-anomaly={args.anomaly}-modification={args.modification}-{args.n_episodes}"
    file_name = f"{args.agent}-tuning.journal"
    study = load_study(file_name, study_name)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = args.n_trials - n_complete
    while n_remaining > 0:
        logging.info(f"Running {study.study_name}: {n_remaining} trials remaining")
        study.optimize(lambda trial: experiment(trial, args), n_trials=1, n_jobs=1)
        study = load_study(file_name, study_name)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = args.n_trials - n_complete
    logging.info(f"Study {study.study_name} completed with {n_complete} trials")
