import logging
import os

import dotenv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, EnvParameters, VAEParameters
from experiment import Experiment
from main import run_parallel

POOL_SIZE = 8
N_RUNS = 8
N_TRIALS = 150


def experiment(trial: optuna.Trial) -> float:
    match AGENT:
        case "rppo":
            agent = PPOParameters.suggest_rppo(trial)
        case "ppo":
            agent = PPOParameters.suggest_ppo(trial)
        case "vae":
            agent = VAEParameters.suggest(trial)
        case other:
            raise ValueError(f"Unknown agent type: {other}")
    n_episodes = 4000
    params = Parameters(
        agent=agent,
        clf_params=ClassificationParameters.paper_params(USE_ANOMALY, WITH_MODIFICATION),
        cardsim=CardSimParameters.paper_params(with_modification=WITH_MODIFICATION),
        env_params=EnvParameters(n_episodes=n_episodes),
    )
    exp = Experiment.create(params, f"logs/tuning/{AGENT}/trial-{trial.number}")
    runs = run_parallel(exp, n_jobs=POOL_SIZE, n_repetitions=N_RUNS)
    amounts = [r.total_amount for r in runs]
    trial.set_user_attr("amounts", amounts)
    trial.set_user_attr("#episodes", [r.n_episodes for r in runs])
    trial.set_user_attr("logdir", exp.logdir)
    total = sum(amounts)
    objective = total / N_RUNS
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
    AGENT = "ppo"
    USE_ANOMALY = True
    WITH_MODIFICATION = False
    study_name = f"{AGENT.upper()}-anomaly={USE_ANOMALY}-modification={WITH_MODIFICATION}"
    file_name = f"{AGENT}-tuning.journal"
    study = load_study(file_name, study_name)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = N_TRIALS - n_complete
    while n_remaining > 0:
        logging.info(f"Running {study.study_name}: {n_remaining} trials remaining")
        study.optimize(experiment, n_trials=1, n_jobs=1)
        study = load_study(file_name, study_name)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = N_TRIALS - n_complete
    logging.info(f"Study {study.study_name} completed with {n_complete} trials")
