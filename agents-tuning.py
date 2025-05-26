import multiprocessing as mp
from multiprocessing.pool import AsyncResult
import dotenv
import optuna
import os
from datetime import datetime, timedelta
from typing import Callable
import pandas as pd

from marlenv.utils.schedule import Schedule

from runner import Runner
import logging
from parameters import CardSimParameters, Parameters, PPOParameters, ClassificationParameters, VAEParameters


N_REPEATS = 8
N_PARALLEL = 8
TIMEOUT = timedelta(minutes=20)


def ppo_parameters(trial: optuna.Trial):
    train_on = trial.suggest_categorical("train_on", ["transition", "episode"])

    c1 = Schedule.linear(
        trial.suggest_float("critic_c1_start", 0.1, 1.0),
        trial.suggest_float("critic_c1_end", 0.001, 0.5),
        trial.suggest_int("critic_c1_steps", 1000, 4000),
    )
    c2 = Schedule.linear(
        trial.suggest_float("entropy_c2_start", 0.001, 0.2),
        trial.suggest_float("entropy_c2_end", 0.0001, 0.1),
        trial.suggest_int("entropy_c2_steps", 1000, 4000),
    )
    train_interval = trial.suggest_int("train_interval", 4, 64)
    minibatch_size = trial.suggest_int("minibatch_size", 2, train_interval)
    lr_actor = trial.suggest_float("lr_actor", 0.0001, 0.01)
    lr_critic = trial.suggest_float("lr_critic", 0.0001, 0.01)
    enable_clipping = trial.suggest_categorical("enable_clipping", [True, False])
    if enable_clipping:
        grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 10)
    else:
        grad_norm_clipping = None

    params = Parameters(
        PPOParameters(
            is_recurrent=False,
            train_on=train_on,  # type: ignore[arg-type]
            critic_c1=c1,
            entropy_c2=c2,
            n_epochs=trial.suggest_int("n_epochs", 10, 100),
            minibatch_size=minibatch_size,
            train_interval=train_interval,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            grad_norm_clipping=grad_norm_clipping,
        ),
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters.paper_params(),
        logdir=f"tuning/r-ppo/trial-{trial.number}",
    )
    return experiment(trial, params)


def rppo_parameters(trial: optuna.Trial):
    c1 = Schedule.linear(
        trial.suggest_float("critic_c1_start", 0.1, 1.0),
        trial.suggest_float("critic_c1_end", 0.001, 0.5),
        trial.suggest_int("critic_c1_steps", 1000, 4000),
    )
    c2 = Schedule.linear(
        trial.suggest_float("entropy_c2_start", 0.001, 0.2),
        trial.suggest_float("entropy_c2_end", 0.0001, 0.1),
        trial.suggest_int("entropy_c2_steps", 1000, 4000),
    )
    train_interval = trial.suggest_int("train_interval", 4, 64)
    minibatch_size = trial.suggest_int("minibatch_size", 2, train_interval)
    lr_actor = trial.suggest_float("lr_actor", 0.0001, 0.01)
    lr_critic = trial.suggest_float("lr_critic", 0.0001, 0.01)
    enable_clipping = trial.suggest_categorical("enable_clipping", [True, False])
    if enable_clipping:
        grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 10)
    else:
        grad_norm_clipping = None

    params = Parameters(
        PPOParameters(
            is_recurrent=True,
            train_on="episode",
            critic_c1=c1,
            entropy_c2=c2,
            n_epochs=trial.suggest_int("n_epochs", 10, 100),
            minibatch_size=minibatch_size,
            train_interval=train_interval,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            grad_norm_clipping=grad_norm_clipping,
        ),
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters.paper_params(),
        logdir=f"tuning/r-ppo/trial-{trial.number}",
    )
    return experiment(trial, params)


def vae_parameters(trial: optuna.Trial):
    latent_dim = trial.suggest_int("latent_dim", 2, 64)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    lr = trial.suggest_float("lr", 0.0001, 0.001)
    trees = trial.suggest_int("trees", 20, 200)
    batch_size = trial.suggest_int("batch_size", 8, 64)
    quantile = trial.suggest_float("quantile", 0.9, 0.999)
    num_epochs = trial.suggest_int("num_epochs", 2_000, 10_000)

    params = Parameters(
        VAEParameters(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            trees=trees,
            batch_size=batch_size,
            num_epochs=num_epochs,
            quantile=quantile,
        ),
        clf_params=ClassificationParameters.paper_params(),
        cardsim=CardSimParameters.paper_params(),
        logdir=f"tuning/vae/trial-{trial.number}",
    )
    return experiment(trial, params)


def run(p: Parameters):
    try:
        runner = Runner(p)
        run = runner.run()
        return run.total_amount
    except Exception as e:
        logging.error(f"Error occurred while running experiment: {e}")
    return 0.0


def experiment(trial: optuna.Trial, params: Parameters):
    submission_times = list[datetime]()
    handles = list[AsyncResult]()
    with mp.Pool(N_PARALLEL) as pool:
        for p in params.repeat(N_REPEATS):
            handles.append(pool.apply_async(run, (p,)))
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
        logging.critical(f"Trial {trial.number} failed to complete any runs.")
        return 0.0
    objective = sum(amounts) / len(amounts)
    logging.critical(f"Trial {trial.number} objective: {objective}")
    return objective


def make_tuning(n_trials: int, name: str, fn: Callable[[optuna.Trial], float], n_jobs: int):
    study = optuna.create_study(
        storage="sqlite:///agent-tuning.db",
        study_name=name,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=True,
    )
    study.optimize(fn, n_trials=n_trials, n_jobs=n_jobs)
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

    make_tuning(150, "ppo", ppo_parameters, 4)
    make_tuning(100, "rppo", rppo_parameters, 4)
    make_tuning(100, "vae", vae_parameters, 4)
