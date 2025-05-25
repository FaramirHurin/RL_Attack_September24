import multiprocessing as mp
import dotenv
import optuna
import os
import pandas as pd

from marlenv.utils.schedule import Schedule
from datetime import timedelta

from runner import PoolRunner, save_episodes
import logging
from parameters import CardSimParameters, Parameters, PPOParameters, ClassificationParameters


CLF_PARAMS = ClassificationParameters(
    use_anomaly=False,
    n_trees=200,
    balance_factor=0.05,
    contamination=0.001,
    training_duration=timedelta(days=150),
    quantiles_features=["amount"],
    quantiles_values=[0.05, 0.95],
    rules={"max_trx_hour": 5, "max_trx_week": 20, "max_trx_day": 10},
)

CARDSIM_PARAMS = CardSimParameters(
    n_days=365 * 2 + 150 + 30,  # 2 years budget + 150 days training + 30 days warmup
    n_payers=20_000,
)


def recurrent_trial(trial: optuna.Trial):
    c1 = Schedule.linear(
        trial.suggest_float("critic_c1_start", 0.1, 0.7),
        trial.suggest_float("critic_c1_end", 0.001, 0.1),
        trial.suggest_int("critic_c1_steps", 1000, 4000),
    )

    c2 = Schedule.linear(
        trial.suggest_float("entropy_c2_start", 0.001, 0.1),
        trial.suggest_float("entropy_c2_end", 0.001, 0.05),
        trial.suggest_int("entropy_c2_steps", 1000, 4000),
    )
    train_interval = trial.suggest_int("train_interval", 8, 256)
    minibatch_size = trial.suggest_int("minibatch_size", 4, train_interval)

    lr_actor = trial.suggest_float("lr_actor", 0.0001, 0.001)
    lr_critic = trial.suggest_float("lr_critic", 0.0001, 0.001)
    enable_clipping = trial.suggest_categorical("enable_clipping", [True, False])
    if enable_clipping:
        grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 1, 10)
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
        n_episodes=4000,
        card_pool_size=50,
        clf_params=CLF_PARAMS,
        cardsim=CARDSIM_PARAMS,
        logdir=f"tuning/r-ppo/trial-{trial.number}",
        seed_value=0,
        save=True,
    )
    with mp.Pool(8) as pool:
        amounts = pool.map(experiment, params.repeat(8))
    objective = sum(amounts) / len(amounts)
    logging.critical(f"Trial {trial.number} objective: {objective}")
    return objective


def experiment(params: Parameters):
    try:
        params.save()
        runner = PoolRunner(params)
        episodes = runner.run()
        total_amount = sum(e.score[0] for e in episodes)
        save_episodes(episodes, params.logdir)
        del runner
        del episodes
        return total_amount
    except Exception as e:
        logging.error(f"Error occurred while running experiment: {e}")
    return 0.0


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(filename="logs.txt", filemode="a", level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    p = Parameters(PPOParameters(), clf_params=CLF_PARAMS, cardsim=CARDSIM_PARAMS)
    p.create_pooled_env()

    study = optuna.create_study(
        storage="sqlite:///r-ppo_tuning.db",
        study_name="r-ppo-maximize-4000",
        direction=optuna.study.StudyDirection.MAXIMIZE,
        load_if_exists=False,
    )
    study.optimize(recurrent_trial, n_trials=150, n_jobs=3)
    df: pd.DataFrame = study.trials_dataframe()
    # Save the DataFrame to a CSV file
    df.to_csv("r-ppo_tuning_results.csv", index=False)
    logging.critical(f"Trials DataFrame:\n{df}")
    print(df)

    print(study.best_params)
