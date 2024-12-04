import pandas as pd
from typing import Literal
import torch
from sklearn.neural_network import MLPClassifier
import rl
from nn_exception import NNException
import warnings
import torch.multiprocessing as mp
import numpy as np
import polars as pl
import typed_argparse as tap
from sklearn.ensemble import RandomForestClassifier
import os

from Classes.column_preprocess import get_column_combinations
from Classes.datset_inteface import Dataset
from rl.agents import PPO
from Classes.baselines_classes import BaselineAgent
from Classes.fraud_env import FraudEnv
from Classes.datset_inteface import DatasetLoader


warnings.filterwarnings("ignore", message="X does not have valid feature names")
pd.set_option("display.max_columns", None)


class Args(tap.TypedArgs):
    dataset_type: str = tap.arg(help="Dataset type")
    logdir: str = tap.arg(help="Directory to save logs")
    k: int = tap.arg(help="Number of known features")
    u: int = tap.arg(help="Number of unknown features")
    c: int = tap.arg(help="Number of controllable features")
    n_steps: int = tap.arg(help="Number of steps to train the agent")
    challenge: Literal["fraudulent", "genuine", "all"] = tap.arg(help="Only use frauds in the environment observations")
    reward_type: Literal["probability", "label"] = tap.arg(help="Type of reward to use")
    run_num: int = tap.arg(help="Run number")
    min_values: np.ndarray = tap.arg(help="Min values features can have")
    max_values: np.ndarray = tap.arg(help="Max values features can have")

    @property
    def directory(self):
        # Tricky, it depends on the database type too
        if self.dataset_type == "Generator":
            u_to_print = self.print_generator_uk(self.u)
            k_to_print = self.print_generator_uk(self.k)
        else:
            u_to_print = len(self.u)
            k_to_print = len(self.k)
        to_return = os.path.join(
            self.logdir, f"k-{k_to_print}", f"u-{u_to_print}", f"reward-{self.reward_type}"
        )  # self.dataset, self.challenge,
        return to_return

    def print_generator_uk(self, columns):
        c = False
        t = False
        for column in columns:
            if column.startswith("CUSTOMER"):
                c = True
            if column.startswith("TERMINAL"):
                t = True
        if c and t:
            return "Cust_Term"
        elif not c and t:
            return "Term"
        elif c and not t:
            return "Cust"
        else:
            return 0


def test_agent(
    env: FraudEnv,
    clf: RandomForestClassifier | MLPClassifier,
    agent: BaselineAgent,
    n_steps: int,
    reward_type: Literal["probability", "label"],
):
    actions = pl.DataFrame(agent.select_actions_batch(n_steps), env.actions)
    transactions = env.get_batch(n_steps)
    modified_transactions = transactions.with_columns([actions[col] for col in env.actions])
    if reward_type == "probability":
        rewards = clf.predict_proba(modified_transactions.to_numpy())[:, 0]
    elif reward_type == "label":
        preds = clf.predict(modified_transactions.to_numpy())
        rewards = np.ones(preds.shape) - preds
    else:
        print("Not the right type of reward")
        raise Exception
    return rewards


def experiment(args: Args, dataset: Dataset, clf: RandomForestClassifier | MLPClassifier, gpu_index: int):
    env = FraudEnv(
        transactions=dataset.env_transactions(args.challenge),
        k=args.k,
        u=args.u,
        c=args.c,
        classifier=clf,
        reward_type=args.reward_type,
        min_values=args.min_values,
        max_values=args.max_values,
    )
    ppo = PPO(
        env.observation_shape[0],
        env.n_actions,
        lr_actor=6e-4,  # 5e-4
        gamma=0.9,
        lr_critic=6e-4,
        k_epochs=20,  # 20
        eps_clip=0.15,  # 15
        gpu_index=gpu_index,
    )
    logs = dict[str, np.ndarray]()
    try:
        logs["PPO"] = rl.train.train_agent(ppo, env, args.n_steps)
    except NNException as e:
        print(f"Exception: {e}")
        print(f"Skipping: {args}")
        os.makedirs(args.directory, exist_ok=True)
        # save the weights
        torch.save(e.nn.state_dict(), os.path.join(args.directory, "weights.pth"))
        with open(os.path.join(args.directory, "error.txt"), "w") as f:
            f.write(str(e))
            f.write(str(args))
        return

    mimicry_transactions = "genuine"
    for sampling in ("multivariate", "univariate", "uniform", "mixture"):
        for dataset_size in ("1k", "100%"):  # "5%",
            train_X = dataset.mimicry_TR(mimicry_transactions, dataset_size, args.c)  # Controllable features
            agent = BaselineAgent(train_X=train_X, generation_method=sampling)
            key = f"{sampling}-{mimicry_transactions}-{dataset_size}"
            logs[key] = test_agent(env, clf, agent, args.n_steps, args.reward_type)
            del agent

    os.makedirs(str(args.directory), exist_ok=True)
    filename = str(args.directory) + "/file.csv"
    pd.DataFrame(logs).to_csv(filename, index=False)

    print("Controllable are" + str(args.c) + " Fixed are" + str(args.k) + " Unknown are" + str(args.u))
    # print(args.directory)
    print(pd.DataFrame(logs).describe())

    # return filename


def run_all_experiments(
    date_time,
    dataset_types,
    n_features_list,
    clusters_list,
    class_sep_list,
    balance_list,
    classifier_names,
    min_max_quantile,
    N_REPETITIONS,
    N_STEPS,
    PROCESS_PER_GPU,
    N_GPUS,
    multithread=False,
):
    # Parallelize task execution after directory creation
    for experiment_number in range(N_REPETITIONS):
        LOGDIR_EXPERIMENT = os.path.join("logs", date_time, str(experiment_number))

        for classifier_name in classifier_names:
            LOGDIR_CLASSIFIER = os.path.join(LOGDIR_EXPERIMENT, classifier_name)

            for dataset_type in dataset_types:
                LOGS_DATASET = os.path.join(LOGDIR_CLASSIFIER, dataset_type)

                dataset_loader = DatasetLoader(
                    dataset_type=dataset_type,
                    classifier=classifier_name,
                    n_features_list=n_features_list,
                    clusters_list=clusters_list,
                    class_sep_list=class_sep_list,
                    balance_list=balance_list,
                )
                datasets, classifiers = dataset_loader.load()

                tasks = []
                for key in datasets.keys():
                    dataset = datasets[key]
                    classifier = classifiers[key]
                    match dataset_type:
                        case "SkLearn":
                            folder_name = f"n_features={key[2]}_n_clusters={key[3]}_class_sep={key[4]}_balance={key[5]}"
                        case "Kaggle":
                            folder_name = f"balance={key[2]}"
                        case "Generator":
                            folder_name = f"balance={key[2]}"
                        case _:
                            raise Exception("Not a valid dataset type")
                    save_path = os.path.join(LOGS_DATASET, folder_name)
                    os.makedirs(save_path, exist_ok=True)

                    save_path = os.path.join(LOGS_DATASET, folder_name)
                    df_negative = dataset.env_transactions("genuine")
                    min_values = df_negative.quantile(min_max_quantile)
                    max_values = df_negative.quantile(1 - min_max_quantile)
                    if "Unnamed_0" in df_negative.columns:
                        df_negative = df_negative.drop("Unnamed_0")
                    columns_combination = get_column_combinations(dataset_type=dataset_type, df=df_negative)

                    for index in range(len(columns_combination)):
                        K_COLUMNS = columns_combination[index]["K_columns"]
                        U_COLUMNS = columns_combination[index]["U_columns"]
                        C_COLUMNS = columns_combination[index]["C_columns"]
                        args = Args(
                            dataset_type=dataset_type,
                            logdir=save_path,
                            k=K_COLUMNS,
                            c=C_COLUMNS,
                            u=U_COLUMNS,
                            n_steps=N_STEPS,
                            challenge="fraudulent",
                            reward_type="label",
                            run_num=experiment_number,
                            min_values=min_values,
                            max_values=max_values,
                        )
                        if multithread:
                            tasks.append((args, dataset, classifier, index % N_GPUS))
                        else:
                            experiment(args, dataset, classifier, 0)

                if multithread:
                    n_processes = N_GPUS * PROCESS_PER_GPU
                    with mp.Pool(n_processes) as pool:
                        results = [pool.apply_async(experiment, task) for task in tasks]
                        for result in results:
                            result.get()


"""
UNDERSAMPLE = util.is_debugging()
dataset_types = ['Generator']  # Kaggle  Generator SkLearn
n_features_list = [16, 32, 64]
clusters_list = [1, 8, 16]
class_sep_list = [0.5, 1, 2, 8]
balance_list = [0.1, 0.5]
classifier_names = ['RF', 'DNN']
min_max_quantile = 0.05
N_REPETITIONS = 2 # 20
N_STEPS = 2000  # 0
PROCESS_PER_GPU = 2
N_GPUS = 1  # 8
run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_names, min_max_quantile, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS)
"""


"""
import logging
import time
from multiprocessing.pool import AsyncResult


# Join here to avoid training all the classifiers while the experiments are running
#join(LOGDIR, handles, experiment_number)

def join(logdir: str, handles: list[tuple[Args, datetime, AsyncResult]], experiment_number):
    experiments_logs = []
    while len(handles) > 0:
        i = 0
        while i < len(handles):
            print('We arrived at handles ' + str(i))
            args, start, path = handles[i]
            logging.info(f"Run {args.run_num} finished, {len(handles)} experiments remaining")
            experiments_logs.append({**args.__dict__, "start": start, "path": path}) # "end": end,
            # pl.DataFrame(experiments_logs).write_csv(f"{logdir}//experiments" + str(experiment_number) + ".csv")
            del handles[i]
            i += 1
        time.sleep(1)
"""
# bandit = QuantileForest_Bandit(action_min, action_max)
# logs["Quantile Forest"] = rl.train.train_bandit(bandit, env, args.n_steps)
