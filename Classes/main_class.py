import os
import random
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import rl
from Classes.baselines_classes import BaselineAgent
from Classes.column_preprocess import get_column_combinations
from Classes.datset_inteface import Dataset, DatasetLoader
from Classes.fraud_env import FraudEnv
from nn_exception import NNException
from rl.agents import PPO

warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class ExperimentParameters:
    dataset_type: str
    logdir: str
    k: list
    u: list
    c: list
    n_steps: int
    challenge: Literal["fraudulent", "genuine", "all"]
    reward_type: Literal["probability", "label"]
    min_values: np.ndarray
    max_values: np.ndarray
    seed: int

    @property
    def directory(self):
        # Tricky, it depends on the database type too
        if self.dataset_type == "Generator":
            u_to_print = self.print_generator_uk(self.u)
            k_to_print = self.print_generator_uk(self.k)
        else:
            u_to_print = len(self.u)
            k_to_print = len(self.k)
        to_return = os.path.join(self.logdir, f"k-{k_to_print}", f"u-{u_to_print}", f"reward-{self.reward_type}")
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


def experiment(params: ExperimentParameters, dataset: Dataset, clf: RandomForestClassifier | MLPClassifier, device_name: str):
    torch.random.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    env = FraudEnv(
        transactions=dataset.env_transactions(params.challenge),
        k=params.k,
        u=params.u,
        c=params.c,
        classifier=clf,
        reward_type=params.reward_type,
        min_values=params.min_values,
        max_values=params.max_values,
    )
    ppo = PPO(
        env.observation_shape[0],
        env.n_actions,
        lr_actor=6e-4,  # 5e-4
        gamma=0.9,
        lr_critic=6e-4,
        k_epochs=20,  # 20
        eps_clip=0.15,  # 15
        device_name=device_name,
    )
    logs = dict[str, np.ndarray]()
    try:
        logs["PPO"] = rl.train.train_agent(ppo, env, params.n_steps)
    except NNException as e:
        print(f"Exception: {e}")
        print(f"Skipping: {params}")
        os.makedirs(params.directory, exist_ok=True)
        # save the weights
        torch.save(e.nn.state_dict(), os.path.join(params.directory, "weights.pth"))
        with open(os.path.join(params.directory, "error.txt"), "w") as f:
            f.write(str(e))
            f.write(str(params))
        return

    mimicry_transactions = "genuine"
    for sampling in ("multivariate", "univariate", "uniform", "mixture"):
        for dataset_size in ("1k", "100%"):  # "5%",
            train_X = dataset.mimicry_TR(mimicry_transactions, dataset_size, params.c)  # Controllable features
            agent = BaselineAgent(train_X=train_X, generation_method=sampling)
            key = f"{sampling}-{mimicry_transactions}-{dataset_size}"
            logs[key] = test_agent(env, clf, agent, params.n_steps, params.reward_type)
            del agent

    os.makedirs(str(params.directory), exist_ok=True)
    filename = str(params.directory) + "/file.csv"
    pd.DataFrame(logs).to_csv(filename, index=False)


def run_experiments(
    logdir: str,
    dataset_types: list,
    n_features_list: list[int],
    clusters_list: list,
    class_sep_list: list,
    balance_list: list,
    classifier_names,
    min_max_quantile,
    first_experiment_num: int,
    n_experiments: int,
    n_steps: int,
    device_name: str,
    use_sklearn_precombinations: bool = False,

):

    for experiment_number in range(first_experiment_num, first_experiment_num + n_experiments):
        print(f"Starting experiments number {experiment_number}")
        for classifier_name in classifier_names:
            for dataset_type in dataset_types:
                dataset_loader = DatasetLoader(
                    dataset_type=dataset_type,
                    classifier=classifier_name,
                    n_features_list=n_features_list,
                    clusters_list=clusters_list,
                    class_sep_list=class_sep_list,
                    balance_list=balance_list,
                )
                datasets, classifiers = dataset_loader.load(use_sklearn_precombinations=use_sklearn_precombinations)

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
                    logdir = os.path.join(logdir, str(experiment_number), classifier_name, dataset_type, folder_name)
                    os.makedirs(logdir, exist_ok=True)

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
                        params = ExperimentParameters(
                            dataset_type=dataset_type,
                            logdir=logdir,
                            k=K_COLUMNS,
                            c=C_COLUMNS,
                            u=U_COLUMNS,
                            n_steps=n_steps,
                            challenge="fraudulent",
                            reward_type="label",
                            min_values=min_values,
                            max_values=max_values,
                            seed=experiment_number,
                        )
                        experiment(params, dataset, classifier, device_name)
