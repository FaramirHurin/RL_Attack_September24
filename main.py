import logging
import os
import time
import pickle
import pandas as pd
from icecream import ic
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.pool import AsyncResult
from typing import Literal, ClassVar

import dotenv
import numpy as np
import polars as pl
import torch.multiprocessing as mp
import typed_argparse as tap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import rl.train
from Classes.baselines import BaselineAgent
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from Classes.fraud_env import FraudEnv
from datasets import Dataset
from rl.agents.ppo_modiified import PPO
from rl.agents.continuous_bandit import ContinuousBandit
import util

dotenv.load_dotenv()
logging.getLogger().setLevel(os.getenv("LOG_LEVEL", "ERROR"))
logging.basicConfig(level=logging.ERROR)  # Only log ERROR level messages and above

pd.options.display.max_columns = None


class NotImplementedError(Exception):
    pass


class Args(tap.TypedArgs):
    logdir: str = tap.arg(help="Directory to save logs")
    fraud_balance: Literal["very-imbalanced", "imbalanced", "balanced"] = tap.arg(help="Fraud balance in synthetic datasets")
    k: int = tap.arg(help="Number of known features")
    u: int = tap.arg(help="Number of unknown features")
    n_steps: int = tap.arg(help="Number of steps to train the agent")
    challenge: Literal["fraudulent", "genuine", "all"] = tap.arg(help="Only use frauds in the environment observations")
    reward_type: Literal["probability", "label"] = tap.arg(help="Type of reward to use")
    run_num: int = tap.arg(help="Run number")

    @property
    def directory(self):
        return os.path.join(self.logdir,  f"k-{self.k}", f"u-{self.u}", self.challenge, f"reward-{self.reward_type}") #self.dataset,

def test_agent(
    env: FraudEnv,
    clf: RandomForestClassifier | MLPClassifier,
    agent: BaselineAgent,
    n_steps: int,
    reward_type: Literal["probability", "label"],
):
    """
        It generates the frauds using the baselines.
        Since the baselines do not use the classifier's feedback, all transactions are generated toghether in a single batch of n_steps size

        :param env: Environment modelling the target classifier
        :param clf: Trained classifier
        :param agent: Baseline agent
        :param n_steps: Number of generated frauds, i.e., size of the generated transactions batch
        :param reward_type: Whether the reward is a probability or a class
        :return: Probability of each performed transaction  to be genuine according to the classifier
    dddd
    """
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


def experiment(args: Args, dataset: Dataset, clf: RandomForestClassifier | MLPClassifier):
    """
    Initialize the fraud detection engine and PPO agent. Runs all baselines and PPO and measures the relative performances.
    Store the performances as a csv file

    :param args: Settings of the frauds environment
    :param dataset:
    :param clf: Trained classifier representing the fraud detection engine
    :return:
    """

    env = FraudEnv(
        transactions=dataset.env_transactions(args.challenge),
        k=args.k,
        u=args.u,
        classifier=clf,
        reward_type=args.reward_type,
    )

    x_train = dataset.train_x
    percentiles = pd.DataFrame(np.percentile(x_train, [1, 99], axis=0), columns=x_train.columns)
    percentiles = percentiles.loc[:, env.actions].values
    action_min = percentiles[0]
    action_max = percentiles[1]
    print(percentiles)

    if env.n_actions == 0:
        DEBUG = 0
        logging.error('N_actions = 0')
    '''
    agent = PPO(
        env.observation_shape[0],
        env.n_actions,
        lr_actor=1e-5,
        action_min=action_min,
        action_max=action_max,
        lr_critic=1e-6,
        # gamma=0,  # 0.85
        k_epochs=1,
        eps_clip=0.1,
    )
    '''
    bandit = ContinuousBandit(action_min, action_max)


    logs = dict[str, np.ndarray]()
    logs["ppo"] = rl.train.train_bandit(bandit, env, args.n_steps) #TODO Change name
    mimicry_transactions = "genuine"
    for sampling in ("multivariate", "univariate", "uniform"):
        for dataset_size in ("1k", "5%", "100%"):
            if sampling == "gan" and dataset_size == "100%":
                continue
            agent = BaselineAgent(
                k=args.k, test_x=dataset.mimicry_TR(mimicry_transactions, dataset_size, args.u), generation_method=sampling
            )
            key = f"{sampling}-{mimicry_transactions}-{dataset_size}"
            logs[key] = test_agent(env, clf, agent, args.n_steps, args.reward_type)
            del agent

    os.makedirs(args.directory, exist_ok=True)
    filename = str(args.directory)+"/file.csv"
    print(pd.DataFrame(logs).describe())
    pd.DataFrame(logs).to_csv(filename)
    print('Ending the cycle, filename and end are ready')
    return filename #, end


def run_all_experiments():
    VARIABLES_STEPS = 2
    N_FEATURES = 32
    N_REPETITIONS = 1 #20
    N_STEPS = 1500 #0
    PROCESS_PER_GPU = 2
    N_GPUS = 1 #8
    LOGDIR_OUTER = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(LOGDIR_OUTER, exist_ok=False)  # ?
    n_processes = N_GPUS * PROCESS_PER_GPU
    # pool = mp.Pool(n_processes)

    for experiment_number in range(N_REPETITIONS):
        LOGDIR = os.path.join(LOGDIR_OUTER, "_" + str(experiment_number))
        os.makedirs(LOGDIR, exist_ok=False)  # ?d
        for fraud_balance in ( [  "balanced" ]) : #"very-imbalanced","imbalanced",
            dataset = get_dataset #TODO Implement and

            N_FEATURES = dataset.shape[1]

            train_x, train_y, test_x, test_y = None #TODO create TR, TS etc.
            print('SHAPES ARE:')
            print( train_x.shape, train_y.shape, test_x.shape, test_y.shape )
            clf = get_classifier #TODO

            # Store test set
            test_x.to_pandas().to_csv(save_path + "/TestX.csv")
            test_y.to_pandas().to_csv(save_path + "TestY.csv")

            handles = [] #?

            print("Experiment number is " + str(experiment_number))
            print('Save path is ' + str(save_path))

            #n_actions_values = tuple(sorted(set(range(N_FEATURES, 0, -VARIABLES_STEPS)).union({0})))
            n_actions_values = tuple(range(8, VARIABLES_STEPS -1, -VARIABLES_STEPS)) # -10

            for n_actions in n_actions_values:  # at least one action
                k_values = tuple(range(0, N_FEATURES - n_actions, VARIABLES_STEPS))
                print('K values are' + str(k_values))
                for k in k_values:
                    u = N_FEATURES - n_actions - k
                    print('N_actions, k, u' + str(n_actions) + ' ' + str(k) + ' '+ str( u))

                    for reward_type in ("probability", "label"):
                        args = Args(
                            logdir=save_path,
                            fraud_balance=fraud_balance,
                            k=k,
                            #dataset=dataset,
                            u=u,
                            n_steps=N_STEPS,
                            challenge="fraudulent",
                            reward_type=reward_type,
                            run_num=experiment_number,
                        )

                        #handle = pool.apply_async(experiment, (args, dataset, clf))
                        handle = experiment(args, dataset, clf)
                        handles.append((args, datetime.now(), handle))
    print('Last args used are ' + str(args))
    print('Last handle is ' + str(handles[-1]))
    # Join here to avoid training all the classifiers while the experiments are running
    print(LOGDIR, handles, experiment_number)
    join(LOGDIR, handles, experiment_number)


@dataclass
class Combination:
    k: int
    u: int
    n_actions: int

    N_FEATURES: ClassVar[int] = 29

    def __init__(self, k: int, n_actions: int):
        self.k = k
        self.n_actions = n_actions
        self.u = Combination.N_FEATURES - k - n_actions
        if self.k + self.u + self.n_actions != Combination.N_FEATURES:
            raise ValueError(f"The sum of k, u and n_actions must be {Combination.N_FEATURES}")
        if self.n_actions < 1:
            raise ValueError("The number of actions must be at least 1")

def join(logdir: str, handles: list[tuple[Args, datetime, AsyncResult]], experiment_number):
    experiments_logs = []
    print('Initial len of handles is ' + str(len(handles)))
    while len(handles) > 0:
        i = 0
        while i < len(handles):
            print('We arrived at handles ' + str(i))
            args, start, path = handles[i]

            logging.info(f"Run {args.run_num} finished, {len(handles)} experiments remaining")
            experiments_logs.append({**args.__dict__, "start": start, "path": path}) # "end": end,
            pl.DataFrame(experiments_logs).write_csv(f"{logdir}//experiments" + str(experiment_number) + ".csv")
            del handles[i]
            i += 1


        time.sleep(1)


if __name__ == "__main__":
    UNDERSAMPLE = util.is_debugging()
    run_all_experiments()