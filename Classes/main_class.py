import logging
import time
import pandas as pd
from datetime import datetime
from multiprocessing.pool import AsyncResult
from typing import Literal, ClassVar
from sklearn.neural_network import MLPClassifier
import rl
import warnings
import numpy as np
import polars as pl
import typed_argparse as tap
from sklearn.ensemble import RandomForestClassifier
import os
import util
import polars as pl
from itertools import product
import random


from Classes.datset_inteface import Dataset
from quantile_forest import QuantileForest_Bandit
from rl.agents import PPO
from Baselines.baselines_classes import BaselineAgent
from fraud_env import FraudEnv
from datset_inteface import DatasetLoader
# Add the folder to sys.path
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/rl'))
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Baselines'))
"""
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(parent_dir, 'rl')
sys.path.append(target_dir)
target_dir = os.path.join(parent_dir, 'Baselines')
sys.path.append(target_dir)
pd.options.display.max_columns = None
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
dotenv.load_dotenv()
logging.getLogger().setLevel(os.getenv("LOG_LEVEL", "ERROR"))
logging.basicConfig(level=logging.ERROR)  # Only log ERROR level messages and above
"""
warnings.filterwarnings("ignore", message="X does not have valid feature names")
pd.set_option('display.max_columns', None)


class Args(tap.TypedArgs):
    logdir: str = tap.arg(help="Directory to save logs")
    fraud_balance: Literal["very-imbalanced", "imbalanced", "balanced"] = tap.arg(help="Fraud balance in synthetic datasets")
    k: int = tap.arg(help="Number of known features")
    u: int = tap.arg(help="Number of unknown features")
    c: int = tap.arg(help="Number of controllable features")
    n_steps: int = tap.arg(help="Number of steps to train the agent")
    challenge: Literal["fraudulent", "genuine", "all"] = tap.arg(help="Only use frauds in the environment observations")
    reward_type: Literal["probability", "label"] = tap.arg(help="Type of reward to use")
    run_num: int = tap.arg(help="Run number")
    min_values: np.array  = tap.arg(help="Min values features can have")
    max_values: np.array  =tap.arg(help="Max values features can have")

    @property
    def directory(self):
        if len(self.u) > 0:
            u_to_print = self.u[0]
        else:
            u_to_print = 0
        if len(self.k) > 0:
            k_to_print = self.k[0]
        else:
            k_to_print = 0
        to_return =  os.path.join(self.logdir,  f"k-{k_to_print}", f"u-{u_to_print}", self.challenge,
                                  f"reward-{self.reward_type}") #self.dataset,

        return to_return

def process_generator_columns(df):

    # Define fixed C_columns as per the provided instructions
    C_columns_fixed = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT']

    # Separate remaining columns, excluding the fixed C_columns
    remaining_columns = [col for col in df.columns if col not in C_columns_fixed]

    # Identify CUSTOMER_ID and TERMINAL columns
    customer_id_columns = [col for col in remaining_columns if col.startswith("CUSTOMER_ID")]
    terminal_columns = [col for col in remaining_columns if col.startswith("TERMINAL")]

    # Remaining columns after CUSTOMER_ID and TERMINAL
    other_columns = [col for col in remaining_columns if col not in customer_id_columns + terminal_columns]

    # Generate all combinations of CUSTOMER_ID and TERMINAL columns among K, U, and C
    combinations = []

    for customer_k, customer_u, customer_c in product(
            [customer_id_columns, []], [customer_id_columns, []], [customer_id_columns, []]):

        # Ensure each CUSTOMER_ID column goes only into one of K, U, or C in this combination
        if len(customer_k) + len(customer_u) + len(customer_c) != len(customer_id_columns):
            continue

        for terminal_k, terminal_u, terminal_c in product(
                [terminal_columns, []], [terminal_columns, []], [terminal_columns, []]):

            # Ensure each TERMINAL column goes only into one of K, U, or C in this combination
            if len(terminal_k) + len(terminal_u) + len(terminal_c) != len(terminal_columns):
                continue

            # Build K, U, and C column groups
            K_columns = customer_k + terminal_k
            U_columns = customer_u + terminal_u
            C_columns = C_columns_fixed + customer_c + terminal_c

            # Append the combination as a dictionary
            combinations.append({
                'K_columns': K_columns,
                'U_columns': U_columns,
                'C_columns': C_columns
            })
    return combinations


def sample_columns(df, seed, num_combinations):
    """
    Helper function to generate random combinations of K, U, and C columns
    from the DataFrame columns using a specified seed.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Get all columns in the DataFrame
    all_columns = df.columns

    # Store generated combinations
    combinations = []

    for _ in range(num_combinations):
        # Shuffle columns randomly and split into K, U, and C of different sizes
        shuffled_columns = random.sample(all_columns, len(all_columns))

        # Split columns into random proportions for K, U, and C
        k_size = random.randint(1, max(1, len(all_columns) // 3))
        u_size = random.randint(1, max(1, (len(all_columns) - k_size) // 2))
        c_size = len(all_columns) - k_size - u_size

        # Assign columns to K, U, and C
        K_columns = shuffled_columns[:k_size]
        U_columns = shuffled_columns[k_size:k_size + u_size]
        C_columns = shuffled_columns[k_size + u_size:]

        if 'Unnamed:0' in K_columns:
            K_columns.remove('Unnamed:0')
        elif 'Unnamed:0' in U_columns:
            U_columns.remove('Unnamed:0')
        elif 'Unnamed:0' in C_columns:
            C_columns.remove('Unnamed:0')
        if 'Unnamed: 0' in K_columns:
            K_columns.remove('Unnamed: 0')
        elif 'Unnamed: 0' in U_columns:
            U_columns.remove('Unnamed: 0')
        elif 'Unnamed: 0' in C_columns:
            C_columns.remove('Unnamed: 0')


        # Append the combination as a dictionary
        combinations.append({
            'K_columns': K_columns,
            'U_columns': U_columns,
            'C_columns': C_columns
        })



    return combinations

def process_sklearn_columns(df, seed=42, num_combinations=5):
    """
    Generates random combinations of K, U, and C columns for the sklearn dataset type.
    """
    return sample_columns(df, seed, num_combinations)

def process_kaggle_columns(df, seed=123, num_combinations=5):
    """
    Generates random combinations of K, U, and C columns for the kaggle dataset type.
    """
    return sample_columns(df, seed, num_combinations)


def get_column_combinations(dataset_type: str, df: pl.DataFrame, seed=42, num_combinations=10):
    # Remove 'Unnamed_0' column if it exists

    if dataset_type == 'Generator':
        return process_generator_columns(df)
    elif dataset_type == 'SkLearn':
        return process_sklearn_columns(df, seed, num_combinations)
    elif dataset_type == 'Kaggle':
        return process_kaggle_columns(df, seed, num_combinations)
    else:
        raise ValueError("Invalid dataset_type. Choose from 'Generator', 'SkLearn', or 'Kaggle'.")
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


def experiment(args: Args, dataset: Dataset, clf: RandomForestClassifier | MLPClassifier):
    env = FraudEnv(
        transactions=dataset.env_transactions(args.challenge),
        k=args.k,
        u=args.u,
        c = args.c,
        classifier=clf,
        reward_type=args.reward_type,
        min_values=args.min_values,
        max_values=args.max_values
    )
    print('Number of frauds is ' + str(sum(dataset._test_y)))
    x_train = dataset.train_x[args.c + args.u + args.k]

    action_min = env.min_values
    action_max = env.max_values
    assert env.n_actions > 0

    bandit = QuantileForest_Bandit(action_min, action_max)
    ppo = PPO(
        env.observation_shape[0],
        env.n_actions,
        lr_actor=5e-4,
        gamma=0.9,
        lr_critic=5e-4,
        k_epochs=30,
        eps_clip=0.15,
    )
    logs = dict[str, np.ndarray]()
    logs["PPO"] = rl.train.train_agent(ppo, env, args.n_steps)
    # logs["Quantile Forest"] = rl.train.train_bandit(bandit, env, args.n_steps)

    mimicry_transactions = "genuine"
    for sampling in ("multivariate", "univariate", "uniform", "mixture"):
        for dataset_size in ("1k",  "100%"): # "5%",
            train_X = dataset.mimicry_TR(mimicry_transactions, dataset_size, args.c) # Controllable features
            agent = BaselineAgent(train_X=train_X, generation_method=sampling, action_min=action_min, action_max=action_max )
            key = f"{sampling}-{mimicry_transactions}-{dataset_size}"
            logs[key] = test_agent(env, clf, agent, args.n_steps, args.reward_type)
            del agent

    os.makedirs(str(args.directory), exist_ok=True)
    filename = str(args.directory)+"/file.csv"
    pd.DataFrame(logs).to_csv(filename, index=False)
    print('Controllable are' + str(args.c) + ' Fixed are' + str(args.k) + ' Unknown are' + str(args.u))
    print(args.directory)
    print(pd.DataFrame(logs).describe())

    return filename


def run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_name, min_max_quantile, VARIABLES_STEPS, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS):
    LOGDIR_OUTER = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(LOGDIR_OUTER, exist_ok=False)
    for dataset_type in dataset_types:

        LOGS_DATASET = os.path.join(LOGDIR_OUTER, dataset_type)
        os.makedirs(LOGS_DATASET, exist_ok=False)

        n_processes = N_GPUS * PROCESS_PER_GPU
        # pool = mp.Pool(n_processes)

        dataset_loader = DatasetLoader( dataset_type=dataset_type, classifier=classifier_name,
                                         n_features_list=n_features_list, clusters_list=clusters_list,
                                         class_sep_list=class_sep_list,balance_list=balance_list, )
        datasets, classifiers = dataset_loader.load()

        for key in datasets.keys():
            dataset = datasets[key]
            classifier = classifiers[key]
            for experiment_number in range(N_REPETITIONS):

                LOGDIR = os.path.join(LOGS_DATASET, "_" + str(experiment_number))
                if dataset_type == 'SkLearn':
                    folder_name = f"n_features={key[2]}_n_clusters={key[3]}_class_sep={key[4]}_balance={key[5]}"
                elif dataset_type == 'Kaggle':
                    folder_name = f"balance={key[2]}"
                elif dataset_type == 'Generator':
                    folder_name = f"balance={key[2]}"

                handles = [] #?
                save_path = f"{LOGDIR}/{folder_name}/{experiment_number}"

                # print('Save path is ' + str(save_path))
                print("Experiment number is " + str(experiment_number))

                os.makedirs(save_path, exist_ok=False)  # ?d
                df_negative = dataset.env_transactions("genuine")
                min_values = df_negative.quantile(min_max_quantile)
                max_values = df_negative.quantile(1 - min_max_quantile)
                if 'Unnamed_0' in df_negative.columns:
                    df_negative = df_negative.drop('Unnamed_0')
                columns_combination = get_column_combinations(dataset_type=dataset_type, df=df_negative)
                print('These keys are ' + str(datasets.keys()) +'EXPERIMENT NUMBER '
                      + str(experiment_number) + ' We are doing these combinations ' + str(len(columns_combination)))

                for index in range(len(columns_combination)):
                    K_COLUMNS = columns_combination[index]['K_columns']
                    U_COLUMNS = columns_combination[index]['U_columns']
                    C_COLUMNS = columns_combination[index]['C_columns']

                    for reward_type in (["label"]): # "probability",
                        args = Args(logdir=save_path, k=K_COLUMNS, c =C_COLUMNS, u = U_COLUMNS, n_steps=N_STEPS,
                        challenge="fraudulent", reward_type=reward_type, run_num=experiment_number,
                                    min_values=min_values, max_values=max_values)
                        handle = experiment(args, dataset, classifier)
                        handles.append((args, datetime.now(), handle))

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


UNDERSAMPLE = util.is_debugging()
dataset_types = [ 'Generator']  # Kaggle  Generator SkLearn
n_features_list = [16, 32, 64]
clusters_list = [1, 8, 16]
class_sep_list = [0.5, 1, 2, 8]
balance_list = [0.1, 0.5]
classifier_name = 'RF'
min_max_quantile = 0.025
VARIABLES_STEPS = 1000
N_REPETITIONS = 1 # 20
N_STEPS = 1000  # 0
PROCESS_PER_GPU = 2
N_GPUS = 1  # 8
run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_name, min_max_quantile, VARIABLES_STEPS, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS)

# IMPOSE RANDOMIZATION