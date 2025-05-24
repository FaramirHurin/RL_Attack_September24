import logging
import os
from datetime import datetime, timedelta

import dotenv
import numpy as np
import orjson
from marlenv import Episode, Transition
from tqdm import tqdm
from agents import Agent
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from banksys import Banksys, Transaction
from environment import SimpleCardSimEnv
from parameters import CardSimParameters, Parameters, PPOParameters, VAEParameters
from banksys.classification.system import ClassificationSystem
from banksys.banksys import Banksys
from banksys.classification.rule_based import rules_dict
dotenv.load_dotenv()  # Load the "private" .env file
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def save_parameters(directory: str, parameters: Parameters):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "parameters.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(parameters, option=orjson.OPT_SERIALIZE_NUMPY))


def save_episodes(episodes: list[Episode], directory: str):
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


def train(env: SimpleCardSimEnv, agent: Agent, n_episodes: int):
    scores = list[float]()
    episodes = list[Episode]()
    with tqdm(range(n_episodes)) as pbar:
        step_num = 0
        avg_score = 0.0
        for e in pbar:
            obs, state = env.reset()
            episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
            transactions = list[Transaction]()
            terminals = list[int]()
            while not episode.is_finished:
                step_num += 1
                action = agent.choose_action(obs.data)
                step, trx = env.step(action)
                if trx is not None:
                    terminals.append(trx.terminal_id)
                    transactions.append(trx)
                t = Transition.from_step(obs, state, action, step)
                agent.update(t, step_num)
                episode.add(t)
                obs, state = step.obs, step.state
            episode.add_metrics({"t_end": env.t.isoformat(), "terminals": terminals})
            scores.append(episode.score[0])
            episodes.append(episode)
            # Update tqdm description with average score
            if len(scores) >= 100:
                avg_score = np.mean(scores[-100:])
                pbar.set_description(f"Avg score (last 100): {avg_score:.2f}")
                if e % 100 == 0:
                    avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                    logging.info(f"{e}: avg score={avg_score:.2f}, avg_length={avg_length:.2f} (last 100)")
            else:
                pbar.set_description("Avg score (last 100): N/A")
    return episodes


def test_and_save_metrics(banksys: Banksys, directory: str):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    os.makedirs(directory, exist_ok=True)
    predicted, actual = banksys.test()
    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    confusion = confusion_matrix(actual, predicted)
    with open(os.path.join(directory, "metrics.json"), "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "confusion": confusion.tolist(),
                },
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )
    DEBUG=0


def main():
    agent_params = PPOParameters()
    #agent_params = RPPOParameters()
    # agent_params = VAEParameters()
    params = Parameters(agent_params, use_anomaly=False, rules={}, seed_value=0, cardsim=CardSimParameters(n_days=100, n_payers=20_000, contamination=0.05))
    env = params.create_env()
    agent = params.create_agent(env)
    # Sanitize the timestamp
    safe_timestamp = datetime.now().isoformat().replace(":", "-")

    # Create the directory path
    directory = os.path.join("logs", f"{params.agent_name}_{safe_timestamp}")
    save_parameters(directory, params)
    test_and_save_metrics(env.system, directory)
    #train(env, agent, params.n_episodes)


def cross_validate_classifier():
    # Load the Banksys system
    banksys = Banksys.load()
    banksys.test(predicted_labels=False)
    # Define the parameters for the classification system
    features_for_quantiles = ["amount"]
    trees_candidates = [100, 200]
    contamination_candidates = [0.005, 0.001]
    balance_factor_candidates = [0.1, 0.05]
    quantiles = [0.005, 0.995]

    results_list = []

    # Create a grid of parameters
    for trees in trees_candidates:
        for contamination in contamination_candidates:
            for balance_factor in balance_factor_candidates:
                # Create the classification system
                clf = ClassificationSystem(features_for_quantiles, quantiles, banksys, trees, contamination, balance_factor)

                # Fit the classifier
                clf.fit(banksys.train_X, banksys.train_y)

                # Make predictions on the test set
                predictions = clf.predict(banksys.test_X)

                # Calculate accuracy and F1 score
                accuracy = np.mean(predictions == banksys.test_y)
                f1 = f1_score(banksys.test_y, predictions)
                recall = recall_score(banksys.test_y, predictions)
                precision = precision_score(banksys.test_y, predictions)

                results = {
                    "trees": trees,
                    "contamination": contamination,
                    "balance_factor": balance_factor,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "recall": recall,
                    "precision": precision,
                }
                results_list.append(results)
    print(results_list)

if __name__ == "__main__":
    # main()
    cross_validate_classifier()
