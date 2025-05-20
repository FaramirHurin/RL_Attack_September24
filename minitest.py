import logging
import os
from datetime import datetime, timedelta

import numpy as np
import orjson
from marlenv import Episode, Transition
from tqdm import tqdm

from banksys import Banksys, Transaction
from environment import SimpleCardSimEnv
from parameters import Parameters, PPOParameters, RPPOParameters, VAEParameters, CardSimParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_parameters(directory: str, parameters: Parameters):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "parameters.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(parameters, option=orjson.OPT_SERIALIZE_NUMPY))


def save_episodes(episodes: list[Episode], directory: str):
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


def train(env: SimpleCardSimEnv, params: Parameters):
    agent = params.get_agent(env)
    scores = list[float]()
    episodes = list[Episode]()
    with tqdm(range(params.n_episodes)) as pbar:
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


def init_environment(params: Parameters):
    try:
        banksys = Banksys.load(params.cardsim)
    except (FileNotFoundError, ValueError):
        print("Banksys not found, creating a new one")
        banksys = params.create_banksys()
        banksys.save(params.cardsim)

    banksys.set_up_run(rules_values=params.rules, use_anomaly=params.use_anomaly)
    env = SimpleCardSimEnv(
        banksys,
        timedelta(days=params.avg_card_block_delay_days),
        customer_location_is_known=params.know_client,
        normalize_location=params.agent_name in ("ppo", "rppo"),
    )
    return env


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


def main():
    # agent_params = PPOParameters()
    agent_params = RPPOParameters()
    # agent_params = VAEParameters()
    params = Parameters(agent_params, use_anomaly=False, rules={}, seed=0, cardsim=CardSimParameters(n_days=100, n_payers=20_000))
    env = init_environment(params)
    exit()
    # Sanitize the timestamp
    safe_timestamp = datetime.now().isoformat().replace(":", "-")

    # Create the directory path
    directory = os.path.join("logs", f"{params.agent_name}_{safe_timestamp}")
    save_parameters(directory, params)
    # test_and_save_metrics(env.system, directory)
    train(env, params, directory)


if __name__ == "__main__":
    main()
