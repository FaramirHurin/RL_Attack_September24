import logging
import os
import random
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import orjson
import torch
import typed_argparse as tap
from marlenv import Episode, Transition
from tqdm import tqdm

from banksys import Banksys, Transaction
from cardsim import Cardsim
from environment import SimpleCardSimEnv
from parameters import PPOParameters, Parameters, RPPOParameters, VAEParameters

# Import Action
from environment.action import Action

# Random integer seed from 0 to 9
seed = np.random.randint(0, 10)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


FEATURE_NAMES = ["amount"]


class Args(tap.TypedArgs):
    algorithm: Literal["vae", "ppo"] = tap.arg("--algo", default="ppo")
    banksys: str = tap.arg("--banksys", default="cache/banksys.pkl")


def fix_episode_for_serialization(ep: Episode):
    def to_float_list(arr):
        return [float(x) for x in arr]

    def fix_metrics(metrics):
        fixed = {}
        for k, v in metrics.items():
            if isinstance(v, datetime):
                fixed[k] = v.isoformat()
            else:
                fixed[k] = v
        return fixed

    return {
        "all_observations": [obs.tolist() for obs in ep.all_observations],
        "all_extras": [ex.tolist() for ex in ep.all_extras],
        "actions": [to_float_list(a) for a in ep.actions],
        "rewards": [r.tolist() for r in ep.rewards],
        "all_available_actions": [a.tolist() for a in ep.all_available_actions],
        "all_states": [s.tolist() for s in ep.all_states],
        "all_states_extras": [se.tolist() for se in ep.all_states_extras],
        "metrics": fix_metrics(ep.metrics),  # preserve score-0
        "episode_len": ep.episode_len,
        "other": ep.other,
        "is_done": ep.is_done,
        "is_truncated": ep.is_truncated,
    }


def save_parameters(directory: str, parameters: Parameters):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "parameters.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(parameters, option=orjson.OPT_SERIALIZE_NUMPY))


def save_episodes(episodes: list[Episode], directory: str):
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps([fix_episode_for_serialization(ep) for ep in episodes], option=orjson.OPT_SERIALIZE_NUMPY))


def train(env: SimpleCardSimEnv, params: Parameters, directory: str):
    agent = params.get_agent(env, device)
    atk_terminals = random.sample(env.system.terminals, len(env.system.terminals) * params.terminal_fract)
    scores = list[float]()
    episodes = list[Episode]()
    with tqdm(range(params.n_episodes)) as pbar:
        i = 0
        avg_score = 0.0
        for e in pbar:
            obs, state = env.reset()
            episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
            transactions = list[Transaction]()
            terminals = list[int]()
            while not episode.is_finished:
                action = agent.choose_action(obs.data)
                step, trx = env.step(action, atk_terminals)
                if trx is not None:
                    terminals.append(trx.terminal_id)
                    transactions.append(trx)
                t = Transition.from_step(obs, state, action, step)
                agent.update(t, i)
                episode.add(t)
                episode.add_metrics({"t_end": env.t, "terminals": terminals})
                obs, state = step.obs, step.state
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
    save_episodes(episodes, directory)


def init_environment(params: Parameters):
    try:
        banksys = Banksys.load(params.cardsim)
    except (FileNotFoundError, ValueError):
        print("Banksys not found, creating a new one")
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(
            n_days=params.cardsim.n_days,
            n_payers=params.cardsim.n_payers,
            start_date=params.cardsim.start_date,
        )
        banksys = Banksys(
            cards=cards,
            terminals=terminals,
            training_duration=timedelta(days=params.n_days_training),
            transactions=transactions,
            feature_names=FEATURE_NAMES,
            quantiles=params.quantiles_anomaly,
        )
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
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

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
    # agent_params = RPPOParameters()
    agent_params = VAEParameters()
    params = Parameters(agent_params, use_anomaly=False, rules={})
    env = init_environment(params)
    # Sanitize the timestamp
    safe_timestamp = datetime.now().isoformat().replace(":", "-")

    # Create the directory path
    directory = os.path.join("logs", f"{params.agent_name}_{safe_timestamp}")
    save_parameters(directory, params)
    # test_and_save_metrics(env.system, directory)
    train(env, params, directory)


if __name__ == "__main__":
    main()
