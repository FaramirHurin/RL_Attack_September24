import random
from datetime import datetime, timedelta
import logging
from typing import Literal
import typed_argparse as tap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import orjson
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from functools import cached_property
from agents import Agent
from imblearn.ensemble import BalancedRandomForestClassifier

from banksys import Banksys, ClassificationSystem, Transaction, Card
from environment import CardSimEnv, SimpleCardSimEnv, Action
from agents.rl import ActorCritic
from agents import PPO
from Baselines.attack_generation import Attack_Generation, VaeAgent
from cardsim import Cardsim
from torch import nn
from marlenv import Episode, Transition
from marlenv.utils import Schedule
from tqdm import tqdm
from sklearn.ensemble import IsolationForest

# Random integer seed from 0 to 9
seed = np.random.randint(0, 10)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def fix_episode_for_serialization(ep):
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
        "metrics": fix_metrics(ep.metrics),  # ✅ preserve score-0
        "episode_len": ep.episode_len,
        "other": ep.other,
        "is_done": ep.is_done,
        "is_truncated": ep.is_truncated
    }


class Args(tap.TypedArgs):
    algorithm: Literal["vae", "ppo"] = tap.arg("--algo", default="ppo")
    banksys: str = tap.arg("--banksys", default="cache/banksys.pkl")

def max_trx_day(transaction:Transaction, transactions:[list[Transaction]], max_number:int=7) -> bool:
    same_day_transactions = [trx for trx in transactions if trx.timestamp.date() == transaction.timestamp.date()]
    return len(same_day_transactions) > max_number

def max_trx_hour(transaction:Transaction, transactions:[list[Transaction]], max_number:int=4) -> bool:
    same_hour_transactions = [trx for trx in transactions if trx.timestamp.hour == transaction.timestamp.hour]
    return len(same_hour_transactions) > max_number

def max_trx_week(transaction:Transaction, transactions:[list[Transaction]], max_number:int=20) -> bool:
    same_week_transactions = [trx for trx in transactions if trx.timestamp.isocalendar()[1] ==
                              transaction.timestamp.isocalendar()[1]]
    return len(same_week_transactions) > max_number

def positive_amount(transaction:Transaction, transactions:[list[Transaction]]) -> bool:
    return transaction.amount < 0.01

FEATURE_NAMES = ['amount']
QUANTILES = [0.01, 0.99]
RULES = [max_trx_hour, max_trx_week, max_trx_day,positive_amount] #



def plot_transactions(transactions: list[Transaction]):
    fig, ax = plt.subplots()
    ax.set_xlim([transactions[0].timestamp, transactions[-1].timestamp])  # type: ignore
    # Optional: format date axis
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d-%b"))

    # Set labels
    ax.set_title("Transactions over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")

    COLOURS = {}
    fraud_dates, fraud_amounts, fraud_colours = [], [], []
    genuine_dates, genuine_amounts, genuine_colours = [], [], []
    for t in transactions:
        if t.card_id not in COLOURS:
            COLOURS[t.card_id] = np.random.rand(3)
        if t.label:
            fraud_dates.append(t.timestamp)
            fraud_amounts.append(t.amount)
            fraud_colours.append(COLOURS[t.card_id])
        else:
            genuine_dates.append(t.timestamp)
            genuine_amounts.append(t.amount)
            genuine_colours.append(COLOURS[t.card_id])
    # Create a scatter plot
    ax.scatter(fraud_dates, fraud_amounts, c=fraud_colours, marker="x", s=50)
    ax.scatter(genuine_dates, genuine_amounts, c=genuine_colours, marker="o", s=50)
    fig.show()


def get_vae(env: SimpleCardSimEnv, device: torch.device):
    TERMINALS = env.system.terminals[:10]
    return VaeAgent(
        device=device,
        criterion=nn.MSELoss(),
        latent_dim=10,
        hidden_dim=120,
        lr=0.0005,
        trees=20,
        banksys=env.system,
        terminal_codes=[t for t in TERMINALS],
        batch_size=8,
        num_epochs=4000,
        know_client=True,
        supervised=False,
        current_time=env.t,
        quantile=0.99,
    )


def get_ppo(env: CardSimEnv | SimpleCardSimEnv, device: torch.device):
    network = ActorCritic(env.observation_size, env.n_actions, device)
    agent = PPO(
        network,
        1, #0.99
        train_interval=64,
        minibatch_size=32,
        lr_actor=1e-4,
        lr_critic=4e-4,
        n_epochs=16,
        critic_c1=Schedule.linear(0.2, 0.001, 10000),
        entropy_c2=Schedule.linear(0.35, 0.001, 1000),
        device=device,
    )
    return agent


def train_simple(env: SimpleCardSimEnv, agent: Agent, agent_name:str, n_episodes: int = 500):
    scores = list[float]()
    episodes = list[Episode]()

    for e in tqdm(range(n_episodes)):
        obs, state = env.reset()
        episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
        transactions = list[Transaction]()
        terminals = list[int]()
        while not episode.is_finished:
            # Normalize the observation if model is PPO
            """
            if isinstance(agent, PPO):
                if known_client:
                    deviding_factor = np.max(obs.data[-2:])
                    obs.data[-2:] = obs.data[-2:] / deviding_factor
                #print(obs.data)
                action = agent.choose_action(obs.data)
            else:
            """
            action = agent.choose_action(obs.data)
            #print('.')
            step, trx = env.step(action)
            if trx is not None:
                terminals.append(trx.terminal_id)
                transactions.append(trx)
            t = Transition.from_step(obs, state, action, step)
            agent.update(t)
            episode.add(t)
            episode.add_metrics({"t_end": env.t, "terminals": terminals})
            obs, state = step.obs, step.state
        scores.append(episode.score[0])
        episodes.append(episode)
        logging.info(
            f"{e:5d} score={episode.score[0]:9.2f}, avg score={np.mean(scores[-50:]):5.2f}, length={len(episode):3d} steps, t_end={env.t.date()} {env.t.time()}"
        )
        if e % 100 == 0:
            # Print average over last 50 episodes
            avg_score = np.mean(scores[-100:])
            print(f"Episode {e}: average score over last 100 episodes: {avg_score:.2f}")
            # Average length of last 50 episodes
            avg_length = np.mean([len(ep) for ep in episodes[-100:]])
            print(f"Episode {e}: average length over last 100 episodes: {avg_length:.2f}")

    os.makedirs("logs", exist_ok=True)
    filename = f"logs/tests/{agent_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}.json"
    if agent_name == 'vae':
        episodes = [fix_episode_for_serialization(ep) for ep in episodes]
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


def main(args: Args):
    try:
        banksys = Banksys.load(args.banksys)
    except FileNotFoundError:
        print('Banksys not found, creating a new one')
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)
        # clf = RandomForestClassifier(n_jobs=-1)
        # banksys = Banksys(cards, terminals, simulator.t_start, feature_names=FEATURE_NAMES,
        #                   quantiles= [0.02, 0.98], rules=RULES)
        #system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], banksys=banksys, rules=RULES)

        clf = BalancedRandomForestClassifier(30, n_jobs=1, sampling_strategy=0.2)
        anomaly_detection_clf = IsolationForest(n_jobs=1, contamination=0.005)


        banksys = Banksys( inner_clf=clf, anomaly_detection_clf=anomaly_detection_clf, cards=cards, terminals=terminals,
                           t_start= simulator.t_start, transactions=transactions, feature_names=FEATURE_NAMES,
                           quantiles=QUANTILES, rules=RULES)

        start = datetime.now()
        test_set = banksys.train_classifier(transactions)
        print(f"Training time: {datetime.now() - start}")
        banksys.save(args.banksys)
        #TODO: banksys.evaluate_classifier(test_set)

    env = SimpleCardSimEnv(banksys, timedelta(days=7), customer_location_is_known=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_name = 'vae' # args.algorithm
    match agent_name:
        case "ppo":
            agent = get_ppo(env, device)
        case "vae":
            agent = get_vae(env, device)
        case other:
            raise ValueError(f"Unknown algorithm: {other}")
    train_simple(env, agent,agent_name, n_episodes=4000)



if __name__ == "__main__":
    args = tap.Parser(Args).bind(main).run()
    # for i in range(10):
    #     main()
