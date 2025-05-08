import random
from datetime import datetime, timedelta
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import orjson
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from functools import cached_property
from imblearn.ensemble import BalancedRandomForestClassifier

from banksys import Banksys, ClassificationSystem, Transaction, Card
from environment import CardSimEnv, SimpleCardSimEnv, Action
from rl.agents.networks import ActorCritic
from rl.agents import PPO
from rl.delayed_parallel_agent import DelayedParallelAgent
from Baselines.attack_generation import Attack_Generation, VaeAgent
from cardsim import Cardsim
from torch import nn
from marlenv import Episode, Transition
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


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


def get_vae(env: CardSimEnv, device: torch.device):
    TERMINALS = env.system.terminals[:5]
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
        0.99,
        train_interval=64,
        minibatch_size=32,
        lr_actor=1e-3,
        lr_critic=1e-3,
        n_epochs=64,
        critic_c1=0.5,
        entropy_c2=0.01,
    )
    return agent


def train(env: CardSimEnv, n_weeks: int = 20, n_cards: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_ppo(env, device)
    agent = DelayedParallelAgent(agent)
    reset_data = env.reset(n_parallel=n_cards)
    agent.reset(env.t, reset_data)
    end_time = env.t + timedelta(weeks=n_weeks)

    while env.t < end_time:
        end_week = env.t + timedelta(weeks=1)
        while env.t < end_week:
            transactions = list[Transaction]()
            card, action = agent.pop_next_action()
            step, trx = env.step(action, card)
            agent.store_transition(env.t, card, action, step)
            if trx is None:
                # Card has been blocked
                new_card = env.steal_card()
                obs = env.get_observation(new_card)
                pass
            if trx is not None:
                transactions.append(trx)
            print("Done")
            plot_transactions(transactions)
            input("Press Enter to continue to next week...")


def train_simple(env: SimpleCardSimEnv, n_episodes: int = 2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_ppo(env, device)
    scores = list[float]()
    episodes = list[Episode]()

    for e in tqdm(range(n_episodes)):
        obs, state = env.reset()
        episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
        transactions = list[Transaction]()
        terminals = list[int]()
        while not episode.is_finished:
            action = agent.choose_action(obs.data)
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
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/{datetime.now()}.json"
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


def main():
    try:
        banksys = Banksys.load()
    except FileNotFoundError:
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)

        # clf = RandomForestClassifier(n_jobs=-1)
        clf = BalancedRandomForestClassifier(n_jobs=-1, sampling_strategy=0.5)  # type:ignore
        system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
        banksys = Banksys(system, cards, terminals, simulator.t_start)
        start = datetime.now()
        test_set = banksys.train_classifier(transactions)
        print(f"Training time: {datetime.now() - start}")
        banksys.save()
        banksys.evaluate_classifier(test_set)

    env = SimpleCardSimEnv(banksys, timedelta(days=7), customer_location_is_known=True)
    train_simple(env, n_episodes=2000)


if __name__ == "__main__":
    for i in range(10):
        main()
