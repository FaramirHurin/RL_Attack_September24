import random
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from banksys import Banksys, ClassificationSystem, Transaction
from CardSim.cardsim import Cardsim
from environment import CardSimEnv
from rl.agents.networks import ActorCritic
from rl.agents.ppo_new import PPO
from rl.delayed_parellel_agent import DelayedParallelAgent
from Baselines.attack_generation import Attack_Generation

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


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


def train(env: CardSimEnv, n_weeks: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ActorCritic(env.observation_size, env.n_actions, device)
    agent = DelayedParallelAgent(PPO(network, 0.99))


    terminals = env.system.terminals[:10]
    # Get the transactions from the terminals
    transactions = []
    for t in terminals:
        new_transactions = t.transactions
        # Add x and y coordinates to the transactions
        new_transactions = list(map(lambda trx: trx.add_coordinates(payee_x=t.x, payee_y=t.y),
                                    new_transactions))
        transactions += new_transactions


    # Turn transactions into a tensor
    transactions_tensor = torch.tensor([[t.amount, t.payee_x, t.payee_y, t.timestamp.hour] for t in transactions])
    transactions_tensor = transactions_tensor.to(device)

    agent = DelayedParallelAgent(PPO(network, 0.99))

    i = 0
    for week_num in range(n_weeks):
        print(f"Week {week_num}")
        observations = env.reset()
        agent.reset(env.t_start, observations)
        transactions = []

        while not agent.is_done:
            i += 1
            print(i)
            t, (card, action) = agent.pop_next_action()
            step, trx = env.step(t, action, card)
            agent.store_transition(t, card, action, step)
            if trx is not None:
                transactions.append(trx)
        print("Done ")
        plot_transactions(transactions)
        input("Press Enter to continue...")


def main():
    try:
        banksys = Banksys.load()
    except FileNotFoundError:
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)

        # clf = RandomForestClassifier(n_jobs=-1)
        clf = BalancedRandomForestClassifier(n_jobs=-1, sampling_strategy=0.1)  # type:ignore
        system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
        banksys = Banksys(system, cards, terminals, transactions)
        start = datetime.now()
        test_set = banksys.train_classifier()
        print(f"Training time: {datetime.now() - start}")
        banksys.save()
        banksys.evaluate_classifier(test_set)

    env = CardSimEnv(banksys, timedelta(days=7), 100)
    train(env)


if __name__ == "__main__":
    main()
