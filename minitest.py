import logging
import os
import random
from datetime import datetime, timedelta
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import orjson
import torch
import typed_argparse as tap
from imblearn.ensemble import BalancedRandomForestClassifier
from marlenv import Episode, Transition
from marlenv.utils import Schedule
from sklearn.svm import OneClassSVM
from torch import nn
from tqdm import tqdm

from agents import PPO, RPPO, Agent
from agents.rl import LinearActorCritic, RecurrentActorCritic
from banksys import Banksys, Transaction
from Baselines.attack_generation import VaeAgent
from cardsim import Cardsim
from environment import CardSimEnv, SimpleCardSimEnv

# Random integer seed from 0 to 9
seed = np.random.randint(0, 10)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


parameters_run = {
    "agent_name": "ppo",  # ppo vae
    "len_episode": 4000,
    "know_client": False,
    "terminal_fract": 1,
    "seed": seed,
    "quantiles_anomaly": [0.01, 0.99],
    "use_anomaly_detection": True,
    "rules_names": ["max_trx_hour", "max_trx_week", "max_trx_day", "positive_amount"],
    "rules_values": {"max_trx_hour": 4, "max_trx_week": 20, "max_trx_day": 7, "positive_amount": 0.01},
    "ppo_hyperparameters": {
        "lr_actor": 1e-4,
        "lr_critic": 2e-4,
        "n_epochs": 32,
        "critic_c1": Schedule.linear(0.4, 0.01, 10000),
        "entropy_c2": Schedule.linear(0.4, 0.1, 10000),
        "train_interval": 64,
        "minibatch_size": 32,
        "discount": 0.99,
        "seed": seed,
    },
    "vae_hyperparameters": {
        "latent_dim": 10,
        "hidden_dim": 120,
        "lr": 0.0005,
        "trees": 20,
        "batch_size": 8,
        "num_epochs": 4000,
        "quantile": 0.99,
        "supervised": False,
    },
}
FEATURE_NAMES = ["amount"]


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
        "is_truncated": ep.is_truncated,
    }


class Args(tap.TypedArgs):
    algorithm: Literal["vae", "ppo"] = tap.arg("--algo", default="ppo")
    banksys: str = tap.arg("--banksys", default="cache/banksys.pkl")


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
    TERMINALS = env.system.terminals[-5:]
    return VaeAgent(
        device=device,
        criterion=nn.MSELoss(),
        latent_dim=parameters_run["vae_hyperparameters"]["latent_dim"],
        hidden_dim=parameters_run["vae_hyperparameters"]["hidden_dim"],
        lr=parameters_run["vae_hyperparameters"]["lr"],
        trees=parameters_run["vae_hyperparameters"]["trees"],
        banksys=env.system,
        terminal_codes=[t for t in TERMINALS],
        batch_size=parameters_run["vae_hyperparameters"]["batch_size"],
        num_epochs=parameters_run["vae_hyperparameters"]["num_epochs"],
        know_client=parameters_run["know_client"],
        supervised=parameters_run["vae_hyperparameters"]["supervised"],
        current_time=env.t,
        quantile=parameters_run["vae_hyperparameters"]["quantile"],
    )


def get_ppo(env: CardSimEnv | SimpleCardSimEnv, device: torch.device):
    network = LinearActorCritic(env.observation_size, env.n_actions, device)
    agent = PPO(
        actor_critic=network,
        gamma=parameters_run["ppo_hyperparameters"]["discount"],  #
        train_interval=parameters_run["ppo_hyperparameters"]["train_interval"],
        minibatch_size=parameters_run["ppo_hyperparameters"]["minibatch_size"],
        lr_actor=parameters_run["ppo_hyperparameters"]["lr_actor"],
        lr_critic=parameters_run["ppo_hyperparameters"]["lr_critic"],
        n_epochs=parameters_run["ppo_hyperparameters"]["n_epochs"],
        critic_c1=parameters_run["ppo_hyperparameters"]["critic_c1"],
        entropy_c2=parameters_run["ppo_hyperparameters"]["entropy_c2"],
        device=device,
    )
    return agent


def get_rppo(env: CardSimEnv | SimpleCardSimEnv, device: torch.device):
    network = RecurrentActorCritic(env.observation_size, env.n_actions, device)
    agent = RPPO(
        actor_critic=network,
        gamma=parameters_run["ppo_hyperparameters"]["discount"],  #
        train_interval=parameters_run["ppo_hyperparameters"]["train_interval"],
        lr_actor=parameters_run["ppo_hyperparameters"]["lr_actor"],
        lr_critic=parameters_run["ppo_hyperparameters"]["lr_critic"],
        n_epochs=parameters_run["ppo_hyperparameters"]["n_epochs"],
        critic_c1=parameters_run["ppo_hyperparameters"]["critic_c1"],
        entropy_c2=parameters_run["ppo_hyperparameters"]["entropy_c2"],
        device=device,
    )
    return agent


def train_simple(env: SimpleCardSimEnv, agent: Agent, agent_name: str, atk_terminals, n_episodes: int = 500):
    scores = list[float]()
    episodes = list[Episode]()

    i = 0
    for e in tqdm(range(n_episodes)):
        obs, state = env.reset()
        episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
        transactions = list[Transaction]()
        terminals = list[int]()
        while not episode.is_finished:
            # Normalize the observation if model is PPO
            if isinstance(agent, PPO) and parameters_run["know_client"]:
                obs.data[-2:] = obs.data[-2:] / 200
                action = agent.choose_action(obs.data)
            else:
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
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

    filename = f"logs/tests/{agent_name}_{time}.json"
    parameters_run["filename"] = filename
    if agent_name == "vae":
        episodes = [fix_episode_for_serialization(ep) for ep in episodes]
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))

    # Append the parameters to a file calles logs/test_parameters.json
    os.makedirs("logs", exist_ok=True)
    filename = "logs/test_parameters.json"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            parameters = orjson.loads(f.read())
    else:
        parameters = []
    parameters.append(parameters_run)
    with open(filename, "wb") as f:
        f.write(orjson.dumps(parameters, option=orjson.OPT_SERIALIZE_NUMPY))


def main(args: Args):
    try:
        banksys = Banksys.load(args.banksys)
    except FileNotFoundError:
        print("Banksys not found, creating a new one")
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)
        # clf = RandomForestClassifier(n_jobs=-1)
        # banksys = Banksys(cards, terminals, simulator.t_start, feature_names=FEATURE_NAMES,
        #                   quantiles= [0.02, 0.98], rules=RULES)
        # system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], banksys=banksys, rules=RULES)

        clf = BalancedRandomForestClassifier(30, n_jobs=1, sampling_strategy=0.2)  # type: ignore
        anomaly_detection_clf = OneClassSVM(nu=0.005)

        # simulator._start + 30 days
        TRAINING_DAYS = 30
        attack_time = simulator.t_start + timedelta(days=TRAINING_DAYS)

        banksys = Banksys(
            inner_clf=clf,
            anomaly_detection_clf=anomaly_detection_clf,
            cards=cards,
            terminals=terminals,
            t_start=simulator.t_start,
            attack_time=attack_time,
            transactions=transactions,
            feature_names=FEATURE_NAMES,
            quantiles=parameters_run["quantiles_anomaly"],
        )

        start = datetime.now()
        print(f"Training time: {datetime.now() - start}")
        banksys.save(args.banksys)
        # TODO: banksys.evaluate_classifier(test_set)

    confusion_matrix = banksys.set_up_run(
        use_anomaly_detection=parameters_run["use_anomaly_detection"],
        rules=parameters_run["rules_names"],
        rules_values=parameters_run["rules_values"],
        return_confusion=False,
    )
    print(confusion_matrix)

    env = SimpleCardSimEnv(
        banksys,
        timedelta(days=7),
        customer_location_is_known=parameters_run["know_client"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_name = parameters_run["agent_name"]  # args.algorithm
    match agent_name:
        case "ppo":
            agent = get_ppo(env, device)
        case "vae":
            agent = get_vae(env, device)
        case other:
            raise ValueError(f"Unknown algorithm: {other}")
    # Random sample terminals
    atk_terminals = random.sample(env.system.terminals, int(len(env.system.terminals) * parameters_run["terminal_fract"]))  #
    train_simple(env, agent, agent_name, atk_terminals, n_episodes=parameters_run["len_episode"])


if __name__ == "__main__":
    args = tap.Parser(Args).bind(main).run()
    # for i in range(10):
    #     main()
