# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
from banksys import Banksys, ClassificationSystem
from cardsim import Cardsim
from environment import CardSimEnv
from rl.agents.ppo_new import PPO
from rl.agents.networks import ActorCritic
import torch


def train(env: CardSimEnv, n_episodes: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ActorCritic(env.observation_size, env.n_actions, device)
    agent = PPO(network, 0.99)
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs, reward, done = env.step(action)
            score += reward
        print(f"Episode {episode + 1}/{n_episodes} - Score: {score}")


def main():
    try:
        banksys = Banksys.load()
    except FileNotFoundError:
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)

        # Sort and separate the last 100 transactions for testing
        transactions = sorted(transactions, key=lambda x: x.timestamp)
        transactions_train = transactions[:-100]

        clf = RandomForestClassifier()
        system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
        banksys = Banksys(system, cards, terminals, transactions_train)
        banksys.save()

    env = CardSimEnv(banksys, timedelta(days=7))
    train(env)


if __name__ == "__main__":
    main()
