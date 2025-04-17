# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
from banksys import Banksys, ClassificationSystem
from cardsim import Cardsim
from environment import CardSimEnv
from rl.agents.ppo_new import PPO
from rl.agents.networks import ActorCritic
import torch
from datetime import datetime
from marlenv import Transition


def train(env: CardSimEnv, n_episodes: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ActorCritic(env.observation_size, env.n_actions, device)
    agent = PPO(network, 0.99)
    t = 0
    for episode in range(n_episodes):
        obs, state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs.data)
            step = env.step(action)
            agent.update_step(Transition.from_step(obs, state, action.to_numpy(), step), t)
            done = step.is_terminal
            obs = step.obs
            state = step.state
            t += 1
            score += step.reward
        print(f"Episode {episode + 1}/{n_episodes} - Score: {score}")


def main():
    try:
        banksys = Banksys.load()
    except FileNotFoundError:
        simulator = Cardsim()
        cards, terminals, transactions = simulator.simulate(n_days=50)

        clf = RandomForestClassifier(n_jobs=-1)
        system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
        banksys = Banksys(system, cards, terminals, transactions)
        start = datetime.now()
        test_set = banksys.train_classifier()
        print(f"Training time: {datetime.now() - start}")
        banksys.save()
        banksys.evaluate_classifier(test_set)

    env = CardSimEnv(banksys, timedelta(days=7))
    train(env)


if __name__ == "__main__":
    main()
