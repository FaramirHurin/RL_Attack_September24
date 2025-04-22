# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
from banksys import Banksys, ClassificationSystem
from cardsim import Cardsim
from environment import CardSimEnv
from rl.agents.ppo_new import PPO
from rl.agents.networks import ActorCritic
import torch
from collections import OrderedDict
from datetime import datetime
from marlenv import Episode


def train(env: CardSimEnv, n_episodes: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ActorCritic(env.observation_size, env.n_actions, device)
    agent = PPO(network, 0.99)
    for episode in range(n_episodes):
        episodes = dict[int, Episode]()
        observations, states = env.reset()
        actions = OrderedDict((card_id, agent.choose_action(obs.data)) for card_id, obs in observations.items())
        step, card_id = env.first_step(list(actions.values()))

        episodes[card_id] = Episode.new(observations[card_id], states[card_id])
        episodes[card_id].add(step, actions[card_id])

        obs = step.obs
        state = step.state
        while not step.is_terminal:
            action = agent.choose_action(step.obs.data)
            step, card_id = env.step(action, card_id)
            if card_id not in episodes:
                episodes[card_id] = Episode.new(obs, state)
            episodes[card_id].add(step, action)

            # agent.update_step(Transition.from_step(obs, state, action.to_numpy(), step), t)
            obs = step.obs
            state = step.state


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

    env = CardSimEnv(banksys, timedelta(days=7), 10)
    train(env)


if __name__ == "__main__":
    main()
