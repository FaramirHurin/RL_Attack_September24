from datetime import timedelta, datetime

# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from banksys import Banksys, ClassificationSystem
from cardsim import Cardsim
from environment import CardSimEnv


def train(env: CardSimEnv, n_episodes: int = 1000):
    agent = ...
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, done = env.step(action)
            score += reward
        print(f"Episode {episode + 1}/{n_episodes} - Score: {score}")


def main():
    simulator = Cardsim()
    cards, terminals, transactions = simulator.simulate(n_days=50)

    # Sort and separate the last 100 transactions for testing
    transactions = sorted(transactions, key=lambda x: x.timestamp)
    transactions_train = transactions[:-100]
    transactions_test = transactions[-100:]

    # Create the classification system
    # clf = BalancedRandomForestClassifier()
    clf = RandomForestClassifier()
    system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
    print("Transactions generated: ", len(transactions))
    banksys = Banksys(system, cards, terminals, transactions_train)

    # Test the add_transaction method
    current_size = len(banksys.transactions_df)
    trx = transactions_test[0]
    # terminal = terminals[trx.terminal_id]
    # action = Action(trx.amount, terminal.x, terminal.y, trx.is_online, 0, 2)
    # step_data = StepData(action, trx.timestamp, trx.card_id)
    label = banksys.classify(trx)
    new_size = len(banksys.transactions_df)
    assert new_size == current_size + 1, "Transaction not added correctly"


if __name__ == "__main__":
    main()
