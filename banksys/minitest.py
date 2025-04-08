from cardsim import Cardsim
from card import Card
from terminal import Terminal
from transaction import Transaction
from banksys import Banksys
from classification import ClassificationSystem
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from environment import StepData
from environment import StepData, Action

def main():
    simulator = Cardsim()
    cards, terminals, transactions = simulator.simulate(n_days=50, n_payers=100)

    # Sort and separate the last 100 transactions for testing
    transactions = sorted(transactions, key=lambda x: x.timestamp)
    transactions_train = transactions[:-100]
    transactions_test = transactions[-100:]

    # Create the classification system
    #clf = BalancedRandomForestClassifier()
    clf = RandomForestClassifier()
    system = ClassificationSystem(clf, ["amount"], [0.02, 0.98], [])
    print('Transactions generated: ', len(transactions))
    banksys = Banksys(system, cards, terminals, transactions_train)

    # Test the add_transaction method
    current_size = len(banksys.transactions_df)
    trx = transactions_test[0]
    terminal = terminals[trx.terminal_id]
    action = Action(trx.amount, terminal.x, terminal.y, trx.is_online, 0, 2)
    step_data = StepData(action, trx.timestamp, trx.card_id)
    banksys.classify(step_data)
    new_size = len(banksys.transactions_df)
    assert new_size == current_size + 1, "Transaction not added correctly"



if __name__ == '__main__':
    main()




