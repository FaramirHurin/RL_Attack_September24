from cardsim.cardsim import Cardsim
from banksys import Banksys, ClassificationSystem
from imblearn.ensemble import BalancedRandomForestClassifier


def main():
    rf = BalancedRandomForestClassifier(sampling_strategy=0.1)  # type: ignore
    clf = ClassificationSystem(rf, ["amount"], [0.02, 0.98], [])
    cardsim = Cardsim()
    cards, terminals, transactions, is_fraud = cardsim.simulate(n_days=35)
    banksys = Banksys(clf, cards, terminals, transactions, is_fraud.tolist())


if __name__ == "__main__":
    main()
