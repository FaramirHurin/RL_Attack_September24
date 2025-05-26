from parameters import Parameters, PPOParameters, ClassificationParameters, CardSimParameters
from datetime import timedelta
from banksys import Banksys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

if __name__ == "__main__":
    params = Parameters(
        agent=PPOParameters(),
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters(
            use_anomaly=False,
            n_trees=100,
            balance_factor=0.05,
            contamination=0.005,
            training_duration=timedelta(days=150),
            quantiles_features=("amount",),
            quantiles_values=(0.01, 1.0),
            rules={
                "max_trx_hour": 6,
                "max_trx_week": 40,
                "max_trx_day": 15,
            },
        ),
    )

    # system = params.create_pooled_env().system

    cards, terminals, transactions = params.cardsim.get_simulation_data()
    system = params.create_pooled_env().system
    banksys = Banksys(
        cards=cards,
        terminals=terminals,
        aggregation_windows=params.aggregation_windows,
        attackable_terminal_factor=params.terminal_fract,
        clf_params=params.clf_params,
    )
    train_x, train_y, test_x, test_y = banksys.fit(transactions)
    clf = banksys.clf
    predictions = clf.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    confusion = confusion_matrix(test_y, predictions)

    results = {
        "parameters": params,
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
        "confusion_matrix": confusion,
    }

    print(results)
