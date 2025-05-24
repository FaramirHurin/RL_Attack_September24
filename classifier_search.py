import logging
import os
from datetime import timedelta, datetime
import orjson
from copy import deepcopy
import dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from parameters import ClassificationParameters, Parameters, PPOParameters, serialize_unknown, CardSimParameters
from banksys import Banksys, ClassificationSystem
import multiprocessing as mp

dotenv.load_dotenv()  # Load the "private" .env file
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def get_test_set(params: Parameters):
    cards, terminals, transactions = params.cardsim.get_simulation_data()
    banksys = Banksys(
        cards=cards,
        terminals=terminals,
        aggregation_windows=params.aggregation_windows,
        attackable_terminal_factor=params.terminal_fract,
        clf_params=params.clf_params,
    )
    train_x, train_y, test_x, test_y = banksys.fit(transactions)
    return banksys, train_x, train_y, test_x, test_y


def process(params: ClassificationParameters, banksys: Banksys, train_x, train_y, test_x, test_y):
    logging.info(
        f"Testing with trees={params.n_trees}, contamination={params.contamination}, balance_factor={params.balance_factor}, quantile={params.quantiles_values}"
    )
    clf = ClassificationSystem(banksys, params)
    # Fit the classifier
    clf.fit(train_x, train_y)
    # Predict & compute metrics
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
    logging.info(results)
    os.makedirs("results", exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.json")
    with open(os.path.join("results", filename), "wb") as f:
        f.write(orjson.dumps(results, default=serialize_unknown, option=orjson.OPT_SERIALIZE_NUMPY))


def cross_validate_classifier():
    params = Parameters(
        PPOParameters(), clf_params=ClassificationParameters(training_duration=timedelta(days=10)), cardsim=CardSimParameters(n_days=50)
    )
    banksys, train_x, train_y, test_x, test_y = get_test_set(params)

    # Define the parameters for the classification system
    trees_candidates = [100, 200]
    contamination_candidates = [0.005, 0.001]
    balance_factor_candidates = [0.1, 0.05]
    quantiles = [[0.005, 0.995], [0.01, 0.99], [0.05, 0.95]]

    results_list = []

    pool = mp.Pool(4)
    handles = []
    # Create a grid of parameters
    for trees in trees_candidates:
        for contamination in contamination_candidates:
            for balance_factor in balance_factor_candidates:
                for quantile in quantiles:
                    params.clf_params.n_trees = trees
                    params.clf_params.contamination = contamination
                    params.clf_params.balance_factor = balance_factor
                    params.clf_params.quantiles_values = quantile
                    handles.append(
                        pool.apply_async(
                            process,
                            args=(deepcopy(params.clf_params), banksys, train_x, train_y, test_x, test_y),
                        )
                    )
    for handle in handles:
        result = handle.get()
        results_list.append(result)

    print(results_list)


if __name__ == "__main__":
    # with open("params.json", "wb") as f:
    #    f.write(orjson.dumps(ClassificationParameters(), default=serialize_unknown))
    cross_validate_classifier()
