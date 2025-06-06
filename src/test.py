from datetime import timedelta
from parameters import Parameters, CardSimParameters, ClassificationParameters
import logging
from banksys import Banksys
import numpy as np
import polars as pl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

if __name__ == "__main__":
    params = Parameters(
        cardsim=CardSimParameters(n_days=150, n_payers=10_000),
        clf_params=ClassificationParameters(),
    )
    # banksys = Banksys.load("trucmuche")
    banksys = params.create_banksys(use_cache=False, silent=False)
    # banksys.save("trucmuche")
    df_list = banksys.simulate_until(banksys.attack_start + timedelta(days=5))
    features = pl.concat(df_list)
    df = banksys._transactions_df.filter(pl.col("timestamp").is_between(banksys.attack_start, banksys.current_time))

    true_labels = df["is_fraud"].to_numpy()

    predicted = banksys.clf.predict(features, true_labels, banksys.current_time)
    details = banksys.clf.get_details()
    print(details)
    print(details.describe())
    assert features["amount"].equals(df["amount"])
    truth = df["is_fraud"].to_numpy().astype(np.bool)

    metrics = {}
    cm = confusion_matrix(truth, predicted)
    logging.info(f"{cm}")
    f1 = f1_score(truth, predicted)
    accuracy = accuracy_score(truth, predicted)
    precision = precision_score(truth, predicted)
    recall = recall_score(truth, predicted)
    metrics = {
        "confusion_matrix": cm.tolist(),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
