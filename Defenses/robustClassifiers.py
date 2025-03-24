import numpy as np
import polars as pl

class EnsembleAlgorithm():
    def __init__(self, classifier_type, features, forbidden_features: list[str],
                 n_estimator_normal, n_estimator_safe):
        self.features = features
        self.safe_classifiers = [classifier_type() for _ in range(n_estimator_safe)]
        self.complete_classifiers = [classifier_type() for _ in range(n_estimator_normal)]
        self.forbidden_features = forbidden_features

    def fit(self, X, y):
        safe_X = X.drop(self.forbidden_features)
        for classifier in self.safe_classifiers:
            classifier.fit(safe_X, y)
        for classifier in self.complete_classifiers:
            classifier.fit(X, y)

    def predict(self, safe_X, X):
        # Calculate probabilities for safe classifiers
        safe_predictions_proba = np.array(
            [classifier.predict_proba(safe_X)[:, 1] for classifier in self.safe_classifiers])

        # Calculate probabilities for complete classifiers
        complete_predictions_proba = np.array(
            [classifier.predict_proba(X)[:, 1] for classifier in self.complete_classifiers])

        # Compute the weighted sum of mean probabilities for each prediction
        predictions = (len(self.complete_classifiers) * np.mean(complete_predictions_proba, axis=0) +
                       len(self.safe_classifiers) * np.mean(safe_predictions_proba, axis=0)) / len(self.safe_classifiers + self.complete_classifiers)

        # Convert predictions to integer values
        return np.round(predictions).astype(bool).tolist()



