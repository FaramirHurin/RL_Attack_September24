

class Classification_system():
    classifier: ...
    rule_based_classifier: ...
    statistical_classifier: ...

    def __init__(self, classifierClass, features_for_quantiles:list, quantiles:list, rules):
        self.classifier = classifierClass()
        self.rule_based_classifier = RuleBasedClassifier(features_for_quantiles, quantiles)
        self.statistical_classifier = StatisticalClassifier(rules)

    def fit(self, X, y):
        self.classifier.fit(X, y)
        self.rule_based_classifier.fit(X)

    def predict(self, X):
        classification_prediction = self.classifier.predict(X)
        statistical_prediction = self.statistical_classifier.predict(X)
        rule_based_prediction = self.rule_based_classifier.predict(X)
        return classification_prediction or statistical_prediction or rule_based_prediction


class StatisticalClassifier():
    def __init__(self, considered_features: list[str], quantiles: list[float]):
        self.considered_features = considered_features
        self.quantiles = quantiles
        pass

    def fit(self, X):
        # Select the quantiles of all the considered_features in X
        self.quantiles_values = X[self.considered_features].quantile(self.quantiles)

    def predict(self, X):
        # Check if the value of each considered_feature in X is in the quantiles_values
        return X[self.considered_features].isin(self.quantiles_values).all(axis=1)



#TODO Implement rules logic. It could be a query freqency check, a value check, etc.
class RuleBasedClassifier():
    def __init__(self, rules: list[dict]):
        self.rules = rules
        pass


    def predict(self, X):
        for rule in self.rules:
            if rule['condition'](X):
                return rule['result']
        return False
