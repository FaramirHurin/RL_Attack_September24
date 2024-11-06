import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import os
print(os.getcwd())


def optimally_fit_classifier(classifier_class, grid, train, test):
    X_train = train.drop(columns=['label'])
    y_train = train['label']
    X_test = test.drop(columns=['label'])
    y_test = test['label']

    classifier = classifier_class()
    grid_search = GridSearchCV(estimator=classifier, param_grid=grid, cv=4, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    report = classification_report(y_test, y_pred)

    return best_clf, grid_search.best_params_, report


def get_dataset_list(dataset_type: Literal["Generator", "Kaggle", "SkLearn"]) -> object:
    parent_dir = os.path.abspath('..')
    dictionary = {'Kaggle':'Kaggle_Dataset', 'Generator': 'Generator_Dataset', 'SkLearn': 'SkLearn_Dataset'}
    csv_folder_path = os.path.join(parent_dir, 'Dataset', dictionary[dataset_type])
    print(csv_folder_path)
    csv_dict = {}
    # Loop through all files in the folder
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_folder_path, file_name)
            df = pd.read_csv(file_path)
            name = file_name[:-4]
            csv_dict[name] = df
    return csv_dict


def preprocess_dataset(fraud_fraction, normalize, dataset):
    fraud_data = dataset[dataset['label'] == 1]
    non_fraud_data = dataset[dataset['label'] == 0]
    num_fraud = len(fraud_data)
    # Select the desired number of frauds based on fraud_fraction
    num_non_fraud = int(num_fraud * (1 - fraud_fraction) / fraud_fraction)

    # If the current number of non-fraud rows exceeds the desired amount, undersample non-fraud data
    if len(non_fraud_data) > num_non_fraud:
        non_fraud_sample = non_fraud_data.sample(n=num_non_fraud, random_state=42)
    else:
        non_fraud_sample = non_fraud_data

    # Concatenate fraud and non-fraud data and resample
    balanced_dataset = pd.concat([fraud_data, non_fraud_sample]).\
        sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_dataset = balanced_dataset

    # If normalize is True, normalize the other columns (except 'label')
    if normalize:
        normalizer = Normalizer()
        features = balanced_dataset.drop('label', axis=1)
        normalized_features = normalizer.fit_transform(features)
        balanced_dataset[features.columns] = normalized_features

    return balanced_dataset

def fit_and_store_classifiers(fraud_fractions, classifier_types, classifier_classes, grids, dataset_type,
                              normalize=True):
    datasets_dict = get_dataset_list(dataset_type)
    for dataset_key in datasets_dict.keys():
        dataset_global = datasets_dict[dataset_key]
        for fraud_fraction in fraud_fractions:
            dataset = preprocess_dataset(fraud_fraction, normalize, dataset_global)
            train_val, test = train_test_split(dataset)
            train, val = train_test_split(train_val)

            base_dir = os.path.join(os.getcwd() ,dataset_type, dataset_key, str(fraud_fraction))
            os.makedirs(base_dir, exist_ok=True)

            train_val_csv_path = os.path.join(base_dir, 'train_val.csv')
            test_csv_path = os.path.join(base_dir, 'test.csv')
            train_val.to_csv(train_val_csv_path, index=False)
            test.to_csv(test_csv_path, index=False)

            for classifier_type in classifier_types:
                classifier_class = classifier_classes[classifier_type]
                grid = grids[classifier_type]
                classifier_dir = os.path.join(base_dir, classifier_type)

                # Create directories if they do not exist
                os.makedirs(classifier_dir, exist_ok=True)
                # Save the trained classifier using pickle
                classifier_filename = f'{dataset_key}_classifier.pickle'
                classifier_path = os.path.join(classifier_dir, classifier_filename)
                print('Classifier Path')
                print(classifier_path)
                trained_classifier, best_params, report = optimally_fit_classifier(classifier_class, grid, train, val)
                print(report)
                with open(classifier_path, 'wb') as f:
                    pickle.dump(trained_classifier, f)



grids = \
    { 'RF':  {
            'n_estimators': [ 100, 150],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5, 10]
        },
    'BRF':  {
            'n_estimators': [100, 150],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'sampling_strategy': [0.1, 0.5],
            'bootstrap': [False],
            'replacement': [False]

        },
    'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
}


classifier_classes = {'RF': RandomForestClassifier}  # , 'BRF': BalancedRandomForestClassifier 'dnn': MLPClassifier, ,
classifier_types = ['RF'] #, 'BRF'

fraud_fractions = [ 0.1, 0.5] #0.002,
datasets_ = ['Kaggle', 'SkLearn']
fit_and_store_classifiers(fraud_fractions, classifier_types, classifier_classes, grids, 'Generator',
                          normalize=True)

