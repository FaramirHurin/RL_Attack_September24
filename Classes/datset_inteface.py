from typing import Literal
import polars as pl
from enum import IntEnum
import os
import json
import pandas as pd
import pickle

NOT_FEATURES_COLUMNS = ["Time", "Class", "label"]


class Label(IntEnum):
    GENUINE = 0
    FRAUDULENT = 1


class Dataset:
    def __init__(self, train, test):
        """
        This constructor should not be called directly.
        Use `from_real` or `from_synthetic` instead.
        """
        self.train_x = pl.DataFrame(train.drop("label", axis=1))
        self._test_x = pl.DataFrame(test.drop("label", axis=1))
        self.train_y = pl.Series(train["label"])
        self._test_y = pl.Series(test["label"])

    def save(self, path: str):
        transactions = self._all_transactions.with_columns(self._all_labels)
        os.makedirs(path, exist_ok=True)
        transactions.write_csv(f"{path}/transactions.csv")
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"test_ratio": 0.2}, f)

    @classmethod
    def load(cls, path: str):
        transactions = pl.read_csv(f"{path}/transactions.csv")
        labels = transactions["label"]
        transactions = transactions.drop("label")
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        return cls(transactions, labels, metadata["test_ratio"])

    @classmethod
    def from_kaggle(cls, balancenes: float, classifier: str):  # We need more features
        parent_dir = os.path.abspath(".")  # Move to the parent directory
        try:
            dir_path = os.path.join(parent_dir, "Classifiers", "Kaggle", "creditcard", str(balancenes))
            train = pd.read_csv(os.path.join(dir_path, "train_val.csv"))
            test = pd.read_csv(os.path.join(dir_path, "test.csv"))
            return cls(train, test)
        except FileNotFoundError:
            dir_path = os.path.join(parent_dir, "Classifiers", "Kaggle", str(balancenes))
            train = pd.read_csv(os.path.join(dir_path, "train_val.csv"))
            test = pd.read_csv(os.path.join(dir_path, "test.csv"))
            return cls(train, test)

    @classmethod
    def from_SkLearn(cls, balancenes: float, classifier: str, n_features: int, n_clusters: int, class_sep: float):
        # Construct the directory path
        parent_dir = os.path.abspath(".")  # Move to the parent directory

        # Construct the directory path relative to the parent directory
        dir_path = os.path.join(
            parent_dir, "Classifiers", "SkLearn", f"features_{n_features}_clusters_{n_clusters}_classsep_{class_sep}", str(balancenes)
        )  # , classifier
        train = pd.read_csv(os.path.join(dir_path, "train_val.csv"))
        test = pd.read_csv(os.path.join(dir_path, "test.csv"))
        return cls(train, test)

    @classmethod
    def from_Generator(cls, balancenes: float, classifier: str):  # We need more features
        # current_dir = os.getcwd()  # Get the current directory
        parent_dir = os.path.abspath(".")  # Move to the parent directory
        dir_path = os.path.join(parent_dir, "Classifiers", "Generator", "dataset", str(balancenes))
        train = pd.read_csv(os.path.join(dir_path, "train_val.csv"))
        test = pd.read_csv(os.path.join(dir_path, "test.csv"))
        return cls(train, test)

    def env_transactions(self, challenge: Literal["fraudulent", "genuine", "all"]):
        """The transactions that the environment challenges the agent with."""
        # This is actually the same implementation as mimicry_transactions and we always take the whole dataset.
        match challenge:
            case "all":
                return self._test_x
            case "fraudulent":
                return self._test_x.filter(self._test_y == Label.FRAUDULENT)
            case "genuine":
                return self._test_x.filter(self._test_y == Label.GENUINE)
        raise ValueError(f"Unknown transaction challenge method {challenge}")
        DEBUG = 0

    def mimicry_TR(
        self, transaction_type: Literal["fraudulent", "genuine", "all"], sample_size: Literal["1k", "2.5%", "5%", "100%"], c: list
    ):
        match transaction_type:
            case "all":
                transactions = self.train_x
            case "fraudulent":
                transactions = self.train_x.filter(self.train_y == Label.FRAUDULENT)
            case "genuine":
                transactions = self.train_x.filter(self.train_y == Label.GENUINE)
            case other:
                raise ValueError(f"Unknown sampling method {other}")
        match sample_size:
            case "100%":
                pass
            case "5%":
                transactions = transactions.sample(fraction=0.05)
            case "2.5%":
                transactions = transactions.sample(fraction=0.025)
            case "1k":
                transactions = transactions.head(1000)
            case other:
                raise ValueError(f"Unknown sample size {sample_size}")
        return transactions.select(transactions[c])

    @property
    def training_data(self):
        """
        The (x, y) data for the classifier to train on.

        Note: this is always the whole dataset.
        """
        return self.train_x.to_numpy(), self.train_y.to_numpy()


class DatasetLoader:
    def __init__(self, dataset_type, classifier, n_features_list=None, clusters_list=None, class_sep_list=None, balance_list=None):
        self.dataset_type = dataset_type
        self.classifier = classifier
        self.n_features_list = n_features_list or []
        self.clusters_list = clusters_list or []
        self.class_sep_list = class_sep_list or []
        self.balance_list = balance_list or []
        self.datasets = {}
        self.classifiers = {}
        self.parent_dir = os.path.abspath(".")  # Parent directory

    def load(self, use_sklearn_precombinations=False):
        """Main method to load datasets based on the dataset type."""
        if self.dataset_type == "SkLearn":
            self._load_sklearn_datasets(use_sklearn_precombinations)
        elif self.dataset_type == "Kaggle":
            self._load_kaggle_datasets()
        elif self.dataset_type == "Generator":
            self._load_generator_datasets()
        else:
            raise ValueError("Invalid dataset type. Must be 'SkLearn', 'Kaggle', or 'Generator'.")

        return self.datasets, self.classifiers

    def _load_sklearn_datasets(self, use_sklearn_precombinations):
        if use_sklearn_precombinations:
            combinations_sklearn = [
                {'n_features': 16, 'clusters': 1, 'class_sep': 8},
                {'n_features': 64, 'clusters': 8, 'class_sep': 8},
                {'n_features': 64, 'clusters': 16, 'class_sep': 8},
                {'n_features': 64, 'clusters': 16, 'class_sep': 1}
            ]
            for combination in combinations_sklearn:
                n_features = combination['n_features']
                n_clusters = combination['clusters']
                class_sep = combination['class_sep']
                for balance in self.balance_list:
                    key = self._create_key(n_features, n_clusters, class_sep, balance)
                    self.datasets[key] = Dataset.from_SkLearn(balance, self.classifier, n_features, n_clusters,
                                                              class_sep)
                    dir_path = self._build_directory_path("SkLearn", n_features, n_clusters, class_sep, balance)
                    self.classifiers[key] = self._load_classifier_from_directory(dir_path)
        else:
            for n_features in self.n_features_list:
                for n_clusters in self.clusters_list:
                    for class_sep in self.class_sep_list:
                        for balance in self.balance_list:
                            key = self._create_key(n_features, n_clusters, class_sep, balance)
                            self.datasets[key] = Dataset.from_SkLearn(balance, self.classifier, n_features, n_clusters, class_sep)
                            dir_path = self._build_directory_path("SkLearn", n_features, n_clusters, class_sep, balance)
                            self.classifiers[key] = self._load_classifier_from_directory(dir_path)

    def _load_kaggle_datasets(self):
        for balance in self.balance_list:
            key = self._create_key(balance)
            self.datasets[key] = Dataset.from_kaggle(balance, self.classifier)
            try:
                dir_path = os.path.join(self.parent_dir, "Classifiers", "Kaggle", "creditcard", str(balance), str(self.classifier))
                self.classifiers[key] = self._load_classifier_from_directory(dir_path)
            except:
                dir_path = os.path.join(self.parent_dir, "Classifiers", "Kaggle", str(balance), str(self.classifier))
                self.classifiers[key] = self._load_classifier_from_directory(dir_path)

    def _load_generator_datasets(self):
        for balance in self.balance_list:
            key = self._create_key(balance)
            self.datasets[key] = Dataset.from_Generator(balance, self.classifier)
            dir_path = os.path.join(self.parent_dir, "Classifiers", "Generator", "dataset", str(balance), str(self.classifier))
            self.classifiers[key] = self._load_classifier_from_directory(dir_path)

    def _load_classifier_from_directory(self, dir_path):
        files = os.listdir(dir_path)
        pickle_files = [file for file in files if file.endswith(".pickle") or file.endswith(".pkl")]
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in directory: {dir_path}")

        pickle_file_path = os.path.join(dir_path, pickle_files[0])
        with open(pickle_file_path, "rb") as file:
            return pickle.load(file)

    def _create_key(self, *args):
        return (self.dataset_type, self.classifier) + args

    def _build_directory_path(self, dataset_subtype, n_features, n_clusters, class_sep, balance):
        return os.path.join(
            self.parent_dir,
            "Classifiers",
            dataset_subtype,
            f"features_{n_features}_clusters_{n_clusters}_classsep_{class_sep}",
            str(balance),
            str(self.classifier),
        )
