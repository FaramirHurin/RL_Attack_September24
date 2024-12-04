from sklearn.datasets import make_classification
from collections import namedtuple
import pandas as pd
import os
from multiprocessing.pool import Pool, AsyncResult


def generate_SKLearn_Data(n_samples, dimensions_list, clusters_list, sep_classes_list):
    Params = namedtuple("Params", ["n_features", "n_clusters", "class_sep"])
    handles = list[tuple[AsyncResult, str]]()
    with Pool(8) as pool:
        for n_features in dimensions_list:
            for n_clusters in clusters_list:
                for class_sep in sep_classes_list:
                    params = Params(n_features, n_clusters, class_sep)
                    filename = f"features_{params.n_features}_clusters_{params.n_clusters}_classsep_{params.class_sep}.csv"

                    n_repeated = 0
                    n_informative = int(n_features * 3 / 4)
                    n_redundant = n_features - n_informative
                    handle = pool.apply_async(
                        make_classification,
                        kwds=dict(
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_informative,
                            n_redundant=n_redundant,
                            n_repeated=n_repeated,
                            n_clusters_per_class=n_clusters,
                            class_sep=class_sep,
                        ),
                    )
                    handles.append((handle, filename))
                    # X, y = make_classification(
                    #     n_samples=n_samples,
                    #     n_features=n_features,
                    #     n_informative=n_informative,
                    #     n_redundant=n_redundant,
                    #     n_repeated=n_repeated,
                    #     n_clusters_per_class=n_clusters,
                    #     class_sep=class_sep,
                    # )
                    # df = pd.DataFrame(X)
                    # df["label"] = y

                    # parent_dir = os.path.abspath(".")
                    # path_to_save = os.path.join(parent_dir, "Dataset", "SkLearn_Dataset", filename)
                    # df.to_csv(path_to_save, index=False)
                    # print(filename)
        for handle, filename in handles:
            X, y = handle.get()
            print(filename, "is done")
            df = pd.DataFrame(X)
            df["label"] = y
            parent_dir = os.path.abspath(".")
            path_to_save = os.path.join(parent_dir, "Dataset", "SkLearn_Dataset", filename)
            df.to_csv(path_to_save, index=False)
