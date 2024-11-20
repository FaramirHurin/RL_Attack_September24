from sklearn.datasets import make_classification
from collections import namedtuple
import pandas as pd
import os

def generate_SKLearn_Data(n_samples, dimensions_list, clusters_list, sep_classes_list):
    Params = namedtuple('Params', ['n_features', 'n_clusters', 'class_sep'])
    for n_features in dimensions_list:
        for n_clusters in clusters_list:
            for class_sep in sep_classes_list:
                params = Params(n_features, n_clusters, class_sep)
                filename = f"features_{params.n_features}_clusters_{params.n_clusters}_classsep_{params.class_sep}.csv"

                n_repeated = 0
                n_informative = int(n_features * 3 / 4)
                n_redundant = n_features - n_informative
                X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                    n_informative=n_informative, n_redundant=n_redundant,n_repeated=n_repeated,
                                    n_clusters_per_class=n_clusters, class_sep=class_sep)
                df = pd.DataFrame(X)
                df['label'] = y

                parent_dir = os.path.abspath('.')
                path_to_save = os.path.join(parent_dir, 'Dataset', 'SkLearn_Dataset', filename)
                df.to_csv(path_to_save, index=False)
                print(filename)


print('Starting')
#generate_SKLearn_Data(n_samples=20000, dimensions_list=[16, 32, 64], clusters_list=[1, 8, 16], sep_classes_list=[0.5, 1, 2, 8])
DEBUG = 0
