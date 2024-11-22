from Dataset.SkLearn_Dataset.DataSet import generate_SKLearn_Data
from Dataset.Kaggle_Dataset.generateKaggle import generate_kaggle_dataset
from Dataset.Generator_Dataset.generate_Generator_dataset import  generate_generator_dataset
from Classifiers.train_classifiers import fit_and_store_all_classifiers
from Classes import main_class
from Classes.main_class import run_all_experiments

GENERATE_DATASETS = False
TRAIN_CLASSIFIERS = False

if GENERATE_DATASETS:
    generate_SKLearn_Data(n_samples=50000, dimensions_list=[16, 32, 64],
                     clusters_list=[1, 8, 16], sep_classes_list=[0.5, 1, 2, 8])
    generate_kaggle_dataset()
    generate_generator_dataset()

if TRAIN_CLASSIFIERS:
    fit_and_store_all_classifiers()


dataset_types = ['Kaggle', 'SkLearn', 'Generator']  # Kaggle  Generator SkLearn
n_features_list = [16, 32, 64]
clusters_list = [1, 8, 16]
class_sep_list = [0.5, 2, 8]
balance_list = [ 0.1, 0.5]
classifier_names = [ 'DNN', 'RF']
min_max_quantile = 0.05
N_REPETITIONS = 20 # 20
N_STEPS = 3000  # 0
PROCESS_PER_GPU = 2
N_GPUS = 8  # 8
run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_names, min_max_quantile, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS)