import typed_argparse as tap
from Dataset.SkLearn_Dataset.DataSet import generate_SKLearn_Data
from Dataset.Kaggle_Dataset.generateKaggle import generate_kaggle_dataset
from Dataset.Generator_Dataset.generate_Generator_dataset import generate_generator_dataset
from Classifiers.train_classifiers import fit_and_store_all_classifiers
from Classes.main_class import run_experiments
import os

GENERATE_DATASETS = False
TRAIN_CLASSIFIERS = False

dataset_types = ["Generator", "Kaggle", "SkLearn"]  # Kaggle  Generator SkLearn
n_features_list = [16, 32, 64]
clusters_list = [1, 8, 16]
class_sep_list = [0.5, 1, 2, 8]
balance_list = [0.1, 0.5]
classifier_names = ["DNN", "RF"]
min_max_quantile = 0.05
N_REPETITIONS = 2  # Number of repetitions per individual worker
N_STEPS = 4_000  # 0


class Args(tap.TypedArgs):
    run_num: int = tap.arg(help="Number of the run")
    logdir: str = tap.arg(help="Directory to store the logs")

    @property
    def device_name(self) -> str:
        import torch

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return "cpu"
        return f"cuda:{self.run_num % n_gpus}"

    @property
    def seed(self) -> int:
        return self.run_num


def main(args: Args):
    if GENERATE_DATASETS:
        generate_SKLearn_Data(
            n_samples=50000,
            dimensions_list=n_features_list,
            clusters_list=clusters_list,
            sep_classes_list=class_sep_list,
        )
        generate_kaggle_dataset()
        generate_generator_dataset()

    if TRAIN_CLASSIFIERS:
        fit_and_store_all_classifiers()

    if not args.logdir.startswith("logs"):
        args.logdir = os.path.join("logs", args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    run_experiments(
        logdir=args.logdir,
        dataset_types=dataset_types,
        n_features_list=n_features_list,
        clusters_list=clusters_list,
        class_sep_list=class_sep_list,
        balance_list=balance_list,
        classifier_names=classifier_names,
        min_max_quantile=min_max_quantile,
        first_experiment_num=args.run_num * N_REPETITIONS,
        n_experiments=N_REPETITIONS,
        n_steps=N_STEPS,
        device_name=args.device_name,
    )


if __name__ == "__main__":
    tap.Parser(Args).bind(main).run()
