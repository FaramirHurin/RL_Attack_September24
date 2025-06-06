import matplotlib.pyplot as plt
from plots import Experiment
import os
from plots import Experiment

from parameters import Parameters, serialize_unknown


def test_experiment():
    logdirs = {
        "VAE": "../src/logs/exp-final/vae",
        "PPO": "../src/logs/exp-final/ppo",
        "R-PPO": "../src/logs/exp-final/rppo",
    }
    experiments = []
    for name, logdir in logdirs.items():
        assert os.path.exists(logdir), f"Log directory {logdir} does not exist"
        if os.path.exists(logdir):
            experiment = Experiment.load(logdir)
            assert len(experiment.runs) > 0, f"No runs found in {logdir}"
            experiments.append(experiment)
    assert len(experiments) > 0, "No experiments loaded"


def test_print_amounts():
    logdir = "../src/logs/exp-final/vae"
    experiment = Experiment.load(logdir)
    experiment.runs['seed-1'].items

    experiment.print_amounts()