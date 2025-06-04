import matplotlib.pyplot as plt
from plots import Experiment
import os
from plots import Experiment

from parameters import Parameters, serialize_unknown


def test_experiment():
    logdirs = {
        "VAE": "../src/logs/test/vae",
        "PPO": "../src/logs/test/ppo",
        "R-PPO": "../src/logs/test/rppo",
    }
    experiments = []
    for name, logdir in logdirs.items():
        assert os.path.exists(logdir), f"Log directory {logdir} does not exist"
        if os.path.exists(logdir):
            experiment = Experiment.load(logdir)
            assert len(experiment.runs) > 0, f"No runs found in {logdir}"
            experiments.append(experiment)