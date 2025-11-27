from experiment import Experiment
import os
import pytest


@pytest.mark.skip
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


@pytest.mark.skip
def test_print_amounts():
    logdir = "../src/logs/exp-final/vae"
    experiment = Experiment.load(logdir)
    experiment.runs["seed-1"].items

    # experiment.print_amounts()
