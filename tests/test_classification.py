import matplotlib.pyplot as plt
from plots import Experiment
import os
from plots import Experiment

from parameters import Parameters, serialize_unknown, PPOParameters, CardSimParameters, ClassificationParameters


def test_performance():
    params = Parameters(
        # agent=PPOParameters.best_rppo(),
        agent=PPOParameters.best_ppo(),
        cardsim=CardSimParameters(),
        clf_params=ClassificationParameters(),
        n_episodes=3000,
        logdir="logs/test/vae/seed-2",
        save=True,
    )
    banksys = params.create_banksys(use_cache=True)
    banksys.fit()



