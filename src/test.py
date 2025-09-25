from banksys import Banksys
from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters


if __name__ == "__main__":
    params = Parameters(
        agent=PPOParameters.best_ppo(True),
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters.paper_params(True),
        n_episodes=6000,
    )
    banksys = params.create_banksys(use_cache=True)
    params.create_env()
