from parameters import Parameters, CardSimParameters, ClassificationParameters, PPOParameters, VAEParameters
from runner import Runner
import hashlib


if __name__ == "__main__":
    agent = VAEParameters.best_vae()
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters.paper_params(True),
        seed_value=0,
    )
    print(params.banksys_dir)
    params.card_pool_size += 1
    print(params.banksys_dir)
    params.cardsim.n_days += 5
    print(params.banksys_dir)
    params.clf_params.use_anomaly = False
    print(params.banksys_dir)
