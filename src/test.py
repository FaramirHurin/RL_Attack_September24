from parameters import Parameters, CardSimParameters, ClassificationParameters, PPOParameters, VAEParameters
from runner import Runner


if __name__ == "__main__":
    agent = VAEParameters.best_vae()
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters.paper_params(True),
        seed_value=0,
    )
    print(hash(params))
    params.card_pool_size += 1
    print(hash(params))
