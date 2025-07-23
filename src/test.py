from parameters import Parameters, CardSimParameters, ClassificationParameters, PPOParameters, VAEParameters
from runner import Runner


if __name__ == "__main__":
    params = Parameters(
        cardsim=CardSimParameters(n_days=150, n_payers=10_000),
        clf_params=ClassificationParameters(use_anomaly=True),
        agent=VAEParameters.best_vae(),
    )
    runner = Runner(params)
    runner.run()
