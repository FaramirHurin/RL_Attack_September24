from parameters import Parameters, CardSimParameters


for seed in range(100):
    for modification in (True, False):
        params = Parameters(
            cardsim=CardSimParameters.paper_params(with_modification=modification),
            seed=seed,
        )
        params.make_env()
