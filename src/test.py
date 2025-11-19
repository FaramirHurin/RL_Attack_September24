from parameters import CardSimParameters
import logging


logging.basicConfig(level=logging.INFO)

for i in range(100):
    # params = CardSimParameters.paper_params(with_modification=False)
    params = CardSimParameters(n_days=50, n_payers=1_000)
    params.get_simulation_data(cache_dir=f"cache/cardsim-{i}")
