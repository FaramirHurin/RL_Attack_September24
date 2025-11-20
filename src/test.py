from parameters import CardSimParameters
import logging


logging.basicConfig(level=logging.INFO)

for i in range(100):
    # params = CardSimParameters.paper_params(with_modification=False)
    params = CardSimParameters.paper_params(with_modification=False)
    params.get_simulation_data(cache_dir=f"cache/cardsim-{i}")
