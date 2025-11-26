import logging
from parameters import Parameters, CardSimParameters, PPOParameters, ClassificationParameters

logging.basicConfig(level=logging.INFO)

params = Parameters(
    PPOParameters(),
    CardSimParameters(),
    ClassificationParameters(),
    regenerate_cache=True,
)

env = params.make_env()
