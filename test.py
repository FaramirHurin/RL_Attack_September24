from parameters import Parameters, PPOParameters
from runner import run
from marlenv.utils import Schedule


if __name__ == "__main__":
    params = Parameters(
        PPOParameters(entropy_c2=0.025, train_interval=64, minibatch_size=32, is_recurrent=True, train_on="episode"),
        seed_value=1,
        logdir="logs/test",
    )
    run(params)
