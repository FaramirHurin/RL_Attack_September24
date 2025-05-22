from parameters import Parameters, PPOParameters
from runner import run


if __name__ == "__main__":
    params = Parameters(
        PPOParameters(entropy_c2=0.025, train_interval=64, minibatch_size=32, is_recurrent=True, train_on="episode"),
        logdir="logs/test",
        terminal_fract=0.1,
    )
    run(params)
