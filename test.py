from parameters import Parameters, RPPOParameters
from runner import run


if __name__ == "__main__":
    params = Parameters(RPPOParameters(entropy_c2=0.1))
    run(params)
