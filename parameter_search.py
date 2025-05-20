import multiprocessing as mp
from itertools import product
from multiprocessing.pool import AsyncResult
import os

from marlenv.utils.schedule import ConstantSchedule, LinearSchedule, Schedule

from banksys import Banksys
from minitest import init_environment, train, save_parameters, save_episodes
from parameters import CardSimParameters, Parameters, RPPOParameters


def schedule_to_str(schedule: Schedule):
    if isinstance(schedule, ConstantSchedule):
        return f"{schedule.value}"
    elif isinstance(schedule, LinearSchedule):
        return f"linear-{schedule.start_value}-{schedule.end_value}-{schedule.n_steps}"
    raise ValueError(f"Unknown schedule type: {type(schedule)}")


def param_to_logdir(params: Parameters):
    agent_params = params.agent
    if isinstance(agent_params, RPPOParameters):
        c1_str = schedule_to_str(agent_params.critic_c1)
        c2_str = schedule_to_str(agent_params.entropy_c2)
        return os.path.join(
            "logs",
            f"{params.agent_name}-{c1_str}-{c2_str}-{agent_params.n_epochs}-{agent_params.train_interval}-{len(params.rules) > 0}-{params.use_anomaly}",
            f"{params.seed}",
        )
    raise NotImplementedError()


def experiment(c1: float, c2: Schedule, n_epochs: int, train_interval: int, with_rules: bool, with_anomaly: bool, seed: int):
    params = Parameters(
        RPPOParameters(
            critic_c1=c1,
            entropy_c2=c2,
            n_epochs=n_epochs,
            train_interval=train_interval,
        ),
        use_anomaly=with_anomaly,
        seed=seed,
        cardsim=CardSimParameters(n_days=100, n_payers=20_000),
    )
    if not with_rules:
        params.rules = {}
    env = init_environment(params)
    logdir = param_to_logdir(params)
    os.makedirs(logdir, exist_ok=True)
    save_parameters(logdir, params)
    episodes = train(env, params)
    save_episodes(episodes, logdir)


def parameter_search_rppo():
    N_PARALLEL = 8
    C1 = [0.5, 0.1, 0.3]
    C2 = [
        Schedule.constant(0.01),
        Schedule.linear(0.1, 0.01, 10_000),
        Schedule.linear(0.2, 0.01, 10_000),
        Schedule.linear(0.2, 0.001, 20_000),
        Schedule.linear(0.5, 0.05, 20_000),
    ]
    N_EPOCHS = [16, 32, 64]
    TRAIN_INTERVAL = [32, 64, 128]
    WITH_RULES = [False, True]
    WITH_ANOMALY = [False, True]

    combinations = list(product(C1, C2, N_EPOCHS, TRAIN_INTERVAL, WITH_RULES, WITH_ANOMALY))
    p = Parameters(RPPOParameters(), cardsim=CardSimParameters(n_days=100, n_payers=20_000))
    try:
        Banksys.load(p.cardsim)
    except (FileNotFoundError, ValueError):
        p.create_banksys().save(p.cardsim)

    with mp.Pool(N_PARALLEL) as pool:
        handles = list[AsyncResult]()
        for c1, c2, n_epochs, train_interval, with_rules, with_anomaly in combinations:
            for seed in range(N_PARALLEL):
                handle = pool.apply_async(experiment, args=(c1, c2, n_epochs, train_interval, with_rules, with_anomaly, seed))
                handles.append(handle)
        for h in handles:
            h.wait()
        pool.join()


if __name__ == "__main__":
    parameter_search_rppo()
