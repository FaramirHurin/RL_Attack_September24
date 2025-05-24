import multiprocessing as mp
from typing import Literal

from marlenv.utils.schedule import Schedule

from runner import PoolRunner, save_episodes
import logging
from parameters import CardSimParameters, Parameters, PPOParameters


def experiment(params: Parameters):
    try:
        params.save()
        runner = PoolRunner(params)
        episodes = runner.run()
        save_episodes(episodes, params.logdir)
        del runner
        del episodes
    except Exception as e:
        logging.error(f"Error occurred while running experiment: {e}")


def parameter_search():
    ppo_types = list[tuple[Literal["episode", "transition"], bool]]([("episode", True), ("episode", False), ("transition", False)])
    for train_on, is_recurrent in ppo_types:
        for c1 in [0.5, 0.1, 0.3]:
            for c2 in [
                Schedule.constant(0.01),
                Schedule.linear(0.1, 0.01, 4_000),
                Schedule.linear(0.2, 0.01, 4_000),
                Schedule.linear(0.2, 0.001, 4_000),
                Schedule.linear(0.5, 0.05, 4_000),
            ]:
                for n_epochs in [16, 32, 64, 128]:
                    for train_interval in [32, 64, 96, 128]:
                        params = Parameters(
                            PPOParameters(
                                is_recurrent=is_recurrent,
                                train_on=train_on,
                                critic_c1=c1,
                                entropy_c2=c2,
                                n_epochs=n_epochs,
                                minibatch_size=train_interval // 2,
                                train_interval=train_interval,
                            ),
                            use_anomaly=True,
                            seed_value=0,
                            rules={},
                            cardsim=CardSimParameters(n_days=365 * 2 + 30 + 7, n_payers=20_000),
                            card_pool_size=50,
                            save=False,
                        )
                        for p in params.repeat(8):
                            yield p


if __name__ == "__main__":
    from time import sleep

    pool = mp.Pool(16)
    next(parameter_search()).create_pooled_env()
    pool.map(experiment, parameter_search())
    sleep(1)
    pool.join()
    pool.terminate()
    pool.close()
