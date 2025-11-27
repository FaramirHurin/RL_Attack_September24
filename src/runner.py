import logging
import os
from multiprocessing.pool import AsyncResult, Pool
from typing import Literal, Optional
from datetime import datetime

import dotenv
import numpy as np
import torch
from marlenv import Episode, Observation, State, Transition
from tqdm import tqdm

from banksys import Payer
from environment import CardSimEnv
from exceptions import AttackPeriodExpired
from parameters import CardSimParameters, ClassificationParameters, Parameters, PPOParameters, VAEParameters, EnvParameters
from plots import Experiment, Run
import utils


class Runner:
    def __init__(
        self,
        params: Parameters,
        env: Optional[CardSimEnv] = None,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = utils.get_device_by_seed(params.seed)
        self.params = params.env_params
        self.episodes = dict[Payer, Episode]()
        self.prev_obs = dict[Payer, Observation]()
        self.prev_states = dict[Payer, State]()
        self.hidden_states = dict[Payer, Optional[torch.Tensor]]()
        if env is None:
            env = params.make_env()
        self.env = env
        self.agent = params.make_agent(self.env, device)
        self.quiet = quiet
        self.n_spawned = 0

    def spawn_card_and_buffer_action(self):
        """
        Spawn a new card and buffers an action for it.
        """
        new_card, obs, state = self.env.spawn_card()
        action, hx = self.agent.choose_action(obs.data, None)
        self.env.buffer_action(action, new_card)

        self.episodes[new_card] = Episode.new(obs, state, {"t_start": self.env.t, "card_id": new_card.id})
        self.prev_obs[new_card] = obs
        self.prev_states[new_card] = state
        self.hidden_states[new_card] = hx
        self.n_spawned += 1

    def cleanup_card(self, card: Payer):
        del self.episodes[card]
        del self.prev_obs[card]
        del self.prev_states[card]
        del self.hidden_states[card]

    def run(self):
        self.env.reset()
        for _ in range(self.params.card_pool_size):
            self.spawn_card_and_buffer_action()

        # Main loop
        episodes = list[Episode]()
        step_num, episode_num = 0, 0
        total, avg_score, avg_length = 0.0, 0.0, 0.0
        scores = list[float]()
        pbar = tqdm(total=self.params.n_episodes, disable=self.quiet, unit="episode")

        while episode_num < self.params.n_episodes:
            step_num += 1
            try:
                card, step, action = self.env.step()
            except AttackPeriodExpired as e:
                logging.warning(f"Attack period expired: {e}")
                return episodes

            transition = Transition.from_step(self.prev_obs[card], self.prev_states[card], action, step)

            # Update self.observations, states and actions,
            self.prev_obs[card] = step.obs
            self.prev_states[card] = step.state

            total += step.reward.item()
            pbar.set_postfix(trx=step_num, refresh=False)
            pbar.set_description(f"{self.env.isodate} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}")

            try:
                self.agent.update_transition(transition, step_num, episode_num)
            except ValueError as e:
                logging.warning(f"Value error during simulation at step={step_num}, episode={episode_num}:\n{e}")
                return episodes

            current_episode = self.episodes[card]
            # current_episode.is_finished = step.done or step.truncated
            current_episode.add(transition)
            if current_episode.is_finished:
                self.cleanup_card(card)
                scores.append(current_episode.score[0])
                episodes.append(current_episode)
                avg_score = np.mean(scores[-100:])
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                pbar.update()
                pbar.set_description(
                    f"{self.env.t.date().isoformat()} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}"
                )
                episode_num += 1
                try:
                    self.agent.update_episode(current_episode, step_num, self.n_spawned)
                except ValueError as e:
                    logging.warning(f"ValueError while updating the agent at step={step_num}, episode={episode_num}: {e}")
                    return episodes

                if self.n_spawned < self.params.n_episodes:
                    self.spawn_card_and_buffer_action()
            else:
                action, self.hidden_states[card] = self.agent.choose_action(step.obs.data, self.hidden_states[card])
                self.env.buffer_action(action, card)
        return episodes


def run(p: Parameters):
    logging.info(f"Starting run with seed {p.seed}...")
    try:
        runner = Runner(p, quiet=False)
        episodes = runner.run()
        return Run.create(p, episodes)
    except Exception as e:
        logging.error(f"Run with seed {p.seed}: Error occurred while running experiment: {e}", exc_info=True)


def run_parallel(exp: Experiment, n_jobs: int = 8, n_repetitions: int = 32):
    runs = list[Run]()
    with Pool(n_jobs) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p in exp.repeat(n_repetitions):
            logging.info(f"Submitting run with seed {p.seed}...")
            handles.append(pool.apply_async(run, (p,)))
        for h in handles:
            r = h.get()
            if r is not None:
                runs.append(r)
                logging.info(f"Run with seed {p.seed} completed with result {r.total_amount:.2f}")
    return runs


def main(
    algorithm: Literal["vae", "ppo", "rppo"],
    anomaly: bool,
    n_repetitions: int = 1,
    ulb_data: bool = False,
    initial_seed: int = 0,
    n_jobs: int = 1,
):
    if algorithm == "vae":
        agent = VAEParameters.best_vae(anomaly)
    elif algorithm == "rppo":
        agent = PPOParameters.best_rppo(anomaly)
    elif algorithm == "ppo":
        agent = PPOParameters.best_ppo(anomaly)
        agent.use_covariance_matrix = False
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(with_modification=False, ulb_data=ulb_data),
        clf_params=ClassificationParameters(use_anomaly=anomaly),
        env_params=EnvParameters(
            n_episodes=6000,
        ),
        seed=initial_seed,
    )
    exp = Experiment.create(params)
    if n_jobs == 1:
        return [run(p) for p in exp.repeat(n_repetitions)]
    return run_parallel(exp, n_jobs=n_jobs, n_repetitions=n_repetitions)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "info").upper()  # info
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        main(algorithm="ppo", anomaly=True, n_repetitions=1, n_jobs=1)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
