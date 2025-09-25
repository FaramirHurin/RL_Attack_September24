import logging
import os
from typing import Literal, Optional
from environment import CardSimEnv
from exceptions import AttackPeriodExpired
import numpy as np
from plots import Experiment, Run
from marlenv import Episode, Transition, Observation, State
import torch
from tqdm import tqdm
from banksys import Card
from parameters import Parameters, PPOParameters, CardSimParameters, ClassificationParameters, VAEParameters
import dotenv
from multiprocessing.pool import Pool, AsyncResult


class Runner:
    def __init__(self, params: Parameters, env: Optional[CardSimEnv] = None, quiet: bool = False, device: Optional[torch.device] = None):
        if device is None:
            device = params.get_device_by_seed()
        self.params = params
        self.episodes = dict[Card, Episode]()
        self.prev_obs = dict[Card, Observation]()
        self.prev_states = dict[Card, State]()
        self.hidden_states = dict[Card, Optional[torch.Tensor]]()
        if env is None:
            env = params.create_env()
        self.env = env
        self.agent = params.create_agent(self.env, device)
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

    def cleanup_card(self, card: Card):
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
    logging.info(f"Starting run with seed {p.seed_value}...")
    try:
        runner = Runner(p, quiet=True)
        logging.info(f"Running with seed {p.seed_value}...")
        episodes = runner.run()
        return Run.create(p, episodes)
    except Exception as e:
        logging.error(f"Run with seed {p.seed_value}: Error occurred while running experiment: {e}", exc_info=True)


def main_parallel(algorithm: Literal["ppo", "rppo", "vae"], use_anomaly: bool, n_jobs: int = 8, n_repetitions: int = 32) -> float:
    match algorithm:
        case "ppo":
            agent = PPOParameters.best_ppo(use_anomaly)
        case "rppo":
            agent = PPOParameters.best_rppo(use_anomaly)
        case "vae":
            agent = VAEParameters.best_vae(use_anomaly)
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    params = Parameters(
        agent,
        clf_params=ClassificationParameters.paper_params(use_anomaly),
        cardsim=CardSimParameters.paper_params(),
        save=False,
        n_episodes=6000,
        seed_value=32,
    )
    exp = Experiment.create(params)
    total = 0.0
    with Pool(n_jobs) as pool:
        handles = list[AsyncResult[Run | None]]()
        for p in exp.repeat(n_repetitions):
            logging.info(f"Submitting run with seed {p.seed_value}...")
            handles.append(pool.apply_async(run, (p,)))
        for h in handles:
            r = h.get()
            if r is None:
                logging.error(f"Run with seed {p.seed_value} failed.")
            else:
                total += r.total_amount
                logging.info(f"Run with seed {p.seed_value} completed with result {r.total_amount:.2f}")
    objective = total / n_repetitions
    logging.info(f"Avg objective: {objective}")
    return objective


def main(
    algorithm: Literal["vae", "ppo", "rppo"],
    n_repetitions: int,
    anomaly: bool,
    ulb_data: bool = False,
    initial_seed: int = 0,
):
    for seed in range(initial_seed, n_repetitions):
        if algorithm == "vae":
            agent = VAEParameters.best_vae(anomaly)
        elif algorithm == "rppo":
            agent = PPOParameters.best_rppo(anomaly)
        elif algorithm == "ppo":
            agent = PPOParameters.best_ppo(anomaly)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        if ulb_data:
            logdir = f"logs/ULB/exp-retrain/{anomaly}-anomaly/{algorithm}/seed-{seed}"
        else:
            logdir = f"logs/exp-retrain/{anomaly}-anomaly/{algorithm}/seed-{seed}"
        params = Parameters(
            # agent=PPOParameters.best_rppo(),
            agent=agent,
            cardsim=CardSimParameters.paper_params(),
            clf_params=ClassificationParameters.paper_params(anomaly),
            n_episodes=6000,
            seed_value=seed,
            logdir=logdir,
            save=True,
            ulb_data=ulb_data,
        )
        Experiment.create(params)
        run(params)


if __name__ == "__main__":
    # setup logging
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        for algo in ("ppo", "rppo", "vae"):
            for use_anomaly in (True, False):
                logging.info(f"Starting experiments for algorithm={algo}, use_anomaly={use_anomaly}")
                main_parallel(algo, use_anomaly, n_jobs=16, n_repetitions=16)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
