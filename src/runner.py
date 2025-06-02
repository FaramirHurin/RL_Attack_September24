import logging
import os
from typing import Optional
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


class Runner:
    def __init__(self, params: Parameters, env: Optional[CardSimEnv] = None, quiet: bool = False, device: Optional[torch.device] = None):
        self.params = params
        self.episodes = dict[Card, Episode]()
        self.observations = dict[Card, Observation]()
        self.actions = dict[Card, np.ndarray]()
        self.states = dict[Card, State]()
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
        self.actions[new_card] = action
        self.observations[new_card] = obs
        self.states[new_card] = state
        self.hidden_states[new_card] = hx
        self.n_spawned += 1

    def cleanup_card(self, card: Card):
        del self.episodes[card]
        del self.observations[card]
        del self.actions[card]
        del self.states[card]
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
                card, step = self.env.step()
                total += step.reward.item()
                pbar.set_postfix(trx=step_num, refresh=False)
                pbar.set_description(
                    f"{self.env.t.date().isoformat()} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}"
                )
            except AttackPeriodExpired as e:
                logging.warning(f"Attack period expired: {e}")
                return episodes

            transition = Transition.from_step(self.observations[card], self.states[card], self.actions[card], step)
            try:
                self.agent.update_transition(transition, step_num, episode_num)
            except ValueError as e:
                logging.warning(f"Value error during simulation at step={step_num}, episode={episode_num}:\n{e}")
                return episodes
            if step_num > 2_500:
                print()

            current_episode = self.episodes[card]
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
                    logging.warning(f"Value error during simulation at step={step_num}, episode={episode_num}:\n{e}")
                    return episodes

                if self.n_spawned < self.params.n_episodes:
                    self.spawn_card_and_buffer_action()
            else:
                action, self.hidden_states[card] = self.agent.choose_action(step.obs.data, self.hidden_states[card])
                self.env.buffer_action(action, card)
        return episodes


def main_parallel():
    import multiprocessing as mp

    params = Parameters(
        agent=VAEParameters.best_vae(),  #   PPOParameters.best_rppo3(),
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters.paper_params(),
        seed_value=30,
        logdir="logs/VAElocal-paper/seed-30",
    )
    exp = Experiment.create(params)
    with mp.Pool(16) as pool:
        pool.map(run, exp.repeat(16))
    logging.info("All runs completed.")


def run(params: Parameters):
    runner = Runner(params)
    episodes = runner.run()
    Run.create(params, episodes)


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("logs.txt", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = Parameters(
        agent=PPOParameters.best_ppo(),
        # agent=VAEParameters.best_vae(),
        cardsim=CardSimParameters(),
        clf_params=ClassificationParameters(),
        logdir="logs/test",
        save=True,
    )
    exp = Experiment.create(params)
    run(params)
