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


def main_parallel(algorithm: str):
    # import multiprocessing as mp

    if algorithm == "vae":
        agent = VAEParameters.best_vae()
        device = torch.device("cuda:0")
    elif algorithm == "rppo":
        agent = PPOParameters.best_rppo()
        device = torch.device("cuda:1")
    elif algorithm == "ppo":
        agent = PPOParameters.best_ppo()
        device = torch.device("cuda:2")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    params = Parameters(
        agent=agent,
        cardsim=CardSimParameters.paper_params(),
        clf_params=ClassificationParameters.paper_params(True),
        seed_value=0,
    )
    # Make sure the simulation data is created and the banksys is trained before running everything in parallel
    params.create_env()
    exp = Experiment.create(params)
    for params in exp.repeat(30):
        run(params, device)
    # with mp.Pool(1) as pool:
    #     pool.map(run, exp.repeat(30))
    logging.info("All runs completed.")


def run(params: Parameters, device: Optional[torch.device] = None, quiet: bool = True):
    logging.info(f"Running seed {params.seed_value} with agent {params.agent_name} in {params.logdir}")
    params.save()
    runner = Runner(params, quiet=quiet, device=device)
    episodes = runner.run()
    Run.create(params, episodes)


def main(n_repetitions: int, anomaly: bool, ulb_data: bool = False, quiet: bool = True):
    for seed in range(0, n_repetitions):
        for algorithm in ("vae", "ppo", "rppo"):
            if algorithm == "vae":
                agent = VAEParameters.best_vae(anomaly)
            elif algorithm == "rppo":
                agent = PPOParameters.best_rppo()
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
                n_episodes=4000,
                seed_value=seed,
                logdir=logdir,
                save=True,
                ulb_data=ulb_data,
            )
            Experiment.create(params)
            run(params, quiet=quiet)


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
        # main_parallel("ppo")
        # main_parallel("rppo")
        # main_parallel("vae")
        main(n_repetitions=1, anomaly=True, ulb_data=False, quiet=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise e
