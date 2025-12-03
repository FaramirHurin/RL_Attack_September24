import logging
from typing import Optional

import numpy as np
import torch
from marlenv import Episode, Observation, State, Transition
from tqdm import tqdm

from banksys import Payer
from exceptions import AttackPeriodExpired
from parameters import Parameters
import utils


class Runner:
    def __init__(
        self,
        params: Parameters,
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
        self.env = params.make_env()
        self.agent = params.make_agent(self.env, device)
        self.quiet = quiet

    def spawn_payer_and_buffer_action(self):
        """
        Spawn a new payer and buffers an action for it.
        """
        new_payer, obs, state = self.env.spawn_payer()
        action, hx = self.agent.choose_action(obs.data, None)
        self.env.buffer_action(action, new_payer)

        self.episodes[new_payer] = Episode.new(obs, state, {"t_start": self.env.t, "payer_id": new_payer.id})
        self.prev_obs[new_payer] = obs
        self.prev_states[new_payer] = state
        self.hidden_states[new_payer] = hx

    def cleanup_payer(self, payer: Payer):
        del self.episodes[payer]
        del self.prev_obs[payer]
        del self.prev_states[payer]
        del self.hidden_states[payer]

    def run(self):
        logging.info(f"Attack starting from {self.env.isodate}")
        for _ in range(self.params.pool_size):
            self.spawn_payer_and_buffer_action()
        n_spawned = self.params.pool_size

        # Main loop
        episodes = list[Episode]()
        step_num, episode_num = 0, 0
        total, avg_score, avg_length = 0.0, 0.0, 0.0
        scores = list[float]()
        pbar = tqdm(
            total=self.params.n_episodes,
            disable=self.quiet,
            unit="episode",
            desc=f"{self.env.isodate} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}",
        )

        while episode_num < self.params.n_episodes:
            step_num += 1
            try:
                payer, step, action = self.env.step()
            except AttackPeriodExpired as e:
                logging.warning(f"Attack period expired: {e}")
                return episodes

            transition = Transition.from_step(self.prev_obs[payer], self.prev_states[payer], action, step)

            # Update self.observations, states and actions,
            self.prev_obs[payer] = step.obs
            self.prev_states[payer] = step.state

            total += step.reward.item()
            pbar.set_postfix(trx=step_num, refresh=False)
            pbar.set_description(f"{self.env.isodate} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}")

            try:
                self.agent.update_transition(transition, step_num, episode_num, self.env.elapsed_seconds)
            except ValueError as e:
                logging.warning(f"Value error during simulation at step={step_num}, episode={episode_num}:\n{e}")
                return episodes

            current_episode = self.episodes[payer]
            # current_episode.is_finished = step.done or step.truncated
            current_episode.add(transition)
            if current_episode.is_finished:
                self.cleanup_payer(payer)
                scores.append(current_episode.score[0])
                episodes.append(current_episode)
                avg_score = np.mean(scores[-100:])
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                pbar.update()
                pbar.set_description(f"{self.env.isodate} avg score={avg_score:.2f} - len-avg={avg_length:.2f} - total={total:.2f}")
                episode_num += 1
                try:
                    self.agent.update_episode(current_episode, step_num, n_spawned, self.env.elapsed_seconds)
                except ValueError as e:
                    logging.warning(f"ValueError while updating the agent at step={step_num}, episode={episode_num}: {e}")
                    return episodes

                if n_spawned < self.params.n_episodes:
                    self.spawn_payer_and_buffer_action()
                    n_spawned += 1
            else:
                action, self.hidden_states[payer] = self.agent.choose_action(step.obs.data, self.hidden_states[payer])
                self.env.buffer_action(action, payer)
        return episodes
