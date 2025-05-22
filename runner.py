import logging
import os
from typing import Optional

import numpy as np
import orjson
from marlenv import Episode, Transition, Observation, State
import torch
from tqdm import tqdm

from banksys import Transaction, Card
from parameters import Parameters, PPOParameters, VAEParameters, CardSimParameters
import dotenv

dotenv.load_dotenv()  # Load the "private" .env file
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def save_episodes(episodes: list[Episode], directory: str):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


class PoolRunner:
    def __init__(self, params: Parameters):
        self.params = params
        self.episodes = dict[Card, Episode]()
        self.observations = dict[Card, Observation]()
        self.actions = dict[Card, np.ndarray]()
        self.states = dict[Card, State]()
        self.hidden_states = dict[Card, Optional[torch.Tensor]]()
        self.env = params.create_pooled_env()
        self.agent = params.create_agent(self.env)
        self.n_spawned = 0

    def reset(self):
        self.episodes.clear()
        self.observations.clear()
        self.actions.clear()
        self.states.clear()
        self.hidden_states.clear()

    def spawn_card_and_buffer_action(self):
        """
        Spawn a new card and buffers an action for it.
        """
        new_card, obs, state = self.env.spawn_card()
        action, hx = self.agent.choose_action(obs.data, None)
        self.env.buffer_action(action, new_card)

        self.episodes[new_card] = Episode.new(obs, state, {"t_start": self.env.t_start, "card_id": new_card.id})
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
        step_num = 0
        episode_num = 0
        scores = list[float]()
        pbar = tqdm(total=self.params.n_episodes, desc="Training")
        while episode_num < self.params.n_episodes:
            logging.debug(f"{self.env.t.isoformat()} - {step_num}")
            step_num += 1
            card, step = self.env.step()

            transition = Transition.from_step(self.observations[card], self.states[card], self.actions[card], step)
            self.agent.update_transition(transition, step_num)

            current_episode = self.episodes[card]
            current_episode.add(transition)
            if current_episode.is_finished:
                self.cleanup_card(card)
                scores.append(current_episode.score[0])
                pbar.update()
                pbar.set_description(f"{self.env.t.date().isoformat()} avg score={np.mean(scores[-100:]):.2f}")
                episode_num += 1
                self.agent.update_episode(current_episode, step_num, self.n_spawned)
                if self.n_spawned < self.params.n_episodes:
                    self.spawn_card_and_buffer_action()
            else:
                action, self.hidden_states[card] = self.agent.choose_action(step.obs.data, self.hidden_states[card])
                self.env.buffer_action(action, card)


def run(params: Parameters):
    env = params.create_env()
    agent = params.create_agent(env)

    scores = list[float]()
    episodes = list[Episode]()
    pbar = tqdm(range(params.n_episodes), desc="Training")
    step_num = 0
    for e in pbar:
        obs, state = env.reset()
        hx = None
        episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
        transactions = list[Transaction]()
        terminals = list[int]()
        while not episode.is_finished:
            step_num += 1
            action, hx = agent.choose_action(obs.data, hx)
            step, trx = env.step(action)
            if trx is not None:
                terminals.append(trx.terminal_id)
                transactions.append(trx)
            t = Transition.from_step(obs, state, action, step)
            agent.update_transition(t, step_num)
            episode.add(t)
            obs, state = step.obs, step.state
        episode.add_metrics({"t_end": env.t.isoformat(), "terminals": terminals})
        scores.append(episode.score[0])
        episodes.append(episode)
        # Update tqdm description with average score
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            pbar.set_description(f"avg score={avg_score:.2f}")
            if e % 100 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                logging.info(f"{e}: avg score={avg_score:.2f}, avg_length={avg_length:.2f} (last 100)")
        else:
            pbar.set_description("Avg score (last 100): N/A")
    pbar.close()
    save_episodes(episodes, params.logdir)


def main():
    params = Parameters(
        PPOParameters(is_recurrent=False, train_on="transition"),
        cardsim=CardSimParameters(n_days=365 * 5),
        logdir="logs/test",
        card_pool_size=50,
        terminal_fract=0.1,
    )
    run(params)
    # PoolRunner(params).run()


if __name__ == "__main__":
    main()
