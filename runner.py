import logging
import os
from typing import Optional
from environment import CardSimEnv, AttackPeriodExpired
import numpy as np
import orjson
from marlenv import Episode, Transition, Observation, State
import torch
from tqdm import tqdm
from banksys import Card
from parameters import Parameters, PPOParameters, CardSimParameters, ClassificationParameters
import dotenv


def save_episodes(episodes: list[Episode], directory: str):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


class Runner:
    def __init__(self, params: Parameters, env: Optional[CardSimEnv] = None, quiet: bool = False):
        self.params = params
        self.episodes = dict[Card, Episode]()
        self.observations = dict[Card, Observation]()
        self.actions = dict[Card, np.ndarray]()
        self.states = dict[Card, State]()
        self.hidden_states = dict[Card, Optional[torch.Tensor]]()
        if env is None:
            env = params.create_env()
        self.env = env
        self.agent = params.create_agent(self.env)
        self.quiet = quiet
        self.n_spawned = 0

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
        episodes = list[Episode]()
        step_num = 0
        episode_num = 0
        scores = list[float]()
        pbar = tqdm(total=self.params.n_episodes, desc="Training", disable=self.quiet)
        try:
            while episode_num < self.params.n_episodes:
                logging.debug(f"{self.env.t.isoformat()} - {step_num}")
                step_num += 1
                card, step = self.env.step()

                transition = Transition.from_step(self.observations[card], self.states[card], self.actions[card], step)
                self.agent.update_transition(transition, step_num, episode_num)

                current_episode = self.episodes[card]
                current_episode.add(transition)
                if current_episode.is_finished:
                    self.cleanup_card(card)
                    scores.append(current_episode.score[0])
                    episodes.append(current_episode)
                    pbar.update()
                    avg_score = np.mean(scores[-100:])
                    pbar.set_description(f"{self.env.t.date().isoformat()} avg score={avg_score:.2f}")
                    episode_num += 1
                    self.agent.update_episode(current_episode, step_num, self.n_spawned)
                    if self.n_spawned < self.params.n_episodes:
                        self.spawn_card_and_buffer_action()
                else:
                    action, self.hidden_states[card] = self.agent.choose_action(step.obs.data, self.hidden_states[card])
                    self.env.buffer_action(action, card)
        except AttackPeriodExpired as e:
            logging.warning(f"Attack period expired: {e}")
        except ValueError as e:
            logging.warning(f"Value error during simulation at step={step_num}, episode={episode_num}:\n{e}")
        return episodes


def main():
    params = Parameters(
        # agent=VAEParameters.best_vae(),
        agent=PPOParameters.best_ppo(),
        # agent=PPOParameters.best_rppo(),
        # agent=VAEParameters(),
        cardsim=CardSimParameters(n_days=365 * 2, n_payers=10_000),
        logdir="logs/rppo",
        card_pool_size=50,
        terminal_fract=0.1,
        seed_value=7,
        clf_params=ClassificationParameters.paper_params(),
        avg_card_block_delay_days=7,
        n_episodes=4000,
    )
    # params.clf_params.rules = {}
    runner = Runner(params)
    runner.run()

    # pool = mp.Pool(8)
    # pool.map(truc, params.repeat(32))
    # sleep(1)
    # pool.join()
    # pool.terminate()
    # pool.close()
    # for param in params.repeat(10):
    #    PoolRunner(param).run()


if __name__ == "__main__":
    dotenv.load_dotenv()  # Load the "private" .env file
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        # filename="logs.txt",
        # filemode="a",
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
