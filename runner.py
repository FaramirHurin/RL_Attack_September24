from tqdm import tqdm
from parameters import Parameters, RPPOParameters, PPOParameters
from marlenv import Episode, Transition
from banksys import Transaction
import numpy as np
import logging
import os
import orjson


def save_episodes(episodes: list[Episode], directory: str):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, "episodes.json")
    with open(filename, "wb") as f:
        f.write(orjson.dumps(episodes, option=orjson.OPT_SERIALIZE_NUMPY))


def run(params: Parameters):
    env = params.create_env()
    agent = params.create_agent(env)

    scores = list[float]()
    episodes = list[Episode]()
    pbar = tqdm(range(params.n_episodes), desc="Training")
    step_num = 0
    for e in pbar:
        obs, state = env.reset()
        episode = Episode.new(obs, state, {"t_start": env.t_start, "card_id": env.current_card.id})
        transactions = list[Transaction]()
        terminals = list[int]()
        while not episode.is_finished:
            step_num += 1
            action = agent.choose_action(obs.data)
            step, trx = env.step(action)
            if trx is not None:
                terminals.append(trx.terminal_id)
                transactions.append(trx)
            t = Transition.from_step(obs, state, action, step)
            agent.update(t, step_num)
            episode.add(t)
            obs, state = step.obs, step.state
        episode.add_metrics({"t_end": env.t.isoformat(), "terminals": terminals})
        scores.append(episode.score[0])
        episodes.append(episode)
        # Update tqdm description with average score
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            pbar.set_description(f"Avg score (last 100): {avg_score:.2f}")
            if e % 100 == 0:
                avg_length = np.mean([len(ep) for ep in episodes[-100:]])
                logging.info(f"{e}: avg score={avg_score:.2f}, avg_length={avg_length:.2f} (last 100)")
        else:
            pbar.set_description("Avg score (last 100): N/A")
    pbar.close()
    save_episodes(episodes, params.logdir)


if __name__ == "__main__":
    params = Parameters(PPOParameters(entropy_c2=0.025))
    for p in params.repeat(10):
        run(p)
