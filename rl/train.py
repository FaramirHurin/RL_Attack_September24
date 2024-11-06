from marlenv import Transition
import torch
import logging
import numpy as np
import util
from Classes.fraud_env import FraudEnv
from .agents import Agent
# from.agents.continuous_bandit import ContinuousBandit
from Classes.quantile_forest import QuantileForest_Bandit
from rl.agents import  PPO
import numpy as np
import logging

def train_agent(agent: PPO, env: FraudEnv, n_steps: int):
    print('N steps are ' + str(n_steps))
    logging.debug('We have this number of feasible features ' + str(agent.policy.n_actions))
    agent = agent.to(util.get_device())
    episode_num = 0
    scores = []
    time_step = 0
    # training loop
    try:
        while time_step < n_steps:
            obs = env.reset()
            score = 0
            done, truncated = False, False
            while not (done or truncated):
                time_step += 1
                # select action with policy
                action, action_logprob = agent.select_action(obs.data)
                obs_, reward, done, truncated, _ = env.step(action)

                # saving reward and is_terminals
                agent.store(Transition(obs, action, reward, done, {}, obs_, truncated, action_probs=action_logprob)) #, probs=action_logprob
                agent.update()
                obs = obs_
                score += reward
            scores.append(score)
            if episode_num % 50 == 0:
                logging.debug(f"Episode {episode_num} - AVG score: {np.mean(scores[-100:]):.3f}")
                print(f"Episode {episode_num} - AVG score: {np.mean(scores[-100:]):.3f}")

            episode_num += 1
    except ValueError as e:
        # This error can be raised if the policy network outputs NaNs
        message = str(e)
        if "MultivariateNormal" not in message:
            # If the error is not related to the MultivariateNormal distribution, we raise it
            raise e
        logging.error(message)
        print(message)
        raise e
        # In that case, we just pad the scores with 0s to match the DataFrame shape
        scores = scores + [0] * (n_steps - len(scores))
    return np.array(scores)


def train_bandit(agent: QuantileForest_Bandit, env: FraudEnv, n_steps: int):
    import logging

    # Print out all loggers and their levels
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            # print(f"Logger: {name}, Level: {logging.getLevelName(logger.level)}")
            logger.setLevel(logging.WARNING)
    logging.debug('We have this number of feasible features: ' + str(env.observation_shape[0]))
    episode_num = 0
    scores = []
    time_step = 0

    try:
        while time_step < n_steps:
            obs = env.reset()
            score = 0
            done, truncated = False, False
            while not (done or truncated):
                time_step += 1
                # Select action using ContinuousBandit
                action = agent.select_action()

                # Step in the environment
                obs_, reward, done, truncated, _ = env.step(action)

                # Update the bandit with the observed action and reward
                agent.update(x=action, y=np.array([[reward]]))

                obs = obs_
                score += reward

            scores.append(score)
            if episode_num % 50 == 0:
                logging.debug(f"Episode {episode_num} - AVG score: {np.mean(scores[-100:]):.3f}")

            episode_num += 1
    except ValueError as e:
        # This error can be raised if there's an issue with the GP model
        message = str(e)
        logging.error(message)
        # In case of an error, pad the scores with 0s to match the DataFrame shape
        scores = scores + [0] * (n_steps - len(scores))
    return np.array(scores)


# Example usage
if __name__ == "__main__":
    # Define your environment and bandit
    env = FraudEnv()  # Replace with your environment setup
    action_min = [0, 0]  # Example minimum bounds for each dimension
    action_max = [1, 1]  # Example maximum bounds for each dimension
    bandit = QuantileForest_Bandit(action_min, action_max)

    # Train the bandit agent
    n_steps = 1000  # Define the number of steps
    logs = dict[str, np.ndarray]()
    logs["bandit"] = train_bandit(bandit, env, n_steps)
