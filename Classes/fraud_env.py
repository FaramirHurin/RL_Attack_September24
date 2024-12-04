from typing import Literal
import numpy as np
import polars as pl
from marlenv import MARLEnv, ContinuousActionSpace, Observation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class FraudEnv(MARLEnv):
    """
    Environment where the agent has to create fraudulent transactions that are not detected as such by the classifier.
    """

    def __init__(
        self,
        transactions: pl.DataFrame,
        k: list,
        u: list,
        c: list,
        classifier: RandomForestClassifier | MLPClassifier,
        reward_type: Literal["probability", "label"],
        min_values: np.ndarray,
        max_values: np.ndarray,
    ):
        """
        Args:
            - transactions: The transactions that the agent will be working on.
            - k: Number of known features.
            - u: Number of unknown features.
            - classifier: The classifier that will be used to determine the reward.
            - reward_type: The type of reward to use. Either "probability" or "label".
        """
        self.transactions = transactions
        self.known_features = k
        self.unknown_features = u
        self.actions = c
        self.class_reward = reward_type == "label"
        self.min_values = min_values
        self.max_values = max_values
        self.state = 0
        self.classifier = classifier
        # print(self.classifier.feature_importances_)
        super().__init__(
            action_space=ContinuousActionSpace(
                n_agents=1, low=[-1] * len(self.actions), high=[1] * len(self.actions), action_names=self.actions
            ),
            observation_shape=(max(1, len(self.known_features)),),
            state_shape=(1,),
        )

    def get_observation(self) -> Observation:
        if len(self.known_features) == 0:
            obs_data = np.random.random((1,)).astype(np.float32)
        else:
            obs_data = self.transactions[self.state][self.known_features].to_numpy().flatten().astype(np.float32)
        return Observation(
            data=obs_data,
            available_actions=self.available_actions()[0],
            state=np.array([self.state], dtype=np.float32),
        )

    def step(self, actions):
        transaction = self.transactions[self.state]
        for col, value in zip(self.actions, actions):
            transaction[0, col] = value  # .to('cpu').numpy()
        transaction = transaction.to_numpy()
        if self.class_reward:
            label = self.classifier.predict(transaction)[0]
            reward = 1.0 - label  # 1 if the transaction is classified as legit, 0 otherwise
        else:
            legit_proba = self.classifier.predict_proba(transaction)[0, 0]
            reward = legit_proba
        if np.any(transaction < self.min_values) or np.any(transaction > self.max_values):
            self.max_values_reward = 0
        self.state += 1
        if self.state == self.transactions.shape[0]:
            self.state = 0
        return self.get_observation(), reward, True, False, {}

    def get_batch(self, batch_size: int):
        all_indices = np.arange(len(self.transactions))
        indices = np.random.choice(all_indices, batch_size, replace=True)
        return self.transactions[indices]

    def reset(self):
        return self.get_observation()

    def get_state(self):
        return np.array([self.state])

    def render(self, mode):
        print(self.transactions[self.state])
