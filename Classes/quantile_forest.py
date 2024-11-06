from sklearn.metrics import  mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern,WhiteKernel
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABC, abstractmethod
from time import time
import logging
from sklearn_quantile import RandomForestQuantileRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import util
from marlenv import Transition


from agent import Agent
#TODO Modify it to work with fraud_env, i.e., to work with the interface Agent

class QuantileForest_Bandit(Agent): # (Agent)
    def __init__(self, action_min, action_max, kernel=None, reward_label:bool=False):
        self.candidates_number = min( action_min.shape[0] * action_min.shape[0] * 10, 10000)
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.dim = len(self.action_min)
        self.X = np.empty((0, self.dim))
        self.Y = []
        self.kernel = C(0.01, (1e-1, 5e1)) * RBF(1, (1e-2, 1e2))  #
        self.reward_label = reward_label
        self.model = None
        self.time = 0
        self.predicted_y = []
        self.chosen_candidates = []

    def _fit_learner(self):
        if self.reward_label:
            self.model = RandomForestClassifier(100)
        else:
            self.model = RandomForestQuantileRegressor(100, q = [0.9]) #, q = [0.99]
        self.model.fit(self.chosen_candidates, np.array(self.Y).ravel())

    def select_action(self):
        x_candidates = np.random.uniform(low = self.action_min, high = self.action_max,  size = (self.candidates_number, self.action_max.shape[0]))
        if self.model:
            y_candidates_means = self.model.predict(np.array(x_candidates)) #  #, y_candidates_variances  , return_std=True
            y_totals = y_candidates_means #+ 1.7 * y_candidates_variances
            chosen_index = np.argmax(y_totals)
            chosen_candidate = x_candidates[chosen_index]
            self.predicted_y.append( self.model.predict(chosen_candidate.reshape(1, -1))[0])
        else:
            chosen_index = np.random.choice(range(len(x_candidates)))
            chosen_candidate = x_candidates[chosen_index]
        self.chosen_candidates.append(chosen_candidate)
        return chosen_candidate


    def update(self, x, y):
        self.Y.append(y.flatten())
        if self.time % 200 == 199 and np.mean(self.Y[-50:]) < 0.9: #
            self._fit_learner()
        if self.time % 300 == 299:
            print(self.time, np.mean(self.Y[-50:]))
            print(mean_absolute_error(np.array(self.predicted_y[-50:]).ravel(), np.array(self.Y[-50:]).ravel()))
        self.time +=1


    def store(self, transition: Transition):
        return


    def to(self, device):
        return

