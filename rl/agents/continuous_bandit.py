import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import logging
from scipy.optimize import minimize

# Set up logging configuration
logging.basicConfig(level=logging.ERROR)
gpy_logger = logging.getLogger('GPy')
gpy_logger.setLevel(logging.ERROR)
if gpy_logger.hasHandlers():
    gpy_logger.handlers.clear()


# Safe transformation functions
def safe_expm1(x):
    x_clamped = np.clip(x, -700, 700)
    return np.expm1(x_clamped)


def stable_acquisition_function(mean, variance):
    # Handle numerical stability for acquisition function
    return safe_expm1(mean + 10 **7 * variance) #1.96

'''
class ContinuousBandit:
    def __init__(self, action_min, action_max, kernel=None):
        self.candidates_number =  action_min.shape[0] ** 2
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.dim = len(self.action_min)
        self.X = np.empty((0, self.dim))
        self.Y = np.empty((0, 1))
        self.kernel = C(2, (1e-3, 1e5)) * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e4))
        self.alpha_value = 2.0  # Increase this value to tolerate more noise

        self.model = None
        self.time = 0

    def fit_gp(self):
        if len(self.X) > 1:
            self.model =  GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha_value)
            self.model.fit(self.X, self.Y)

    def suggest_action(self):
        x_candidates = np.random.uniform(low = self.action_min, high = self.action_max,  size = (self.candidates_number, self.action_max.shape[0]))
        if self.model:
            y_candidates_means, y_candidates_variances = self.model.predict(np.array(x_candidates), return_std=True)
            y_totals = y_candidates_means + y_candidates_variances
            chosen_index = np.argmax(y_totals)
        else:
            chosen_index = np.random.choice(range(len(x_candidates)))
        chosen_candidate = x_candidates[chosen_index]
        return chosen_candidate

    def update(self, x, y):
        x = np.atleast_2d(x)
        y = y.flatten()
        self.X = np.vstack([self.X, x])
        self.Y = np.vstack([self.Y, y])
        if self.time % 50 == 0 and np.mean(self.Y[-100:]) < 0.9:
            self.fit_gp()
        self.time +=1

'''
