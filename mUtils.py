from math import log, log10, log2

from scipy.stats import pearsonr
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error
from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Queue:
    def __init__(self, size):
        self.queue = []
        self.size = size

    def lock_bound(self):
        self.lockBound = len(self.queue)

    def enqueue(self, item):
        if len(self.queue) >= self.size:
            del self.queue[self.lockBound]
        self.queue.append(item)

    def __getitem__(self, item):
        return self.queue[item]

    def __len__(self):
        return len(self.queue)

    def dequeue(self):
        if len(self.queue) <= self.lockBound:
            return None
        return self.queue.pop(self.lockBound)

    def size(self):
        return len(self.queue)

    def to_list(self):
        return self.queue

    def clear(self):
        del self.queue[self.lockBound:]

# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic

# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic


def pearson_correlation_coef(estimator, X, y):
    y_rounded = y[:]
    y_hat = estimator.predict(X)
    r, p_value = pearsonr(y_hat, y_rounded)
    return r

def regression_accuracy_scorer(estimator, X, y):
    y_rounded = y[:]
    y_hat = estimator.predict(X)
    # r, p_value = pearsonr(y_hat, y_rounded)
    # r = (r+1)/2
    # rmse = mean_squared_error(y_rounded, y_hat, squared=False)
    # mse = mean_squared_error(y_rounded, y_hat)
    # evs = explained_variance_score(y_rounded, y_hat)
    r2Scores = r2_score(y_rounded, y_hat)
    r2Adj = 1 - (1 - r2Scores) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)
    # bic = calculate_bic(X.shape[0], mse, X.shape[1] + 1)

    # mse = mean_squared_error(y_rounded, y_hat)
    # r = jensenshannon(y_rounded, y_hat)
    # return ((evs + r) / 2 - rmse)
    # return (r*alpha + (1-alpha)*evs)/2 - rmse #worked fine when with -2*std
    # return evs*alpha + r2Adj*(1-alpha)
    # return r*alpha - (1-alpha)*rmse
    # return r
    # return 1-rmse
    # return alpha*r2Adj + (1-alpha)*evs
    # return 1 - mean_absolute_error(y_rounded, y_hat)
    return r2Adj, 0
