import numpy as np


def mse(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = len(x)
    return np.sum((x-y)**2)/n




