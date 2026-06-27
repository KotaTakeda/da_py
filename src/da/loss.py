import numpy as np


def loss_rms(X, Y):
    return norm_rms(X - Y)


def norm_rms(X):
    return np.linalg.norm(X, axis=-1) / np.sqrt(X.shape[-1])


def loss_sup(X, Y):
    return norm_sup(X - Y)


def norm_sup(X):
    return np.max(np.abs(X), axis=-1)
