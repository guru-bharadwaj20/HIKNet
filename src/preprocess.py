import numpy as np


def standardize(X, eps=1e-8):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return (X - mean) / (std + eps)
