import numpy as np

from config import PARAMS


STATS = ["mean", "std", "min", "max", "slope"]


def slope(X):
    """Least-squares slope of each parameter over the 60 time steps: (instances, 60, 24) -> (instances, 24)."""    
    time_steps = np.arange(X.shape[1])
    t = time_steps - time_steps.mean()
    deviations = np.nan_to_num(X - np.nanmean(X, axis=1, keepdims=True))  # make missing value 0 instead of nan
    return (t @ deviations) / np.sum(t ** 2)


def summary_features(X):
    """(instances, 60, 24) -> (instances, 120): create new features: mean, std, min, max and slope of each parameter."""
    return np.hstack([
        np.nanmean(X, axis=1),
        np.nanstd(X, axis=1),
        np.nanmin(X, axis=1),
        np.nanmax(X, axis=1),
        slope(X),
    ])


def stat_param_names(): #for example: mean_TOTUSJH
    return [f"{stat}_{param}" for stat in STATS for param in PARAMS]


def last_step_features(X):
    """(instances, 60, 24) -> (instances, 24)"""
    return X[:, -1]
