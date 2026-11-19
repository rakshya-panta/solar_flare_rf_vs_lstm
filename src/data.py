import numpy as np

from config import DATA_DIR, TEST, TRAIN, VAL


def load(partitions):
    files = [np.load(DATA_DIR / f"{name}.npz") for name in partitions]
    X = np.concatenate([f["X"] for f in files])
    y = np.concatenate([f["y"] for f in files])
    return X, y


def load_splits():
    return load(TRAIN), load(VAL), load(TEST)


def load_labels(partitions):
    """Only the labels y, without loading the measurements X."""
    return np.concatenate([np.load(DATA_DIR / f"{name}.npz")["y"] for name in partitions])


def calc_pos_weight(y):
    """M/X weight that balances the two classes: num of non-M/X instances / num of M/X instances."""
    return float((y == 0).sum() / (y == 1).sum())
