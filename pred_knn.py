# pred.py

import os
from typing import List

import numpy as np
import pandas as pd  # needed indirectly via DataReader

from util.dataset_reader import DataReader

# ---------- load KNN data once at import time ----------

_HERE = os.path.dirname(__file__)

_X_TRAIN = np.load(os.path.join(_HERE, "knn_X_train.npy"))        # (N_train, D)
_Y_TRAIN = np.load(os.path.join(_HERE, "knn_y_train.npy"))        # (N_train,)
_MEAN = np.load(os.path.join(_HERE, "knn_mean.npy"))              # (D,)
_STD = np.load(os.path.join(_HERE, "knn_std.npy"))                # (D,)
_LABEL_CLASSES = np.load(os.path.join(_HERE, "label_classes.npy"), allow_pickle=True)
_K = int(np.load(os.path.join(_HERE, "knn_k.npy"))[0])            # scalar k


def _normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    return (X - _MEAN) / _STD


def _knn_predict_int(X_test: np.ndarray, k: int) -> np.ndarray:
    """
    KNN prediction (integer labels) using L2 distance.
    X_test: (M, D)
    Returns: (M,) int labels in same space as _Y_TRAIN.
    """
    X_test = _normalize(X_test)

    # squared L2 distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    # X_test: (M, D), _X_TRAIN: (N, D)
    M = X_test.shape[0]

    # (M, 1)
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
    # (1, N)
    train_sq = np.sum(_X_TRAIN ** 2, axis=1, keepdims=True).T
    # (M, N)
    dists = test_sq + train_sq - 2.0 * (X_test @ _X_TRAIN.T)

    # Get indices of k smallest distances for each test example
    # argpartition is O(N) per row, faster than full argsort
    neighbor_idx = np.argpartition(dists, k, axis=1)[:, :k]   # (M, k)

    neighbor_labels = _Y_TRAIN[neighbor_idx]                  # (M, k)

    # Majority vote per row
    preds = np.empty(M, dtype=_Y_TRAIN.dtype)
    for i in range(M):
        counts = np.bincount(neighbor_labels[i])
        preds[i] = counts.argmax()

    return preds


# ---------- API required by the assignment ----------

def predict_all(csv_filename: str) -> List[str]:
    """
    Given path to CSV, returns list of predicted labels (strings),
    one per row in the CSV, in order.
    """
    dr = DataReader(csv_filename)
    X, _ = dr.to_numpy()             # ignore labels if present
    X = np.asarray(X, dtype=np.float32)

    y_int = _knn_predict_int(X, _K)
    preds = [_LABEL_CLASSES[int(i)] for i in y_int]
    return preds


if __name__ == "__main__":
    # Sanity check + accuracy on the training CSV
    csv_path = "training_data_clean.csv"

    # Ground-truth labels from DataReader
    dr = DataReader(csv_path)
    X, y_true = dr.to_numpy()          # y_true: array/Series of strings

    # Our KNN predictions
    preds = predict_all(csv_path)      # list of strings

    preds = np.array(preds)
    y_true = np.array(y_true)

    accuracy = (preds == y_true).mean()
    print(f"{len(preds)} preds; first 10:", preds[:10])
    print("Accuracy on training_data_clean.csv:", accuracy)

