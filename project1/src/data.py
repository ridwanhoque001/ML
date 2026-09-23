"""
Loading and preparing the Project 1 datasets.

File format
-----------
Every provided CSV stores the two classes side by side rather than stacked.
For a d-dimensional problem each row holds 2*d numbers: the first d are one
Class 0 sample, the remaining d are one Class 1 sample. A file with N rows
therefore contains N samples of each class, 2*N in total.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Presented in the order the assignment lists them (steps 10 through 26).
DATASETS = [
    "Gaussian_2D_Wide", "Gaussian_2D_Narrow", "Gaussian_2D_Overlap",
    "Gaussian_3D_Wide", "Gaussian_3D_Narrow", "Gaussian_3D_Overlap",
    "Moons_2D_Wide",    "Moons_2D_Narrow",    "Moons_2D_Overlap",
]


def load_dataset(name, data_dir=DATA_DIR):
    """Read one CSV and return (X, y) with the classes stacked.

    Returns
    -------
    X : ndarray, shape (2N, d)
    y : ndarray, shape (2N,) of 0/1 labels
    """
    path = os.path.join(data_dir, f"{name}.csv")
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] % 2 != 0:
        raise ValueError(f"{name}: expected an even number of columns")

    d = raw.shape[1] // 2
    class0, class1 = raw[:, :d], raw[:, d:]

    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(len(class0), dtype=int),
                        np.ones(len(class1), dtype=int)])
    return X, y


def stratified_split(X, y, fractions=(0.6, 0.2, 0.2), seed=0):
    """Split into train/validation/test, preserving the class balance.

    Splitting each class separately guarantees the same 50/50 class ratio in
    all three subsets. A purely random split could, by chance, hand one subset
    a badly skewed mix, which would make the accuracy figures harder to read.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")

    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []

    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        idx_train.append(idx[:n_train])
        idx_val.append(idx[n_train:n_train + n_val])
        idx_test.append(idx[n_train + n_val:])

    out = []
    for parts in (idx_train, idx_val, idx_test):
        sel = np.concatenate(parts)
        rng.shuffle(sel)            # interleave the classes
        out.append((X[sel], y[sel]))
    return out


class Standardizer:
    """Zero mean, unit variance scaling, fitted on the training set only.

    The Gaussian files have coordinates around 5 to 12 while the moons files
    sit within roughly -1 to 2. Feeding raw values that far from zero into
    tanh or sigmoid units drives them straight into saturation, where the
    derivative is almost zero and the network barely learns.

    The statistics come from the training split alone. Computing them over all
    the data would let information from the validation and test sets leak into
    training and would flatter the reported scores.
    """

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0     # guard a constant feature
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) treating class 1 as positive."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def classification_metrics(y_true, y_pred):
    """Accuracy, precision, recall and F1, plus the raw confusion counts.

    Accuracy alone is a fair summary here only because both classes are the
    same size. Precision and recall are reported as well so that any tendency
    of the network to favour one class shows up rather than averaging away.
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
