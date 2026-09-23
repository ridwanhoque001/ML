"""
Dataset characterisation and a Bayes-optimal reference point.

The network's accuracy on a dataset is only meaningful against what is
achievable on that dataset. Where the classes genuinely overlap, no classifier
of any kind can reach 100 percent, so the right question is not "did it get
100?" but "did it get close to the ceiling?".

For the Gaussian sets the ceiling can be estimated directly: fit a full
covariance Gaussian to each class, then measure the error rate of the optimal
(Bayes) rule for those two fitted densities by Monte Carlo. For the moons the
generating densities are not Gaussian, so a nearest-neighbour estimate of the
Bayes rate is used instead.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.
"""

import csv
import os

import numpy as np

from data import DATASETS, load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _log_gaussian_pdf(X, mean, cov):
    d = X.shape[1]
    cov = cov + 1e-9 * np.eye(d)            # keep the inverse well conditioned
    diff = X - mean
    sol = np.linalg.solve(cov, diff.T).T
    quad = np.einsum("ij,ij->i", diff, sol)
    sign, logdet = np.linalg.slogdet(cov)
    return -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))


def bayes_accuracy_gaussian(X, y, n_samples=400_000, seed=0):
    """Accuracy of the optimal rule for two fitted Gaussians, equal priors."""
    rng = np.random.default_rng(seed)
    params = []
    for label in (0, 1):
        Xi = X[y == label]
        params.append((Xi.mean(axis=0), np.cov(Xi, rowvar=False)))

    correct = 0
    for true_label, (mean, cov) in enumerate(params):
        draws = rng.multivariate_normal(mean, cov, size=n_samples // 2)
        l0 = _log_gaussian_pdf(draws, *params[0])
        l1 = _log_gaussian_pdf(draws, *params[1])
        pred = (l1 > l0).astype(int)
        correct += np.sum(pred == true_label)
    return correct / (2 * (n_samples // 2))


def knn_bayes_estimate(X, y, k=15):
    """Leave-one-out k-nearest-neighbour accuracy.

    Used for the moons, where the class densities are not Gaussian. With a
    reasonably large k this is a practical stand-in for the achievable ceiling:
    it is a consistent estimator of the Bayes rule as the sample grows.
    """
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)             # exclude the point itself
    nn = np.argsort(d2, axis=1)[:, :k]
    votes = y[nn].mean(axis=1)
    return float(np.mean((votes > 0.5).astype(int) == y))


def characterize(name):
    X, y = load_dataset(name)
    X0, X1 = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    centroid_distance = float(np.linalg.norm(mu0 - mu1))

    # Pooled within-class spread, as a single number for comparison.
    pooled = np.sqrt(0.5 * (X0.var(axis=0) + X1.var(axis=0))).mean()
    # A separation index: centroid gap measured in units of within-class spread.
    separation_index = centroid_distance / pooled

    if name.startswith("Gaussian"):
        ceiling = bayes_accuracy_gaussian(X, y)
        method = "Gaussian Bayes"
    else:
        ceiling = knn_bayes_estimate(X, y)
        method = "15-NN leave-one-out"

    return {
        "dataset": name,
        "n_total": int(len(y)),
        "dimensions": int(X.shape[1]),
        "centroid_distance": centroid_distance,
        "pooled_within_sd": float(pooled),
        "separation_index": float(separation_index),
        "ceiling_accuracy": float(ceiling),
        "ceiling_method": method,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = [characterize(n) for n in DATASETS]

    with open(os.path.join(RESULTS_DIR, "dataset_characteristics.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'dataset':22}{'n':>6}{'dim':>5}{'cent.dist':>11}"
          f"{'sep.index':>11}{'ceiling':>10}")
    for r in rows:
        print(f"{r['dataset']:22}{r['n_total']:6}{r['dimensions']:5}"
              f"{r['centroid_distance']:11.2f}{r['separation_index']:11.2f}"
              f"{r['ceiling_accuracy']*100:9.1f}%")
    return rows


if __name__ == "__main__":
    main()
