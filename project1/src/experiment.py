"""
Experiment harness: architecture selection and the repeated evaluation runs.

Two things happen here.

1. select_architecture() sweeps a small grid of hidden-layer sizes and learning
   rates, scoring each on the *validation* split only. This is what justifies
   the architecture reported in the write-up; the test split is never consulted.

2. run_repetitions() then runs the chosen configuration many times per dataset,
   varying the random seed so that both the weight initialization and the
   train/validation/test partition change from run to run. A single run tells
   us almost nothing, because a lucky split or a lucky initialization can move
   accuracy by several points; the spread across repetitions is the honest
   measure of performance.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.
"""

import csv
import os
import time

import numpy as np

from ann import NeuralNetwork
from data import (DATASETS, Standardizer, classification_metrics,
                  load_dataset, stratified_split)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# The configuration used for all reported runs. Chosen by select_architecture();
# see the write-up for the justification.
DEFAULT_CONFIG = {
    "hidden": (8,),
    "hidden_activation": "tanh",
    "learning_rate": 0.05,
    "momentum": 0.9,
    "batch_size": 16,
    "max_epochs": 1000,
    "patience": 50,
    "l2": 0.0,
}


def build_network(n_features, config, seed):
    return NeuralNetwork(
        layer_sizes=[n_features, *config["hidden"], 1],
        hidden_activation=config["hidden_activation"],
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        l2=config["l2"],
        seed=seed,
    )


def run_single(name, config=None, seed=0, return_model=False):
    """One complete train/validate/test cycle on one dataset."""
    config = config or DEFAULT_CONFIG
    X, y = load_dataset(name)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y, seed=seed)

    # Scale using training statistics only, then apply the same transform
    # to the validation and test splits.
    scaler = Standardizer().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = (scaler.transform(a) for a in (Xtr, Xva, Xte))

    net = build_network(Xtr.shape[1], config, seed)
    start = time.perf_counter()
    net.fit(Xtr_s, ytr, Xva_s, yva)
    elapsed = time.perf_counter() - start

    result = {
        "dataset": name,
        "seed": seed,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "epochs_run": net.epochs_run,
        "best_epoch": net.best_epoch,
        "train_time_s": elapsed,
        "train_loss": net.loss(Xtr_s, ytr),
        "val_loss": net.loss(Xva_s, yva),
        "test_loss": net.loss(Xte_s, yte),
    }
    for split, (Xs, ys) in (("train", (Xtr_s, ytr)),
                            ("val",   (Xva_s, yva)),
                            ("test",  (Xte_s, yte))):
        m = classification_metrics(ys, net.predict(Xs))
        for k, v in m.items():
            result[f"{split}_{k}"] = v

    if return_model:
        return result, net, scaler, (Xtr, ytr, Xva, yva, Xte, yte)
    return result


def run_repetitions(name, config=None, n_reps=30, base_seed=0):
    """Repeat the full cycle n_reps times with different seeds."""
    return [run_single(name, config, seed=base_seed + r) for r in range(n_reps)]


def summarize(rows, keys=("test_accuracy", "test_precision", "test_recall",
                          "test_f1", "test_loss", "epochs_run", "train_time_s")):
    """Mean, standard deviation, min and max for each metric of interest."""
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": vals.mean(), "std": vals.std(ddof=1),
                  "min": vals.min(), "max": vals.max()}
    return out


def select_architecture(datasets=None, seeds=(0, 1, 2), verbose=True):
    """Grid search over hidden size and learning rate, scored on validation.

    Kept deliberately small. The point is to justify the reported architecture
    with evidence rather than to squeeze out the last fraction of a percent,
    and a large grid evaluated on a 40-sample validation split would mostly be
    fitting noise.
    """
    datasets = datasets or ["Gaussian_2D_Overlap", "Moons_2D_Overlap",
                            "Gaussian_3D_Overlap"]
    grid = []
    for hidden in [(2,), (4,), (8,), (16,), (8, 8)]:
        for lr in [0.01, 0.05, 0.2]:
            grid.append({"hidden": hidden, "learning_rate": lr})

    records = []
    for g in grid:
        config = {**DEFAULT_CONFIG, **g}
        accs, epochs = [], []
        for name in datasets:
            for s in seeds:
                r = run_single(name, config, seed=s)
                accs.append(r["val_accuracy"])
                epochs.append(r["epochs_run"])
        rec = {"hidden": str(g["hidden"]), "learning_rate": g["learning_rate"],
               "val_accuracy_mean": float(np.mean(accs)),
               "val_accuracy_std": float(np.std(accs, ddof=1)),
               "epochs_mean": float(np.mean(epochs))}
        records.append(rec)
        if verbose:
            print(f"  hidden={rec['hidden']:8} lr={rec['learning_rate']:<5} "
                  f"val acc {rec['val_accuracy_mean']:.4f} "
                  f"+/- {rec['val_accuracy_std']:.4f}   "
                  f"epochs {rec['epochs_mean']:.0f}")
    return records


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(n_reps=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Architecture selection (scored on validation splits only)")
    arch = select_architecture()
    write_csv(arch, os.path.join(RESULTS_DIR, "architecture_search.csv"))
    best = max(arch, key=lambda r: (r["val_accuracy_mean"], -r["epochs_mean"]))
    print(f"\n  best on validation: hidden={best['hidden']} "
          f"lr={best['learning_rate']}\n")

    print(f"Main runs: {len(DATASETS)} datasets x {n_reps} repetitions")
    all_rows, summary_rows = [], []
    for name in DATASETS:
        rows = run_repetitions(name, n_reps=n_reps)
        all_rows.extend(rows)
        s = summarize(rows)
        summary_rows.append({
            "dataset": name,
            "test_accuracy_mean": s["test_accuracy"]["mean"],
            "test_accuracy_std": s["test_accuracy"]["std"],
            "test_accuracy_min": s["test_accuracy"]["min"],
            "test_accuracy_max": s["test_accuracy"]["max"],
            "test_precision_mean": s["test_precision"]["mean"],
            "test_recall_mean": s["test_recall"]["mean"],
            "test_f1_mean": s["test_f1"]["mean"],
            "test_loss_mean": s["test_loss"]["mean"],
            "epochs_mean": s["epochs_run"]["mean"],
            "epochs_std": s["epochs_run"]["std"],
            "train_time_s_mean": s["train_time_s"]["mean"],
        })
        print(f"  {name:22} test acc {s['test_accuracy']['mean']*100:6.2f}% "
              f"+/- {s['test_accuracy']['std']*100:4.2f}   "
              f"(min {s['test_accuracy']['min']*100:5.1f}, "
              f"max {s['test_accuracy']['max']*100:5.1f})   "
              f"epochs {s['epochs_run']['mean']:5.0f}")

    write_csv(all_rows, os.path.join(RESULTS_DIR, "all_runs.csv"))
    write_csv(summary_rows, os.path.join(RESULTS_DIR, "summary.csv"))
    print(f"\nWrote {len(all_rows)} runs to results/all_runs.csv")
    return summary_rows


if __name__ == "__main__":
    main()
