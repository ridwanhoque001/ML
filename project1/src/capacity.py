"""
Capacity experiment: does a wider hidden layer close the Moons Overlap gap?

Moons 2D Overlap is the one dataset where the network falls short of the
estimated ceiling (93.5 percent measured against 94.6 estimated). The report
speculates that eight tanh units are too few to trace the boundary between two
interpenetrating crescents. This script tests that claim rather than asserting
it.

Gaussian 2D Overlap runs as a control. That dataset is already at its ceiling,
and its optimal boundary is close to a straight line, so extra capacity should
buy nothing there. If width helps the moons and not the Gaussian, the
explanation is about boundary complexity. If it helps both, or neither, the
explanation is something else.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.
"""

import csv
import os

import numpy as np

from experiment import DEFAULT_CONFIG, RESULTS_DIR, run_repetitions, summarize

WIDTHS = [(4,), (8,), (16,), (32,), (64,), (16, 16)]
TARGETS = ["Moons_2D_Overlap", "Gaussian_2D_Overlap"]
CEILINGS = {"Moons_2D_Overlap": 0.946, "Gaussian_2D_Overlap": 0.925}


def main(n_reps=30):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for name in TARGETS:
        print(f"\n{name}   (estimated ceiling {CEILINGS[name]*100:.1f}%)")
        print(f"  {'hidden':>10}{'params':>8}{'test acc':>18}"
              f"{'train acc':>11}{'epochs':>8}")
        for hidden in WIDTHS:
            config = {**DEFAULT_CONFIG, "hidden": hidden}
            runs = run_repetitions(name, config, n_reps=n_reps)
            s = summarize(runs, keys=("test_accuracy", "train_accuracy",
                                      "test_loss", "epochs_run"))

            # Parameter count for a 2-input network with this hidden stack.
            sizes = [2, *hidden, 1]
            n_params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

            rows.append({
                "dataset": name,
                "hidden": str(hidden),
                "n_params": n_params,
                "test_accuracy_mean": s["test_accuracy"]["mean"],
                "test_accuracy_std": s["test_accuracy"]["std"],
                "train_accuracy_mean": s["train_accuracy"]["mean"],
                "test_loss_mean": s["test_loss"]["mean"],
                "epochs_mean": s["epochs_run"]["mean"],
                "ceiling": CEILINGS[name],
                "gap_to_ceiling": CEILINGS[name] - s["test_accuracy"]["mean"],
            })
            print(f"  {str(hidden):>10}{n_params:8}"
                  f"{s['test_accuracy']['mean']*100:11.2f} +/-"
                  f"{s['test_accuracy']['std']*100:5.2f}"
                  f"{s['train_accuracy']['mean']*100:11.2f}"
                  f"{s['epochs_run']['mean']:8.0f}")

    path = os.path.join(RESULTS_DIR, "capacity_sweep.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")
    return rows


if __name__ == "__main__":
    main()
