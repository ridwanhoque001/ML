# CS 4033/5033 Project 1: Backpropagation Network for Two-Class Classification

Ridwan Hoque. Fall 2026.

A feedforward neural network and its backpropagation training loop, written
from scratch in NumPy, applied to nine two-class classification problems.

## Layout

    report/project1.pdf   the write-up (submit this)
    report/project1.tex   its LaTeX source
    src/                  all source code
    data/                 the nine provided CSVs, renamed to remove spaces
    results/              raw and summarized experimental data
    figures/              the five figures used in the report

## Source files

| File | Purpose |
|---|---|
| `src/ann.py` | The network: forward pass, backpropagation, mini-batch SGD with momentum, early stopping, gradient checker |
| `src/data.py` | CSV loading, stratified splitting, standardization, metrics |
| `src/experiment.py` | Architecture grid search and the 270 evaluation runs |
| `src/analysis.py` | Dataset characteristics and Bayes-ceiling estimates |
| `src/capacity.py` | Hidden-layer capacity sweep on the two overlapping datasets |
| `src/plots.py` | All five figures |

## Reproducing

    pip install numpy matplotlib
    cd src
    python3 experiment.py     # grid search + 270 runs, about 4 minutes
    python3 analysis.py       # dataset characteristics and ceilings
    python3 capacity.py       # capacity sweep, about 1 minute
    python3 plots.py          # all figures

Then rebuild the report:

    cd ../report && pdflatex project1.tex && pdflatex project1.tex

## Results data

`results/all_runs.csv` has one row per run (270 rows: 9 datasets x 30 seeds).
Columns: dataset, seed, split sizes, epochs run, best epoch, training time,
loss on each split, and accuracy/precision/recall/F1/tn/fp/fn/tp for each of
the training, validation and test splits.

`results/summary.csv` aggregates those to one row per dataset.
`results/architecture_search.csv` holds the hyperparameter grid search.
`results/dataset_characteristics.csv` holds separation indices and ceilings.

## Headline result

The network reaches the estimated Bayes ceiling on eight of the nine datasets.
Accuracy tracks class separation rather than dimensionality: Gaussian 3D
Overlap (95.7%) outscores Gaussian 2D Overlap (92.9%) because the third
coordinate carries signal rather than noise.

The ninth dataset, Moons 2D Overlap, appeared to fall 1.1 points short. A
capacity sweep from 4 to 64 hidden units showed that extra capacity changes
nothing, including training accuracy, so the network is not capacity-limited
and the shortfall most likely lies in the ceiling estimate.

## Credit

All code is original and written for this assignment; no ML library is used.
An AI assistant (Claude) was used for implementation, experiments and drafting,
as permitted by the course. See the report's final section for full detail.
