# CS 4033/5033 — Machine Learning Fundamentals

Coursework for CS 4033/5033, Machine Learning Fundamentals, University of
Oklahoma, Fall 2026. Ridwan Hoque.

## Contents

| Folder | Assignment |
|---|---|
| `hw1/` | Homework 1 — artificial neuron representation, decision boundaries, and the perceptron learning rule |
| `project1/` | Project 1 — a backpropagation network for two-class classification, written from scratch in NumPy |

## Homework 1

Analytic work on a single artificial neuron: the augmented input vector, the
decision boundary as the solution of `net = 0`, linear separability, and a full
trace of the perceptron learning rule to convergence. The write-up includes
scans of the handwritten working as an appendix.

## Project 1

A feedforward network and its backpropagation training loop implemented from
scratch; no machine learning library supplies the forward pass, the gradients,
or the optimizer. Applied to nine synthetic two-class problems.

Headline results:

- Backpropagation gradients agree with central differences to 1.6e-10.
- The network reaches the estimated Bayes ceiling on eight of nine datasets.
- Accuracy tracks class separation, not dimensionality: Gaussian 3D Overlap
  (95.7%) outscores Gaussian 2D Overlap (92.9%) because the third coordinate
  carries signal rather than noise.
- A capacity sweep from 4 to 64 hidden units changes nothing, including
  training accuracy, so the one dataset that appeared short of its ceiling is
  not capacity-limited.

See `project1/README.md` for how to reproduce everything.

## Credit

All code is original and written for these assignments. An AI assistant
(Claude) was used for implementation, experiments and drafting, as permitted by
the course; each write-up carries a full credit statement.
