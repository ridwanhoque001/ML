"""
A feedforward artificial neural network trained by error backpropagation.

Written from scratch using only NumPy for array arithmetic; no machine learning
library supplies any part of the forward pass, the gradients, or the optimizer.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.

Design summary
--------------
    architecture     fully connected, any number of hidden layers
    hidden units     configurable per layer
    hidden act.      tanh (default), logistic sigmoid, or ReLU
    output act.      logistic sigmoid, one unit, interpreted as P(class = 1)
    loss             binary cross entropy
    initialization   Xavier/Glorot uniform, biases at zero
    optimizer        mini-batch stochastic gradient descent with momentum
    stopping         early stopping on validation loss, best weights restored
"""

import numpy as np


# --------------------------------------------------------------------------
# Activation functions
#
# Each activation is a (forward, derivative) pair. The derivative is expressed
# in terms of the *pre-activation* z, except where writing it in terms of the
# output is both cheaper and exactly equivalent (noted inline).
# --------------------------------------------------------------------------

def _sigmoid(z):
    """Numerically stable logistic sigmoid.

    The naive form 1/(1+exp(-z)) overflows for large negative z. Splitting on
    the sign of z keeps every exponential argument non-positive.
    """
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _sigmoid_prime(z):
    s = _sigmoid(z)
    return s * (1.0 - s)


def _tanh(z):
    return np.tanh(z)


def _tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


def _relu(z):
    return np.maximum(0.0, z)


def _relu_prime(z):
    # The derivative at exactly z = 0 is undefined; 0 is the conventional choice.
    return (z > 0).astype(z.dtype)


ACTIVATIONS = {
    "tanh":    (_tanh,    _tanh_prime),
    "sigmoid": (_sigmoid, _sigmoid_prime),
    "relu":    (_relu,    _relu_prime),
}


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

class NeuralNetwork:
    """A feedforward network for two-class classification.

    Parameters
    ----------
    layer_sizes : sequence of int
        Units per layer, input first and output last, e.g. [2, 8, 1] is a
        two-input network with one hidden layer of eight units and one output.
    hidden_activation : {"tanh", "sigmoid", "relu"}
        Activation applied at every hidden layer. The output layer is always a
        logistic sigmoid, because a single sigmoid output paired with binary
        cross entropy is the standard formulation for two-class problems.
    learning_rate : float
        Step size eta in the weight update.
    momentum : float
        Fraction of the previous update carried into the current one. Zero
        disables momentum and recovers plain mini-batch SGD.
    batch_size : int
        Examples per weight update. 1 gives fully stochastic (per-example)
        updates; a value at least as large as the training set gives full
        batch gradient descent.
    max_epochs : int
        Hard cap on passes through the training data.
    patience : int
        Early stopping. Training halts when the validation loss has failed to
        improve for this many consecutive epochs, and the weights from the best
        epoch are restored.
    l2 : float
        Coefficient of an optional L2 penalty on the weights (not the biases).
        Zero disables regularization.
    seed : int or None
        Seed for weight initialization and batch shuffling, so a run can be
        reproduced exactly.
    """

    def __init__(self, layer_sizes, hidden_activation="tanh", learning_rate=0.1,
                 momentum=0.9, batch_size=16, max_epochs=1000, patience=50,
                 l2=0.0, seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        if layer_sizes[-1] != 1:
            raise ValueError("this implementation has a single output unit")
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {hidden_activation!r}")

        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.seed = seed

        self._act, self._act_prime = ACTIVATIONS[hidden_activation]
        self._init_parameters()

        # Populated by fit().
        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}
        self.epochs_run = 0
        self.best_epoch = 0

    # ---------------------------------------------------------------- setup

    def _init_parameters(self):
        """Xavier/Glorot uniform initialization.

        Weights are drawn from U(-limit, +limit) with limit = sqrt(6/(fan_in +
        fan_out)). This keeps the variance of the activations roughly constant
        as signals move forward through the layers, which matters for the
        saturating activations (tanh and sigmoid) used here: weights that are
        too large push units into their flat regions, where the derivative is
        near zero and learning stalls.

        Biases start at zero. There is no symmetry problem in doing so because
        the weights are already asymmetric.
        """
        rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        # Momentum buffers: the previous update for each parameter array.
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._v_b = [np.zeros_like(b) for b in self.biases]

    # -------------------------------------------------------------- forward

    def _forward(self, X):
        """Run the forward pass.

        Returns
        -------
        activations : list of ndarray
            activations[0] is the input; activations[i] is the output of
            layer i. Retained because backpropagation needs them.
        pre_activations : list of ndarray
            The weighted sums z for each layer, needed for the activation
            derivatives.
        """
        activations = [X]
        pre_activations = []
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            # Hidden layers use the chosen activation; the output layer is
            # always a sigmoid so its value reads as a probability.
            a = _sigmoid(z) if i == n_layers - 1 else self._act(z)
            activations.append(a)

        return activations, pre_activations

    def predict_proba(self, X):
        """Return P(class = 1) for each row of X."""
        return self._forward(np.asarray(X, dtype=float))[0][-1]

    def predict(self, X, threshold=0.5):
        """Return hard class labels in {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ----------------------------------------------------------------- loss

    def loss(self, X, y):
        """Mean binary cross entropy, plus the L2 penalty if enabled."""
        p = self.predict_proba(X)
        return self._loss_from_proba(p, y)

    def _loss_from_proba(self, p, y):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        # Clip to keep log() finite when the network is fully confident.
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if self.l2 > 0.0:
            bce += self.l2 * sum(np.sum(w ** 2) for w in self.weights)
        return float(bce)

    # -------------------------------------------------------- backward pass

    def _backward(self, activations, pre_activations, y):
        """Backpropagate the error and return gradients for weights and biases.

        The output layer pairs a sigmoid with binary cross entropy. Their
        derivatives combine and cancel, leaving the delta for the output layer
        as simply (prediction - target); the sigmoid factor sigma'(z) does not
        appear. This is not only cheaper but avoids the vanishing gradient that
        a squared-error loss would suffer at a saturated output.
        """
        n = y.shape[0]
        y = y.reshape(-1, 1)
        n_layers = len(self.weights)

        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        # Output layer: the sigmoid/cross-entropy simplification.
        delta = (activations[-1] - y) / n

        for i in reversed(range(n_layers)):
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if self.l2 > 0.0:
                grads_w[i] = grads_w[i] + 2.0 * self.l2 * self.weights[i]
            if i > 0:
                # Propagate the error to the previous layer, then scale by that
                # layer's local activation derivative.
                delta = (delta @ self.weights[i].T) * self._act_prime(pre_activations[i - 1])

        return grads_w, grads_b

    # -------------------------------------------------------------- fitting

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """Train by mini-batch gradient descent with momentum.

        If validation data is supplied, early stopping is active: the weights
        that achieved the lowest validation loss are restored when training
        ends. Without validation data the network simply runs for max_epochs.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)

        rng = np.random.default_rng(
            None if self.seed is None else self.seed + 10_000)
        n = X_train.shape[0]
        batch = min(self.batch_size, n)

        best_val = np.inf
        best_params = None
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            # Reshuffle every epoch so the network does not learn the order in
            # which examples happen to sit in the file.
            order = rng.permutation(n)
            Xs, ys = X_train[order], y_train[order]

            for start in range(0, n, batch):
                Xb, yb = Xs[start:start + batch], ys[start:start + batch]
                acts, pre = self._forward(Xb)
                gw, gb = self._backward(acts, pre, yb)

                for i in range(len(self.weights)):
                    # v <- momentum * v - lr * grad ; theta <- theta + v
                    self._v_w[i] = self.momentum * self._v_w[i] - self.learning_rate * gw[i]
                    self._v_b[i] = self.momentum * self._v_b[i] - self.learning_rate * gb[i]
                    self.weights[i] += self._v_w[i]
                    self.biases[i] += self._v_b[i]

            # End-of-epoch bookkeeping.
            tr_p = self.predict_proba(X_train)
            tr_loss = self._loss_from_proba(tr_p, y_train)
            tr_acc = float(np.mean((tr_p >= 0.5).astype(int) == y_train))
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            if has_val:
                va_p = self.predict_proba(X_val)
                va_loss = self._loss_from_proba(va_p, y_val)
                va_acc = float(np.mean((va_p >= 0.5).astype(int) == y_val))
                self.history["val_loss"].append(va_loss)
                self.history["val_acc"].append(va_acc)

                if va_loss < best_val - 1e-6:
                    best_val = va_loss
                    best_params = ([w.copy() for w in self.weights],
                                   [b.copy() for b in self.biases])
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.epochs_run = epoch

            if verbose and epoch % 50 == 0:
                msg = f"epoch {epoch:4d}  train loss {tr_loss:.4f}"
                if has_val:
                    msg += f"  val loss {self.history['val_loss'][-1]:.4f}"
                print(msg)

            if has_val and epochs_without_improvement >= self.patience:
                break

        if best_params is not None:
            self.weights, self.biases = best_params

        return self

    # ------------------------------------------------------- gradient check

    def gradient_check(self, X, y, epsilon=1e-6, tolerance=1e-6):
        """Compare backpropagation gradients against numerical derivatives.

        This is the correctness test for the backward pass. For each parameter
        the loss is re-evaluated at theta + epsilon and theta - epsilon, and the
        resulting central difference is compared with the analytic gradient.
        Returns the largest relative discrepancy found.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        acts, pre = self._forward(X)
        grads_w, grads_b = self._backward(acts, pre, y)

        worst = 0.0
        for params, grads in ((self.weights, grads_w), (self.biases, grads_b)):
            for layer_idx, (theta, grad) in enumerate(zip(params, grads)):
                it = np.nditer(theta, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    original = theta[idx]

                    theta[idx] = original + epsilon
                    loss_plus = self.loss(X, y)
                    theta[idx] = original - epsilon
                    loss_minus = self.loss(X, y)
                    theta[idx] = original

                    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
                    analytic = grad[idx]
                    denom = max(1.0, abs(numerical) + abs(analytic))
                    worst = max(worst, abs(numerical - analytic) / denom)
                    it.iternext()

        return worst, worst < tolerance
