"""Adversarial robustness of the MLP detector: FGSM and adversarial training.

FGSM needs the gradient of the loss with respect to the input. sklearn's
MLPClassifier exposes its weights (coefs_, intercepts_), so the backward pass is
written here in numpy instead of adding PyTorch:

    forward   a_0 = x;  z_l = a_{l-1} W_l + b_l;  a_l = relu(z_l);  p = softmax(z_L)
    backward  dL/dz_L = p - onehot(y)                      (cross-entropy with softmax)
              dL/da_{l-1} = dL/dz_l W_l^T;  dL/dz_{l-1} = dL/da_{l-1} * [z_{l-1} > 0]
              dL/dx = dL/dz_1 W_1^T

The attack works in the preprocessed space (one-hot + standardized) and
perturbs only the columns of features an attacker can change; the protocol,
service and flag one-hots stay fixed. Perturbed values are clipped so the raw
feature never goes below zero (byte counts and durations cannot be negative).

    x_adv = clip(x + eps * sign(dL/dx) * mutable_mask)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _activation(name: str):
    if name == "relu":
        return lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(float)
    if name == "tanh":
        return np.tanh, lambda z, a: 1.0 - a ** 2
    if name == "logistic":
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return sig, lambda z, a: a * (1.0 - a)
    if name == "identity":
        return (lambda z: z), (lambda z, a: np.ones_like(z))
    raise ValueError(f"Unsupported activation {name!r}")


def forward(mlp: MLPClassifier, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Pre-activations, activations and output probabilities for a fitted MLPClassifier."""
    act, _ = _activation(mlp.activation)
    zs, activations = [], [X]
    a = X
    n_layers = len(mlp.coefs_)
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        z = a @ W + b
        zs.append(z)
        if i < n_layers - 1:
            a = act(z)
            activations.append(a)
    out = zs[-1]
    if out.shape[1] == 1:  # binary: logistic output
        p1 = 1.0 / (1.0 + np.exp(-out[:, 0]))
        probs = np.column_stack([1.0 - p1, p1])
    else:
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    return zs, activations, probs


def input_gradient(mlp: MLPClassifier, X: np.ndarray, y) -> np.ndarray:
    """dL/dx of the mean-free per-sample cross-entropy loss, one row per sample."""
    _, dact = _activation(mlp.activation)
    zs, activations, probs = forward(mlp, X)
    classes = list(mlp.classes_)
    idx = np.array([classes.index(v) for v in y])
    if zs[-1].shape[1] == 1:
        delta = (probs[:, 1] - (idx == 1)).reshape(-1, 1)  # d(BCE)/dz for the single logistic unit
    else:
        delta = probs.copy()
        delta[np.arange(len(idx)), idx] -= 1.0
    for layer in range(len(mlp.coefs_) - 1, 0, -1):
        grad_a = delta @ mlp.coefs_[layer].T
        delta = grad_a * dact(zs[layer - 1], activations[layer])
    return delta @ mlp.coefs_[0].T


def loss(mlp: MLPClassifier, X: np.ndarray, y) -> np.ndarray:
    """Per-sample cross-entropy, used to check the gradient numerically."""
    _, _, probs = forward(mlp, X)
    classes = list(mlp.classes_)
    idx = np.array([classes.index(v) for v in y])
    return -np.log(np.clip(probs[np.arange(len(idx)), idx], 1e-12, None))


# --------------------------------------------------------------------------- the attack


class FGSM:
    """FGSM against an MLP pipeline (preprocessor + MLPClassifier)."""

    def __init__(self, pipeline: Pipeline, mutable: Sequence[str]):
        self.pipeline = pipeline
        self.pre = pipeline.named_steps["pre"]
        self.mlp: MLPClassifier = pipeline.named_steps["model"]
        if not isinstance(self.mlp, MLPClassifier):
            raise TypeError("FGSM needs a pipeline ending in MLPClassifier")
        names = list(self.pre.get_feature_names_out())
        self.mask = np.array([n in set(mutable) for n in names], dtype=float)
        if not self.mask.any():
            raise ValueError("None of the mutable features appear in the preprocessed columns")
        self.lower = self._raw_zero(names)

    def _raw_zero(self, names: List[str]) -> np.ndarray:
        """Where raw value 0 lands in the preprocessed space, per column (-inf if unconstrained)."""
        lower = np.full(len(names), -np.inf)
        for name, transformer, cols in self.pre.transformers_:
            if isinstance(transformer, StandardScaler):
                for col, mean, scale in zip(cols, transformer.mean_, transformer.scale_):
                    lower[names.index(col)] = (0.0 - mean) / scale
        return lower

    def transform(self, X) -> np.ndarray:
        return np.asarray(self.pre.transform(X), dtype=float)

    def perturb(self, Xt: np.ndarray, y, eps: float) -> np.ndarray:
        """Adversarial examples in the preprocessed space."""
        if eps == 0:
            return Xt.copy()
        grad = input_gradient(self.mlp, Xt, list(y))
        return np.maximum(Xt + eps * np.sign(grad) * self.mask, self.lower)

    def accuracy(self, Xt: np.ndarray, y) -> float:
        return float(np.mean(self.mlp.predict(Xt) == np.asarray(y)))

    def accuracy_vs_epsilon(self, X, y, epsilons: Sequence[float]) -> List[Dict[str, float]]:
        """Accuracy of this model under white-box FGSM at each epsilon (in standard deviations)."""
        Xt = self.transform(X)
        return [{"epsilon": float(e), "accuracy": self.accuracy(self.perturb(Xt, y, e), y)} for e in epsilons]


def adversarial_training(pipeline: Pipeline, X, y, mutable: Sequence[str], eps: float = 0.5,
                         rounds: int = 3, seed: int = 0) -> Pipeline:
    """Retrain the MLP on clean plus FGSM examples, crafting new examples against the current model each round.

    Round 1 attacks the original model; each later round attacks the model just
    trained, so the defence is not only fitted to the first model's weaknesses.
    The preprocessor is kept as fitted on clean data, so both models see the same
    feature space and can be attacked and compared on the same axes.
    """
    Xt = FGSM(pipeline, mutable).transform(X)
    y = np.asarray(y)
    current = pipeline
    X_parts, y_parts = [Xt], [y]
    for _ in range(rounds):
        X_parts.append(FGSM(current, mutable).perturb(Xt, y, eps))
        y_parts.append(y)
        mlp = clone(pipeline.named_steps["model"]).set_params(random_state=seed)
        mlp.fit(np.vstack(X_parts), np.concatenate(y_parts))
        current = Pipeline([("pre", pipeline.named_steps["pre"]), ("model", mlp)])
    return current


def robustness_curves(clean: Pipeline, robust: Pipeline, X, y, mutable: Sequence[str],
                      epsilons: Sequence[float]) -> Dict[str, List[Dict[str, float]]]:
    """Accuracy against epsilon for both models, each attacked with its own gradients."""
    return {
        "standard": FGSM(clean, mutable).accuracy_vs_epsilon(X, y, epsilons),
        "adversarially_trained": FGSM(robust, mutable).accuracy_vs_epsilon(X, y, epsilons),
    }
