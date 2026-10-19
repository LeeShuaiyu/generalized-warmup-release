"""Exact projected local-response measurements for the legacy 2-D toy MLP."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ToyModel:
    """A small fully connected ReLU classifier stored as NumPy arrays."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def copy(self) -> "ToyModel":
        return ToyModel(
            [weight.copy() for weight in self.weights],
            [bias.copy() for bias in self.biases],
        )


def init_model(seed: int, dims: tuple[int, ...] = (2, 10, 10, 2)) -> ToyModel:
    """Match the archived toy experiment's He initialization exactly."""

    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        weights.append(
            rng.normal(0.0, math.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
        )
        biases.append(np.zeros(fan_out, dtype=np.float64))
    return ToyModel(weights, biases)


def forward(
    model: ToyModel, x: np.ndarray, *, cache: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
]:
    activations = [x]
    preactivations: list[np.ndarray] = []
    hidden = x
    for weight, bias in zip(model.weights[:-1], model.biases[:-1]):
        preactivation = hidden @ weight + bias
        preactivations.append(preactivation)
        hidden = np.maximum(preactivation, 0.0)
        activations.append(hidden)
    logits = hidden @ model.weights[-1] + model.biases[-1]
    activations.append(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if cache:
        return logits, probabilities, activations, preactivations
    return logits, probabilities


def train_random_labels(
    model: ToyModel,
    *,
    seed: int,
    steps: int = 12_000,
    batch_size: int = 2_048,
    learning_rate: float = 8e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    snapshot_steps: tuple[int, ...] = (),
) -> dict[int, ToyModel]:
    """Match the archived balanced-random-label Adam warm-up exactly."""

    requested = set(snapshot_steps)
    if any(step < 0 or step > steps for step in requested):
        raise ValueError("snapshot step outside the warm-up range")
    snapshots = {0: model.copy()} if 0 in requested else {}
    rng = np.random.default_rng(seed)
    first_moment_w = [np.zeros_like(weight) for weight in model.weights]
    second_moment_w = [np.zeros_like(weight) for weight in model.weights]
    first_moment_b = [np.zeros_like(bias) for bias in model.biases]
    second_moment_b = [np.zeros_like(bias) for bias in model.biases]

    for step in range(1, steps + 1):
        x = rng.uniform(-10.0, 10.0, size=(batch_size, 2))
        y = np.concatenate(
            [
                np.zeros(batch_size // 2, dtype=np.int64),
                np.ones(batch_size - batch_size // 2, dtype=np.int64),
            ]
        )
        rng.shuffle(y)
        _, probabilities, activations, preactivations = forward(model, x, cache=True)
        gradient = probabilities.copy()
        gradient[np.arange(batch_size), y] -= 1.0
        gradient /= batch_size

        gradient_w: list[np.ndarray] = []
        gradient_b: list[np.ndarray] = []
        for layer in reversed(range(len(model.weights))):
            previous = activations[layer]
            gradient_w.insert(0, previous.T @ gradient)
            gradient_b.insert(0, gradient.sum(axis=0))
            if layer > 0:
                gradient = gradient @ model.weights[layer].T
                gradient *= preactivations[layer - 1] > 0.0

        for index in range(len(model.weights)):
            first_moment_w[index] = (
                beta1 * first_moment_w[index] + (1.0 - beta1) * gradient_w[index]
            )
            second_moment_w[index] = (
                beta2 * second_moment_w[index]
                + (1.0 - beta2) * gradient_w[index] ** 2
            )
            first_moment_b[index] = (
                beta1 * first_moment_b[index] + (1.0 - beta1) * gradient_b[index]
            )
            second_moment_b[index] = (
                beta2 * second_moment_b[index]
                + (1.0 - beta2) * gradient_b[index] ** 2
            )
            corrected_mw = first_moment_w[index] / (1.0 - beta1**step)
            corrected_vw = second_moment_w[index] / (1.0 - beta2**step)
            corrected_mb = first_moment_b[index] / (1.0 - beta1**step)
            corrected_vb = second_moment_b[index] / (1.0 - beta2**step)
            model.weights[index] -= learning_rate * corrected_mw / (
                np.sqrt(corrected_vw) + epsilon
            )
            model.biases[index] -= learning_rate * corrected_mb / (
                np.sqrt(corrected_vb) + epsilon
            )
        if step in requested:
            snapshots[step] = model.copy()

    if requested != set(snapshots):
        raise RuntimeError("not every requested warm-up snapshot was captured")
    return snapshots


def input_logit_jacobian(model: ToyModel, x: np.ndarray) -> np.ndarray:
    """Return the exact piecewise-linear logit Jacobian, shape ``N x D x C``."""

    if x.ndim != 2 or x.shape[1] != model.weights[0].shape[0]:
        raise ValueError("unexpected toy input shape")
    count, dimensions = x.shape
    jacobian = np.broadcast_to(np.eye(dimensions), (count, dimensions, dimensions)).copy()
    hidden = x
    for weight, bias in zip(model.weights[:-1], model.biases[:-1]):
        preactivation = hidden @ weight + bias
        jacobian = np.einsum("ndi,ij->ndj", jacobian, weight)
        jacobian *= (preactivation > 0.0)[:, None, :]
        hidden = np.maximum(preactivation, 0.0)
    return np.einsum("ndi,ij->ndj", jacobian, model.weights[-1])


def unit_directions(count: int) -> np.ndarray:
    """Return equally spaced unoriented directions on the 2-D unit circle."""

    if count <= 0:
        raise ValueError("direction count must be positive")
    angles = np.arange(count, dtype=np.float64) * math.pi / count
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def projected_response_metrics(
    model: ToyModel,
    points: np.ndarray,
    directions: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Measure ``||(I-11^T/C)J_z(x)v||_2`` without stochastic JVP error."""

    if directions.ndim != 2 or directions.shape[1] != points.shape[1]:
        raise ValueError("direction/input dimension mismatch")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("all directions must have unit norm")
    jacobian = input_logit_jacobian(model, points)
    directional = np.einsum("kd,ndc->nkc", directions, jacobian)
    projected = directional - directional.mean(axis=2, keepdims=True)
    responses = np.linalg.norm(projected, axis=2)
    centered_jacobian = jacobian - jacobian.mean(axis=2, keepdims=True)
    pointwise_isotropic_rms = np.sqrt(
        np.sum(centered_jacobian**2, axis=(1, 2)) / points.shape[1]
    )
    return {
        "mean_log_response": float(np.log(responses + epsilon).mean()),
        "geometric_mean_response": float(np.exp(np.log(responses + epsilon).mean())),
        "mean_response": float(responses.mean()),
        "rms_response": float(np.sqrt(np.mean(responses**2))),
        "mean_exact_isotropic_rms": float(pointwise_isotropic_rms.mean()),
        "zero_response_fraction": float(np.mean(responses <= epsilon)),
    }


def confidence_metrics(model: ToyModel, points: np.ndarray) -> dict[str, float]:
    _, probabilities = forward(model, points)
    confidence = probabilities.max(axis=1)
    return {
        "mean_confidence": float(confidence.mean()),
        "confidence_bias": float(confidence.mean() - 0.5),
    }
