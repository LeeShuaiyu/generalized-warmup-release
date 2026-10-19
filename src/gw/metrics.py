"""Canonical classification metrics used by every study."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(probs).argmax(axis=1) == np.asarray(labels)))


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(probs).argmax(axis=1)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.union1d(labels, predictions)
    scores = []
    for cls in classes:
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-np.log(np.clip(probs[np.arange(labels.size), labels], eps, 1.0)).mean())


def multiclass_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    target = np.zeros_like(probs)
    target[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label equal-width ECE with bins (lo, hi], except first includes 0."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for i in range(n_bins):
        mask = (confidence >= edges[i]) & (confidence <= edges[i + 1]) if i == 0 else (
            (confidence > edges[i]) & (confidence <= edges[i + 1])
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def classification_metrics(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    probs = softmax(logits)
    return {
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "nll": nll(probs, labels),
        "brier": multiclass_brier(probs, labels),
        f"ece_{n_bins}": ece(probs, labels, n_bins=n_bins),
    }
