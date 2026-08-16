from __future__ import annotations

import math

import numpy as np


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def roc_curve_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int(labels.size - labels.sum()))
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    return (
        np.concatenate(([0.0], false_positives / negatives, [1.0])),
        np.concatenate(([0.0], true_positives / positives, [1.0])),
    )


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = (true_positive + true_negative) / max(1, labels.size)
    precision = true_positive / max(1, true_positive + false_positive)
    sensitivity = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "roc_auc": binary_roc_auc(labels, probabilities),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "threshold": float(threshold),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def bootstrap_auc_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    repetitions: int = 1000,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        indices = generator.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indices]
        if sampled_labels.min() == sampled_labels.max():
            continue
        estimates.append(binary_roc_auc(sampled_labels, probabilities[indices]))
    if not estimates:
        return math.nan, math.nan
    lower, upper = np.percentile(estimates, (2.5, 97.5))
    return float(lower), float(upper)

