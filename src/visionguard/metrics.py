"""Validation-only threshold selection and binary classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold=0.5) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def choose_threshold(labels, probabilities, false_positive_cost=3) -> dict:
    """Minimise 3*FP+FN on validation; deterministic ties prefer F1 then 0.5."""
    probabilities = np.asarray(probabilities)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Expected finite probabilities in [0,1]")
    candidates = np.unique(np.r_[0.0, 0.5, probabilities, np.nextafter(1.0, 2.0)])
    results = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, threshold)
        cost = false_positive_cost * metrics["false_positive"] + metrics["false_negative"]
        results.append((cost, -metrics["f1"], abs(threshold - 0.5), threshold, metrics))
    cost, _, _, threshold, metrics = min(results, key=lambda row: row[:4])
    return {
        "threshold": float(threshold),
        "validation_cost": int(cost),
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": 1,
        "metrics": metrics,
    }
