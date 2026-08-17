from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, roc_auc_score


def classification_metrics(labels: list[float], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs)
    pred = (p >= threshold).astype(int)
    out = {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(y)) > 1 else 0.0,
        "threshold": float(threshold),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = roc_auc_score(y, p)
        out["auprc"] = average_precision_score(y, p)
    else:
        out["auroc"] = 0.0
        out["auprc"] = 0.0
    return out


def threshold_tuned_classification_metrics(labels: list[float], probs: list[float]) -> dict[str, float]:
    threshold = best_f1_threshold(labels, probs)
    metrics = classification_metrics(labels, probs, threshold=threshold)
    return {f"tuned_{key}": value for key, value in metrics.items()}


def best_f1_threshold(labels: list[float], probs: list[float]) -> float:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), p]))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (p >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def regression_metrics(targets: list[float], preds: list[float]) -> dict[str, float]:
    y = np.asarray(targets, dtype=float)
    p = np.asarray(preds, dtype=float)
    out = {
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
    }
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson"] = float(pearsonr(y, p).statistic)
        out["spearman"] = float(spearmanr(y, p).statistic)
    else:
        out["pearson"] = 0.0
        out["spearman"] = 0.0
    return out
