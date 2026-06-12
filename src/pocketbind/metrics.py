from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score


def _safe_metric(fn, *args) -> float:
    try:
        value = fn(*args)
    except Exception:
        return math.nan
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def binary_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    y = y_true.astype(float).to_numpy()
    score = y_score.astype(float).to_numpy()
    return {
        "roc_auc": _safe_metric(roc_auc_score, y, score),
        "auc0.1": _safe_metric(lambda yt, ys: roc_auc_score(yt, ys, max_fpr=0.1), y, score),
        "auprc": _safe_metric(average_precision_score, y, score),
    }


def regression_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    y = y_true.astype(float).to_numpy()
    score = y_score.astype(float).to_numpy()
    return {
        "pearson": _safe_metric(pearsonr, y, score),
        "spearman": _safe_metric(spearmanr, y, score),
        "mse": _safe_metric(mean_squared_error, y, score),
    }


def ppv_at_n(y_true: pd.Series, y_score: pd.Series, n: int | None = None) -> float:
    y = y_true.astype(float).to_numpy()
    score = y_score.astype(float).to_numpy()
    if n is None:
        n = int(np.sum(y > 0))
    if n <= 0:
        return math.nan
    order = np.argsort(-score)[:n]
    return float(np.mean(y[order] > 0))


def evaluate(df: pd.DataFrame, *, label_col: str, score_col: str, task: str) -> dict[str, float]:
    if task == "ba":
        positives = float((df[label_col] >= 0.426).sum())
    else:
        positives = float((df[label_col] > 0).sum())
    out = {"n": float(len(df)), "positives": positives}
    if task == "ba":
        out.update(regression_metrics(df[label_col], df[score_col]))
        out.update(binary_metrics((df[label_col] >= 0.426).astype(int), df[score_col]))
    else:
        out.update(binary_metrics(df[label_col], df[score_col]))
    out["ppv_at_n"] = ppv_at_n(df[label_col], df[score_col])
    return out
