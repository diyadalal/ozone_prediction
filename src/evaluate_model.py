"""Evaluation helpers to use for all models"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit


def best_threshold(y_true, proba):
    #Threshold that maximises F1, chosen on the validation set
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = f1s.argmax()
    return thr[i], f1s[i]


def evaluate(name, y_true, proba, threshold=0.5, verbose=True):
    #ROC-AUC, PR-AUC and F1 at threshold. Returns a dict.
    res = {
        "model/split": name,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, proba >= threshold),
        "threshold": threshold,
    }
    if verbose:
        print(f"{name}: ROC-AUC={res['roc_auc']:.3f}  PR-AUC={res['pr_auc']:.3f}  "
              f"F1@{threshold:.2f}={res['f1']:.3f}")
    return res


def full_report(y_true, proba, threshold):
    #Confusion matrix + precision/recall/F1 per class.
    pred = (proba >= threshold).astype(int)
    print("Confusion matrix (rows=actual, cols=predicted):\n", confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred, digits=3))


def time_series_cv(model, X, y, n_splits=5):
    """Expanding-window CV: each fold trains on the past, validates on the next block.
    Steadier model comparisons than a single split.
    Returns a DataFrame with one row per fold.
    """
    rows = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": k, "n_test": len(te), "positives": int(y.iloc[te].sum()),
                     "roc_auc": roc_auc_score(y.iloc[te], p),
                     "pr_auc": average_precision_score(y.iloc[te], p)})
    return pd.DataFrame(rows)