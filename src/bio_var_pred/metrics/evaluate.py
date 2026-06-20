import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_gene(group: pd.DataFrame, score_col: str) -> dict | None:
    """
    Compute ROC-AUC and PR-AUC for a single gene.
    Returns None if the group has only one class or any NaN scores.
    """
    sub = group.dropna(subset=[score_col])
    if sub["label"].nunique() < 2:
        return None

    y_true = sub["label"].array
    y_score = sub[score_col].array

    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc":  average_precision_score(y_true, y_score),
        "n_variants":   len(sub),
        "n_pathogenic": int(y_true.sum()),
        "n_benign":     int((1 - y_true).sum()),
    }


def assign_stratum(n):
    if n <= 10:
        return "5–10"
    elif n <= 50:
        return "11–50"
    else:
        return ">50"
