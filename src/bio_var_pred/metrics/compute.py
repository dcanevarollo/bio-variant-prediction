import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_gene_metrics(group: pd.DataFrame, negate: bool) -> pd.Series:
    """Compute ROC-AUC and PR-AUC for a single gene."""
    y_true  = group["label"].array
    y_score = group["esm2_log_ratio"].array
    if negate:
        y_score = -y_score

    n_total = len(y_true)
    n_pathogenic = int(y_true.sum())
    n_benign = n_total - n_pathogenic
    prevalence = n_pathogenic / n_total

    # Both classes guaranteed by is_assessable filter
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)

    return pd.Series({
        "n_variants": n_total,
        "n_pathogenic": n_pathogenic,
        "n_benign": n_benign,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })


def compute_gene_baseline(group: pd.DataFrame) -> pd.Series:
    """ROC-AUC for log_af baseline (negated: rarer = more pathogenic)."""
    y_true  = group["label"].array
    y_score = -group["log_af"].array  # negate: lower af → higher pathogenicity score

    if len(np.unique(y_true)) < 2:
        return pd.Series({"roc_auc_logaf": np.nan, "pr_auc_logaf": np.nan})

    return pd.Series({
        "roc_auc_logaf": roc_auc_score(y_true, y_score),
        "pr_auc_logaf" : average_precision_score(y_true, y_score),
    })
