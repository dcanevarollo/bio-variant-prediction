import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import PreTrainedModel, EsmModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from bio_var_pred.embeddings.assess import get_windowed_sequence, to_prott5_sequence
from bio_var_pred.embeddings.score import get_residue_embedding


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

def compute_delta_norm(
    row: pd.Series,
    model: EsmModel | T5EncoderModel | PreTrainedModel,
    tokenizer: AutoTokenizer | T5Tokenizer,
    device: torch.device,
    layer: int = 33
) -> float:
    """
    Compute ||epsilon_mut - epsilon_wt||_2 for a single variant.

    Two forward passes:
      1. Wild-type sequence → embedding at pos_protein, layer 33
      2. Mutant sequence (aa_mut substituted at pos_protein) → same
    """
    seq_wt = row["protein_seq"]
    pos = row["pos_protein"]   # 1-based
    aa_mut = row["aa_mut"]

    # Build mutant sequence
    seq_mut = seq_wt[: pos - 1] + aa_mut + seq_wt[pos:]

    # Apply centred window to both sequences
    seq_wt_w, pos_w = get_windowed_sequence(seq_wt, pos)
    seq_mut_w, _ = get_windowed_sequence(seq_mut, pos)  # same window boundaries

    if isinstance(model, T5EncoderModel):
        seq_wt_w = to_prott5_sequence(seq_wt_w)
        seq_mut_w = to_prott5_sequence(seq_mut_w)

    eps_wt  = get_residue_embedding(seq_wt_w,  pos_w, model, tokenizer, device, layer)
    eps_mut = get_residue_embedding(seq_mut_w, pos_w, model, tokenizer, device, layer)

    delta = eps_mut - eps_wt
    return float(np.linalg.norm(delta))
