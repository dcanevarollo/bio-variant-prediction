from typing import Optional

import torch
import torch.nn.functional as F
from transformers import EsmForMaskedLM, EsmTokenizer


@torch.no_grad()
def get_log_probs_for_sequence(
        seq: str,
        model: EsmForMaskedLM,
        tokenizer: EsmTokenizer,
        device: torch.device,
) -> torch.Tensor:
    """
    :return: Log-likelihoods on the vocabulary for each sequence position.
    :rtype: (seq_len, vocab_size) - exclude special tokens [CLS] and [EOS].
    """
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len + 2, vocab_size)
    logits = logits[0, 1:-1, :]

    return  F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)


# noinspection PyProtectedMember
def compute_log_ratio(
        log_probs: torch.Tensor,
        pos_protein: int,
        aa_wt: str,
        aa_mut: str,
        tokenizer: EsmTokenizer
) -> Optional[float]:
    """
    Computes log_P(aa_mut | context) - log_P(aa_wt | context) in the mutation position.
    :return: None if some aminoacid is not in the vocabulary.
    """
    pos_0based = pos_protein - 1
    unknow_id = tokenizer.unk_token_id

    wt_id = tokenizer._convert_token_to_id(aa_wt)
    mut_id = tokenizer._convert_token_to_id(aa_mut)

    if wt_id == unknow_id or mut_id == unknow_id:
        return None

    return  (log_probs[pos_0based, mut_id] - log_probs[pos_0based, wt_id]).item()
