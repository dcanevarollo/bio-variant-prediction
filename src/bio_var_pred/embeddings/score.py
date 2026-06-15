from typing import Optional

import torch
import torch.nn.functional as F
from torch import device, Tensor
from transformers import EsmForMaskedLM, EsmTokenizer, T5ForConditionalGeneration, T5Tokenizer, PreTrainedModel

from bio_var_pred.embeddings.assess import get_windowed_sequence, to_prott5_sequence, mask_position


@torch.no_grad()
def get_esm2_log_probs_for_sequence(
        seq: str,
        model: EsmForMaskedLM | PreTrainedModel,
        tokenizer: EsmTokenizer,
        device: device,
) -> Tensor:
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
def compute_esm2_log_ratio(
        log_probs: Tensor,
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


@torch.no_grad()
def compute_prott5_log_ratio(
    seq: str,
    pos_protein: int,
    aa_wt: str,
    aa_mut: str,
    model: T5ForConditionalGeneration | PreTrainedModel,
    tokenizer: T5Tokenizer,
    device: device,
    window_size = 512
) -> Optional[float]:
    """
    Compute log P(aa_mut | masked context) - log P(aa_wt | masked context)
    for a single missense variant using Prot-T5.

    :return: None if either amino acid is not in the tokenizer vocabulary.
    """
    windowed_seq, new_pos_protein = get_windowed_sequence(seq, pos_protein, window_size)
    seq_spaced = to_prott5_sequence(windowed_seq)
    masked_seq = mask_position(seq_spaced, new_pos_protein)

    inputs = tokenizer(
        masked_seq,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)

    # The decoder is seeded with <extra_id_0> so that its first output
    # position predicts the token that should fill the masked span.
    sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    decoder_input_ids = torch.tensor([[sentinel_id]], device=device)

    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)

    logits = outputs.logits[0, 0, :]  # (1, 1, vocab_size) - one decoder step, one batch element
    log_probs = F.log_softmax(logits, dim=-1)

    unknow_id = tokenizer.unk_token_id

    # The Prot-T5 SentencePiece tokenizer prefixes every token with '▁'
    # (U+2581, the SentencePiece word-boundary marker). A bare letter like
    # 'A' is therefore stored as '▁A' in the vocabulary (id=3), not 'A'.
    # Using convert_tokens_to_ids('A') returns <unk>; we must use '▁A'.
    wt_id = tokenizer.convert_tokens_to_ids("\u2581" + aa_wt)
    mut_id = tokenizer.convert_tokens_to_ids("\u2581" + aa_mut)

    if wt_id == unknow_id or mut_id == unknow_id:
        return None

    return (log_probs[mut_id] - log_probs[wt_id]).item()
