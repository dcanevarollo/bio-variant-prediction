import re
from pandas import DataFrame


def is_assessable(group: DataFrame) -> bool:
    """
    Checks whether a given gene is assessable or not. An assessable gene means that is contains at least more than 5
    different variants with labels being benign or pathogenic.
    """
    return len(group) >= 5 and group["label"].nunique() == 2


def get_windowed_sequence(seq: str, pos_protein: int, window_size = 1024) -> tuple[str, int]:
    """
    Extracts a window of `window_size` residues centered around `pos_protein` (1-based).
    """
    seq_len = len(seq)

    if seq_len <= window_size:
        return seq, pos_protein

    pos_0based = pos_protein - 1
    half_window = window_size // 2

    start = pos_0based - half_window
    end = pos_0based + (window_size - half_window)

    if start < 0:
        start = 0
        end = window_size
    elif end > seq_len:
        end = seq_len
        start = seq_len - window_size

    windowed_seq = seq[start:end]
    new_pos_protein = pos_0based - start + 1

    return windowed_seq, new_pos_protein


def to_prott5_sequence(seq: str) -> str:
    """
    Convert a contiguous amino-acid string to the space-separated format
    expected by the Prot-T5 tokenizer, replacing any rare amino acids
    (B, Z, U, O) with 'X' as recommended in the ProtTrans documentation.
    """
    seq = re.sub(r"[BZOUJ]", "X", seq)  # normalize rare / ambiguous residues
    return " ".join(list(seq))


def mask_position(seq: str, pos_protein: int) -> str:
    """
    Replace the residue at `pos_1based` (1-based) in a space-separated
    sequence with the T5 sentinel token <extra_id_0>.

    Example
    -------
    seq_spaced = "M A G R S"
    pos_1based = 3
    → "M A <extra_id_0> R S"
    """
    tokens = seq.split(" ")
    tokens[pos_protein - 1] = "<extra_id_0>"
    return " ".join(tokens)
