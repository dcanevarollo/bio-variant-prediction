import torch
from torch import device
from pandas import DataFrame

from bio_var_pred.embeddings.assess import is_assessable


def get_device() -> device:
    print(f"PyTorch version: {torch.__version__}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device.type}")
    return device


def get_assessable_df(df: DataFrame) -> DataFrame:
    assessable_df = df.groupby("gene").filter(is_assessable).reset_index(drop=True)

    print(f"Assessable variants : {len(assessable_df)}")
    print(f"Assessable genes    : {assessable_df['gene'].nunique()}")
    print(f"Unique proteins     : {assessable_df['transcript_id'].nunique()}")

    return assessable_df


def filter_unknown_sequences(df: DataFrame) -> DataFrame:
    has_x = df["protein_seq"].str.contains("X", regex=False)
    print(f"Variants with 'X': {has_x.sum()} ({100 * has_x.mean():.2f}%)")
    print(f"Affected genes    : {df.loc[has_x, 'gene'].nunique()}")

    df = df[~has_x].reset_index(drop=True)
    print(f"\nAfter X-filter: {len(df)} variants, {df['gene'].nunique()} genes")

    return df
