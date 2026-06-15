import pandas as pd
from pandas import DataFrame
from pathlib import Path


def restore_checkpoint(checkpoint_path: Path, df: DataFrame) -> tuple[DataFrame, DataFrame, set]:
    if checkpoint_path.exists():
        checkpoint_df = pd.read_parquet(checkpoint_path)
        processed_ids = set(checkpoint_df["id"].tolist())
        print(f"Found checkpoint: {len(processed_ids)} variants already processed")
    else:
        checkpoint_df = pd.DataFrame()
        processed_ids = set()
        print(f"No checkpoint found. Initializing from scratch")

    # Not yet processed subset
    remaining_df = df[~df["id"].isin(processed_ids)].reset_index(drop=True)
    print(f"Remaining variants: {len(remaining_df)}")

    return remaining_df, checkpoint_df, processed_ids


def save_checkpoint(checkpoint_path: Path, batch_results: list, df: DataFrame) -> DataFrame:
    batch_df = pd.DataFrame(batch_results)
    df = pd.concat([df, batch_df], ignore_index=True)
    df.to_parquet(checkpoint_path, index=False)
    print(f"\tSaved checkpoint: {len(df)} variants accumulated")

    return df
