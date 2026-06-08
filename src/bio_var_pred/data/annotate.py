import os
import subprocess
import pandas as pd
import requests
from typing import Generator
from pandas import DataFrame
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from pathlib import Path
from urllib3 import Retry


ENSEMBL_URL = f"https://rest.ensembl.org"

def annotate_with_snpeff(input_vcf: Path, output_vcf: Path, heap="16g") -> Path:
    if output_vcf.exists() or output_vcf.with_suffix(output_vcf.suffix + ".gz").exists():
        print(f"{output_vcf.name} already annotated. Skipping new annotation.")
        return output_vcf

    env = os.environ.copy()
    env["_JAVA_OPTIONS"] = f"-Xmx{heap}"

    print("Downloading snpEff data...")
    subprocess.run(
        ["snpEff", "download", "GRCh38.99"],
        check=True,
        env=env
    )

    print("Annotating VCF file...")
    subprocess.run(
        ["snpEff", "GRCh38.99", input_vcf],
        stdout=open(output_vcf, "w"),
        check=True,
        env=env
    )

    return output_vcf

def index(vcf_path: Path) -> None:
    if vcf_path.with_suffix(vcf_path.suffix + ".tbi").exists():
        print(f"{vcf_path.name} already indexed. Skipping new index.")
        return

    subprocess.run(
        ["tabix", "-p", "vcf", str(vcf_path)],
        check=True
    )

def fetch_protein_sequences(transcripts_ids: list[str], batch_size = 100) -> DataFrame:
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[420, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    records: list[dict[str, str | None]] = []
    n_batches = (len(transcripts_ids) + batch_size - 1) // batch_size

    for batch in tqdm(
        __chunks(transcripts_ids, batch_size),
        total=n_batches,
        desc="Fetching protein sequences"
    ):
        try:
            response = session.post(
                f"{ENSEMBL_URL}/sequence/id",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "ids": batch,
                    "type": "protein"
                },
                timeout=60
            )

            response.raise_for_status()

            results = response.json()
            seq_dict = { result.get("query"): result.get("seq") for result in results }
        except Exception as e:
            print(f"Batch failed ({batch[0]} ... {batch[-1]}): {e}")
            seq_dict = {}

        for transcript_id in batch:
            records.append({
                "transcript_id": transcript_id,
                "protein_seq": seq_dict.get(transcript_id, None)
            })

    return pd.DataFrame(records)

def __chunks(data: list[str], batch_size: int) -> Generator[list[str], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:(i + batch_size)]
