import os
import subprocess
from pathlib import Path
from src.utils.paths import INTERIM_DATA_DIR


def annotate_with_snpeff(input_vcf: Path, output_vcf: Path, heap="16g") -> Path:
    if output_vcf.exists():
        print(f"{output_vcf} already exists. Skipping annotation.")
        return output_vcf

    env = os.environ.copy()
    env["_JAVA_OPTIONS"] = f"-Xmx{heap}"

    snpeff_genes_file = INTERIM_DATA_DIR / "snpEff_genes.txt"
    if not snpeff_genes_file.exists():
        print("Download genes annotation for snpEff...")
        subprocess.run(
            ["snpEff", "download", "GRCh38.99"],
            check=True,
            env=env
        )

    subprocess.run(
        ["snpEff", "GRCh38.99", input_vcf],
        stdout=open(output_vcf, "w"),
        check=True,
        env=env
    )

    return output_vcf
