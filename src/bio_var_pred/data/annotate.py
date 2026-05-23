import os
import subprocess
from pathlib import Path
from bio_var_pred.utils.paths import INTERIM_DATA_DIR


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
