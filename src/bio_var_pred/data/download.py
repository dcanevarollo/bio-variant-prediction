import subprocess
from pathlib import Path
from bio_var_pred.utils.paths import RAW_DATA_DIR


CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"

def download_clinvar() -> Path:
    return __get_output_file("clinvar.vcf.gz", CLINVAR_URL)

def __get_output_file(file_name: str, url: str) -> Path:
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / file_name

    if output_file.exists():
        print(f"{output_file} already exists. Skipping download.")
        return output_file

    subprocess.run([
        "wget",
        "-O",
        str(output_file),
        url
    ], check=True)

    return output_file
