import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
from bio_var_pred.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR


__CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
__GNOMAD_EXOMES_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/"
    "release/4.1/vcf/exomes/"
    "gnomad.exomes.v4.1.sites.chr1.vcf.bgz"
)
__CHUNK_SIZE = 1024 * 1024 # 1 MiB

def download_clinvar() -> Path:
    output_file, exists = __get_output_file("clinvar.vcf.gz")

    if not exists:
        subprocess.run([
            "wget",
            "-O",
            str(output_file),
            __CLINVAR_URL
        ], check=True)

    return output_file

def download_gnomad() -> Path:
    output_file, exists = __get_output_file("gnomad.exomes.chr1.vcf.bgz", INTERIM_DATA_DIR)

    if not exists:
        response = requests.get(__GNOMAD_EXOMES_URL, stream=True)
        response.raise_for_status()

        total_size= int(response.headers.get("content-length", 0))

        with (
            open(output_file, "wb") as file,
            tqdm(
                desc=output_file.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as progress_bar
        ):
            for chunk in response.iter_content(chunk_size=__CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    return output_file

def __get_output_file(output_file_name: str, output_dir=RAW_DATA_DIR) -> tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / output_file_name

    if output_file.exists():
        print(f"{output_file.name} already exists. Skipping download.")
        return output_file, True

    return output_file, False
