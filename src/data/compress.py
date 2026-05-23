import subprocess
from pathlib import Path


def bgzip_and_index(vcf_file: Path) -> Path:
    gz_file = Path(str(vcf_file) + ".gz")
    if gz_file.exists():
        print(f"{gz_file} already exists. Skipping compress.")
        return gz_file

    subprocess.run(
        ["bgzip", "-f", str(vcf_file)],
        check=True
    )
    
    subprocess.run(
        ["tabix", "-p", "vcf", str(gz_file)],
        check=True
    )

    return gz_file
