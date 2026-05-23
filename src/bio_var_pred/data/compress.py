import subprocess
from pathlib import Path


def bgzip(vcf_path: Path) -> Path:
    gz_file = vcf_path.with_suffix(vcf_path.suffix + ".gz")
    if gz_file.exists():
        print(f"{gz_file.name} already compressed. Skipping new compress.")
        return gz_file

    subprocess.run(
        ["bgzip", "-f", str(vcf_path)],
        check=True
    )

    return gz_file
