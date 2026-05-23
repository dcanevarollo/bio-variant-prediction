from cyvcf2 import VCF
from tqdm import tqdm
from pandas import DataFrame
import pandas as pd
import re


def parse_clinvar(vcf_path: str) -> DataFrame:
    vcf = VCF(vcf_path)

    records: list[dict] = []

    for variant in tqdm(vcf, desc="Parsing ClinVar"):
        info = variant.INFO

        clin_sig = info.get("CLNSIG")
        if clin_sig is None:
            continue
        clin_sig = str(clin_sig)

        if "Pathogenic" in clin_sig:
            label = 1
        elif "Benign" in clin_sig:
            label = 0
        else:
            continue

        ann = info.get("ANN")
        if ann is None:
            continue
        ann_entries = str(ann).split(",")

        for entry in ann_entries:
            # ANN format:
            # Allele | Annotation | Impact | Gene_Name | ...
            fields = entry.split("|")
            if len(fields) < 11:
                continue

            # Keep only missense variants
            annotation = fields[1]
            if annotation != "missense_variant":
                continue

            gene = fields[3]

            # HGVS protein notation
            # Usually field 10 in snpEff ANN
            hgvs_p = fields[10]

            if not hgvs_p.startswith("p."):
                continue

            # Example: p.Arg117His
            try:
                match = re.match(
                    r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})",
                    hgvs_p
                )

                if match is None:
                    continue

                aa_wt = match.group(1)
                pos_protein = int(match.group(2))
                aa_mut = match.group(3)
            except:
                continue

            records.append({
                "id": variant.ID,
                "chrom": variant.CHROM,
                "pos_genomic": variant.POS,
                "ref": variant.REF,
                "alt": ",".join(variant.ALT),
                "gene": gene,
                "pos_protein": pos_protein,
                "aa_wt": aa_wt,
                "aa_mut": aa_mut,
                "label": label
            })

            # Only keep first valid missense annotation
            break

    return pd.DataFrame(records)
