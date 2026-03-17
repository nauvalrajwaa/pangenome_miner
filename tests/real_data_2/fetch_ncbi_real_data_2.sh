#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCESSIONS_FILE="${ROOT_DIR}/assembly_accessions.txt"
RAW_DIR="${ROOT_DIR}/ncbi_raw"
GENOMES_DIR="${ROOT_DIR}/genomes"
ANNOTATIONS_DIR="${ROOT_DIR}/annotations"

# Set to 0 to process ALL accessions, or a positive integer to cap the run
MAX_GENOMES=0

if ! command -v datasets >/dev/null 2>&1; then
  echo "Error: NCBI datasets CLI is not installed." >&2
  echo "Install: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip is required." >&2
  exit 1
fi

rm -rf "${RAW_DIR}" "${GENOMES_DIR}" "${ANNOTATIONS_DIR}"
mkdir -p "${RAW_DIR}" "${GENOMES_DIR}" "${ANNOTATIONS_DIR}"

count=0
while IFS= read -r accession; do
  [[ -z "${accession}" ]] && continue
  if [[ "${MAX_GENOMES}" -gt 0 && "${count}" -ge "${MAX_GENOMES}" ]]; then
    echo "Reached MAX_GENOMES limit (${MAX_GENOMES}). Stopping download."
    break
  fi
  zip_path="${RAW_DIR}/${accession}.zip"
  extract_dir="${RAW_DIR}/${accession}"

  datasets download genome accession "${accession}" \
    --assembly-source RefSeq \
    --annotated \
    --include genome,gff3 \
    --filename "${zip_path}" \
    --no-progressbar

  mkdir -p "${extract_dir}"
  unzip -o "${zip_path}" -d "${extract_dir}" >/dev/null
  (( count++ ))
done < "${ACCESSIONS_FILE}"

REAL_DATA_ROOT="${ROOT_DIR}" MAX_GENOMES="${MAX_GENOMES}" python - <<'PY'
from pathlib import Path
import os
import shutil

root = Path(os.environ["REAL_DATA_ROOT"])
raw_root = root / "ncbi_raw"
genomes_dir = root / "genomes"
annotations_dir = root / "annotations"
accessions = [line.strip() for line in (root / "assembly_accessions.txt").read_text().splitlines() if line.strip()]
max_genomes = int(os.environ.get("MAX_GENOMES", 0))
if max_genomes > 0:
    accessions = accessions[:max_genomes]

missing = []
for accession in accessions:
    assembly_dir = raw_root / accession / "ncbi_dataset" / "data" / accession
    fasta_candidates = sorted(assembly_dir.glob("*_genomic.fna"))
    gff_candidates = sorted(assembly_dir.glob("*.gff"))
    if not fasta_candidates or not gff_candidates:
        missing.append(accession)
        continue
    shutil.copy2(fasta_candidates[0], genomes_dir / f"{accession}.fna")
    shutil.copy2(gff_candidates[0], annotations_dir / f"{accession}.gff")

if missing:
    raise SystemExit(f"Missing genome/gff files for: {', '.join(missing)}")

print(f"Prepared {len(accessions)} genomes in {genomes_dir}")
print(f"Prepared {len(accessions)} annotations in {annotations_dir}")
PY

echo
echo "Dataset ready: ${ROOT_DIR}"
echo "Genomes     : ${GENOMES_DIR}"
echo "Annotations : ${ANNOTATIONS_DIR}"
echo
echo "Run the pipeline with:"
echo "python main.py --genomes ${GENOMES_DIR} --annotations ${ANNOTATIONS_DIR}"
