# real_data_2

Second real-data test bundle for manual Phase 1-3 runs.

This folder is set up to download 15 complete RefSeq `Streptomyces` genomes
plus matching GFF3 annotations from NCBI Datasets.

## Files

- `assembly_accessions.txt` - pinned list of 15 assembly accessions
- `fetch_ncbi_real_data_2.sh` - download + unpack + normalize into `genomes/` and `annotations/`

## Prepare the dataset

```bash
bash tests/real_data_2/fetch_ncbi_real_data_2.sh
```

## Run the pipeline

```bash
python main.py \
  --genomes tests/real_data_2/genomes \
  --annotations tests/real_data_2/annotations
```

## Notes

- The script requires `datasets` and `unzip`.
- Output FASTA/GFF files are renamed to the accession stem so the project can
  match genome and annotation files directly.
- The raw NCBI download is preserved under `tests/real_data_2/ncbi_raw/`.
