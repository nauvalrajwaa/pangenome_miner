# PanAdapt-BGC Miner

**Pangenome Adaptive & Alien Biosynthetic Gene Cluster Miner**

A modular, end-to-end Python pipeline for discovering Biosynthetic Gene Clusters (BGCs) in microbial genomes by chaining three analytical phases: **Pangenome Analysis → HGT Detection → AI BGC Prediction**. Designed for soil MAGs and other environmental metagenome-assembled genomes.

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Flowchart](#workflow-flowchart)
3. [Phase 1 — Pangenome Miner](#phase-1--pangenome-miner)
4. [Phase 2 — HGT Detective](#phase-2--hgt-detective)
5. [Phase 3 — AI BGC Predictor](#phase-3--ai-bgc-predictor)
6. [Output Files](#output-files)
7. [Installation](#installation)
8. [Usage & Examples](#usage--examples)
9. [Project Structure](#project-structure)
10. [Testing](#testing)

---

## Overview

PanAdapt-BGC Miner answers one central question:

> **Which genes in an accessory/adaptive genome are horizontally transferred biosynthetic gene clusters of biotechnological interest?**

It ingests raw genome assemblies (FASTA) and their NCBI PGAP or Prokka annotations (GFF3), then pipes them through three phases, each building on the previous one. Final output is a set of publication-ready plots and self-contained HTML reports — one per phase.

---

## Workflow Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                      │
│    genomes/*.fna  ──────┐                                               │
│    annotations/*.gff ──►│  main.py --genomes ... --annotations ...      │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — PangenomeMiner (pangenome_miner.py)                         ║
║                                                                         ║
║  1. Parse GFF3 → extract CDS features (gene_id, coords, product, seq)  ║
║  2. Load FASTA → attach DNA sequence to every gene record               ║
║  3. K-mer Jaccard clustering → group orthologous genes into clusters    ║
║     (threshold: --identity, default 0.80)                               ║
║  4. Build presence/absence matrix  [clusters × strains]                 ║
║  5. Partition pangenome:                                                 ║
║     ┌─────────────────────────────────────────────────────────────┐     ║
║     │  Core genes     present in ≥ 95% of strains (conserved)    │     ║
║     │  Shell genes    between accessory and core thresholds       │     ║
║     │  Accessory genes present in ≤ max(10%, 1/n_strains)        │     ║
║     └─────────────────────────────────────────────────────────────┘     ║
║                                                                         ║
║  Outputs:  PangenomeResult dataclass → passed to Phase 2               ║
║            presence_absence_matrix.csv                                  ║
║            phase1/presence_absence_heatmap.png                          ║
║            phase1/pangenome_summary.png                                 ║
║            phase1/phase1_report.html                                    ║
╚══════════════════════════════╤══════════════════════════════════════════╝
                               │  accessory_records  (List[GeneRecord])
                               │  fasta_store        (raw DNA sequences)
                               ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — HGT Detective (hgt_detective.py)                            ║
║                                                                         ║
║  Input: accessory genes only (shell + accessory partition)              ║
║                                                                         ║
║  Per-strain host profile:                                               ║
║    ├── Mean host GC content (from full genome sequence)                 ║
║    └── Mean tetranucleotide (4-mer) frequency vector (256-dim)          ║
║                                                                         ║
║  Per-gene feature extraction:                                           ║
║    ├── gc_deviation    = |gene_GC − host_GC| / host_GC                 ║
║    ├── kmer_deviation  = 1 − cosine_similarity(gene_4mer, host_4mer)   ║
║    ├── mge_proximity   = within 10 kbp of a transposase / integrase    ║
║    └── gc_content      = raw GC fraction of gene sequence              ║
║                                                                         ║
║  Anomaly detection:                                                     ║
║    └── Isolation Forest (sklearn) on [gc_dev, kmer_dev, mge_prox, gc]  ║
║        contamination = 0.30  →  top 30% anomalous genes flagged        ║
║                                                                         ║
║  Evidence assembly:                                                     ║
║    ├── "High GC deviation"    if gc_deviation > 0.05                   ║
║    ├── "High k-mer deviation" if kmer_deviation > 0.03                 ║
║    ├── "MGE proximal"         if within 10 kbp of tnpA/integrase       ║
║    └── "Isolation Forest anomaly"  if IsolationForest label == −1      ║
║                                                                         ║
║  Outputs:  HGTResult dataclass → alien_records passed to Phase 3       ║
║            phase2/genomic_island_<strain>.png  (one per strain)        ║
║            phase2/hgt_feature_distributions.png                        ║
║            phase2/phase2_report.html                                   ║
╚══════════════════════════════╤══════════════════════════════════════════╝
                               │  alien_records  (HGT-flagged genes)
                               ▼
╔═════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — AI BGC Predictor (bgc_predictor.py)                        ║
║                                                                         ║
║  Input: HGT-flagged alien gene records from Phase 2                    ║
║                                                                         ║
║  Feature vector (per alien gene):                                       ║
║    [gc_content, gc_deviation, kmer_deviation, anomaly_score,           ║
║     gene_length_norm, mge_proximity, is_multimodule_keyword,           ║
║     is_regulatory_keyword]                                              ║
║                                                                         ║
║  Keyword boost scoring (product annotation):                            ║
║    ├── NRPS:        "nrps", "non-ribosomal peptide synthetase", "NrpS" ║
║    ├── PKS-I/II:    "polyketide synthase", "pks", "ketosynthase"       ║
║    ├── Terpene:     "terpene", "geranyl", "sesquiterpene"              ║
║    ├── Siderophore: "siderophore", "enterobactin", "aerobactin"        ║
║    └── RiPP:        "lasso peptide", "bacteriocin", "microcin"        ║
║                                                                         ║
║  PyTorch model:  BGCClassifier  (3-layer MLP)                          ║
║    Input  → 128 → 64 → 8 logits                                        ║
║    Classes: NRPS | PKS-I | PKS-II | Hybrid | Terpene |                ║
║             Siderophore | RiPP | Non-BGC                               ║
║    (mock weights — replace with trained checkpoint for production)     ║
║                                                                         ║
║  Thresholding:  confidence ≥ 0.25  →  BGC hit                         ║
║                 confidence ≥ 0.70  →  high-confidence hit              ║
║                                                                         ║
║  Outputs:  BGCResult dataclass                                          ║
║            phase3/bgc_class_distribution.png                           ║
║            phase3/bgc_heatmap.png          (strain × BGC class)        ║
║            phase3/bgc_confidence_landscape.png                         ║
║            phase3/phase3_report.html                                   ║
║            phase3/prediction_matrix.csv                                ║
╚═════════════════════════════════════════════════════════════════════════╝
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FINAL OUTPUT                                      │
│  output/                                                                │
│  ├── phase1_presence_absence_matrix.csv                                 │
│  ├── phase1/                                                            │
│  │   ├── presence_absence_heatmap.png                                   │
│  │   ├── pangenome_summary.png                                          │
│  │   └── phase1_report.html                                             │
│  ├── phase2/                                                            │
│  │   ├── genomic_island_<strain>.png  (×N strains)                     │
│  │   ├── hgt_feature_distributions.png                                  │
│  │   └── phase2_report.html                                             │
│  └── phase3/                                                            │
│      ├── bgc_class_distribution.png                                     │
│      ├── bgc_heatmap.png                                                │
│      ├── bgc_confidence_landscape.png                                   │
│      ├── phase3_report.html                                             │
│      └── prediction_matrix.csv                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Pangenome Miner

**Module:** `pipeline/pangenome_miner.py`

### What it does

Phase 1 converts a collection of genome assemblies into a structured pangenome. It answers: *"Which genes are shared by all strains (core), which are strain-variable (shell), and which are rare/unique (accessory)?"*

### Algorithm detail

| Step | Description |
|------|-------------|
| **GFF3 parsing** | Reads every `CDS` feature from each `.gff`/`.gff3` file. Extracts `gene_id` (from `ID=`, `Name=`, or `locus_tag=` attributes), genomic coordinates, strand, and `product` annotation. |
| **FASTA ingestion** | Loads the paired `.fna`/`.fasta`/`.fa` file. Slices out the DNA sequence for each CDS using the GFF3 coordinates. Stores in `_fasta_store[strain_id][contig_id]`. |
| **K-mer Jaccard clustering** | Computes 4-mer (tetranucleotide) frequency sets for every gene. Two genes are placed in the same ortholog cluster if their Jaccard similarity ≥ `--identity`. Uses single-linkage agglomeration in O(n²) — adequate for small strain panels; replace with MMseqs2 subprocess for >20 strains. |
| **Presence/absence matrix** | A boolean DataFrame `[n_clusters × n_strains]`. Each cell is `True` if the cluster has a representative in that strain. |
| **Pangenome partition** | Core ≥ 95% · Shell between thresholds · Accessory ≤ max(10%, 1/n_strains). The adaptive floor prevents an impossible threshold with small strain panels (e.g. 10% of 5 strains = 0.5 → auto-raised to 20%). |

### Key classes

```python
PangenomeMiner(
    core_threshold=0.95,       # fraction of strains → core
    accessory_threshold=0.10,  # max fraction → accessory
    identity_threshold=0.80,   # Jaccard cutoff for ortholog clustering
)
result: PangenomeResult = miner.run(genomes_dir="...", annotations_dir="...")
```

---

## Phase 2 — HGT Detective

**Module:** `pipeline/hgt_detective.py`

### What it does

Phase 2 screens every accessory gene for signatures of **horizontal gene transfer (HGT)**. It combines nucleotide composition analysis with Mobile Genetic Element (MGE) proximity and unsupervised anomaly detection to flag "alien" genes that likely arrived via lateral transfer.

### Algorithm detail

| Step | Description |
|------|-------------|
| **Host profile** | For each strain, compute the mean GC content and mean tetranucleotide frequency across all contigs. This represents the "native" genomic signature. |
| **GC deviation** | `|gene_GC − host_GC| / host_GC`. High values (>5%) indicate foreign origin. |
| **K-mer deviation** | `1 − cosine_similarity(gene_4mer_vector, host_4mer_vector)`. Captures compositional mismatch beyond GC alone. |
| **MGE proximity** | Scans all annotated genes for MGE keywords (transposase, integrase, IS element, phage, recombinase, resolvase). Any gene within **10 kbp** of an MGE is flagged `mge_proximity=True`. |
| **Isolation Forest** | Fits an `IsolationForest` (scikit-learn) on the 4-feature matrix `[gc_dev, kmer_dev, mge_prox, gc_content]` with `contamination=0.30`. Genes with label `−1` (top 30% anomalies) are marked `is_hgt=True`. Fallback: if the variance is near-zero, all top-30% by GC deviation are flagged instead. |
| **Evidence assembly** | Each flagged gene gets a human-readable evidence list: "High GC deviation", "High k-mer deviation", "MGE proximal", "Isolation Forest anomaly". |

### Key classes

```python
HGTDetective(
    contamination=0.30,    # fraction of genes expected to be alien
    n_estimators=200,      # Isolation Forest trees
    random_state=42,
    min_seq_length=90,     # skip genes shorter than this (bp)
)
hgt_result: HGTResult = detective.run(phase1_result, fasta_store)
```

`alien_records` — the subset of HGT-flagged records — is the sole input to Phase 3.

---

## Phase 3 — AI BGC Predictor

**Module:** `pipeline/bgc_predictor.py`

### What it does

Phase 3 asks: *"Among the alien genes found by Phase 2, which ones are part of a Biosynthetic Gene Cluster?"* It uses a PyTorch MLP classifier combined with product-annotation keyword boosting to assign each alien gene a BGC class and confidence score.

### BGC classes

| Class | Description |
|-------|-------------|
| **NRPS** | Non-Ribosomal Peptide Synthetase — produces peptide natural products |
| **PKS-I** | Modular Type I Polyketide Synthase — produces macrolides, rapamycin-like |
| **PKS-II** | Iterative Type II PKS — produces aromatic polyketides, tetracyclines |
| **Hybrid** | NRPS–PKS hybrid clusters |
| **Terpene** | Terpenoid biosynthesis — produces geosmin, hopanoids |
| **Siderophore** | Iron-chelating compounds — desferrioxamine-type |
| **RiPP** | Ribosomally synthesised & Post-translationally modified Peptides — lasso peptides, bacteriocins |
| **Non-BGC** | No BGC signature detected |

### Algorithm detail

| Step | Description |
|------|-------------|
| **Feature engineering** | 8-dimensional vector per alien gene: `[gc_content, gc_deviation, kmer_deviation, anomaly_score, gene_length_norm, mge_proximity, multimodule_keyword_flag, regulatory_keyword_flag]`. |
| **Keyword boost** | Scans `product` field for domain-specific keywords (e.g. "adenylation domain", "ketosynthase", "terpene cyclase"). Boosts the matching class logit by +2.0 before softmax. |
| **PyTorch MLP** | `BGCClassifier`: `Linear(8→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→8)`. Weights currently mock-random; replace `models/bgc_classifier.pth` with a trained checkpoint for production. |
| **Thresholding** | Confidence ≥ 0.25 → BGC hit · Confidence ≥ 0.70 → high-confidence hit. |
| **NumPy fallback** | If PyTorch is unavailable, uses a pure-NumPy MLP with identical topology and the same mock weights. |

### Key classes

```python
BGCPredictor(
    seed=42,
    min_confidence=0.25,      # minimum softmax score to call a BGC hit
    use_keyword_boost=True,   # boost logits from product annotation keywords
    model_path=None,          # path to trained .pth weights (None = mock)
)
bgc_result: BGCResult = predictor.run(hgt_result)
```

---

## Output Files

```
output/
├── phase1_presence_absence_matrix.csv   # boolean clusters × strains table
├── phase1/
│   ├── presence_absence_heatmap.png     # seaborn clustermap with dendrogram
│   ├── pangenome_summary.png            # stacked bar (core/shell/accessory) + stats
│   └── phase1_report.html              # self-contained HTML report
├── phase2/
│   ├── genomic_island_<strain>.png      # linear map of each strain's alien genes
│   ├── hgt_feature_distributions.png   # 4-panel feature distribution plots
│   └── phase2_report.html              # HTML report with HGT table + plots
└── phase3/
    ├── bgc_class_distribution.png       # bar chart of BGC class counts
    ├── bgc_heatmap.png                  # per-strain × BGC class heatmap
    ├── bgc_confidence_landscape.png     # anomaly score vs BGC confidence scatter
    ├── phase3_report.html              # HTML report with predictions table
    └── prediction_matrix.csv           # full per-gene prediction scores
```

---

## Installation

### Prerequisites

- Python 3.10+
- conda (recommended) or pip

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd proj_6_magsanalysis

# 2. Create a conda environment (recommended)
conda create -n panadapt python=3.10
conda activate panadapt

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Install MMseqs2 for faster ortholog clustering at scale
conda install -c bioconda mmseqs2
```

---

## Usage & Examples

### Minimal run — real genome data

```bash
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     output/
```

### Full run — all parameters specified

```bash
python main.py \
  --genomes              tests/real_data/genomes \
  --annotations          tests/real_data/annotations \
  --output               output/ \
  --core-threshold       0.95 \
  --accessory-threshold  0.10 \
  --identity             0.80 \
  --verbose
```

### Quick smoke test — synthetic mock data

```bash
python main.py --mock --output output/mock_run/
```

### Running on your own MAGs

```bash
# Expected input layout:
#   genomes/
#     strain_1.fna
#     strain_2.fna
#     strain_3.fna
#   annotations/
#     strain_1.gff     ← GFF3 from NCBI PGAP or Prokka
#     strain_2.gff
#     strain_3.gff

python main.py \
  --genomes     /path/to/my_mags/genomes \
  --annotations /path/to/my_mags/annotations \
  --output      /path/to/results \
  --identity    0.75 \
  --verbose
```

### Replicating the paper test run (5 Streptomyces abikoensis soil MAGs)

```bash
python main.py \
  --genomes     tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output      tests/output \
  --identity    0.80 \
  --verbose
```

Expected runtime: ~4–6 minutes (Phase 1 k-mer clustering dominates; 5 strains × ~6,500 CDS each).

### CLI reference

```
usage: panadapt-bgc-miner [-h]
       [--genomes GENOMES] [--annotations ANNOTATIONS]
       [--output OUTPUT]
       [--core-threshold CORE_THRESHOLD]
       [--accessory-threshold ACCESSORY_THRESHOLD]
       [--identity IDENTITY]
       [--mock] [--verbose]

options:
  --genomes              Directory of FASTA genome files (.fna/.fasta/.fa)
  --annotations          Directory of GFF3 annotation files (.gff/.gff3)
  --output               Root output directory  [default: output/]
  --core-threshold       Fraction of strains for Core genes  [default: 0.95]
  --accessory-threshold  Max fraction of strains for Accessory  [default: 0.10]
  --identity             Ortholog clustering Jaccard threshold  [default: 0.80]
  --mock                 Use auto-generated synthetic data (ignores --genomes/--annotations)
  --verbose, -v          Enable DEBUG logging
```

---

## Project Structure

```
proj_6_magsanalysis/
├── main.py                        # Pipeline entry point & CLI
├── requirements.txt
├── README.md
├── .gitignore
│
├── pipeline/
│   ├── __init__.py
│   ├── pangenome_miner.py         # Phase 1 — ortholog clustering & pangenome partition
│   ├── phase1_visualizer.py       # Phase 1 — heatmap, summary plot, HTML report
│   ├── hgt_detective.py           # Phase 2 — HGT scoring & Isolation Forest
│   ├── phase2_visualizer.py       # Phase 2 — genomic island map, distributions, HTML
│   ├── bgc_predictor.py           # Phase 3 — PyTorch BGC classifier
│   └── phase3_visualizer.py       # Phase 3 — class distribution, heatmap, HTML
│
├── models/
│   └── bgc_classifier.pth         # (trained weights go here; currently empty)
│
├── tests/
│   ├── test_pangenome_miner.py    # 18 Phase 1 unit tests
│   ├── test_hgt_detective.py      # 32 Phase 2 unit tests
│   └── real_data/
│       ├── genomes/               # 5 × Streptomyces abikoensis .fna files
│       └── annotations/           # 5 × matching NCBI PGAP .gff files
│
├── mock_data/
│   └── generate_mock_data.py      # Synthetic genome generator (5-strain)
│
├── utils/                         # Utility helpers (future use)
└── output/                        # Pipeline output (git-ignored)
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run Phase 1 tests only
python -m pytest tests/test_pangenome_miner.py -v

# Run Phase 2 tests only
python -m pytest tests/test_hgt_detective.py -v

# Run with coverage report
python -m pytest tests/ --cov=pipeline --cov-report=term-missing
```

Current test status: **50/50 passing** (18 Phase 1 + 32 Phase 2).
