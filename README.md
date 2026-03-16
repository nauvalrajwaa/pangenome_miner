# PanAdapt-BGC Miner

**Pangenome Adaptive & Alien Biosynthetic Gene Cluster Miner**

A modular, end-to-end Python pipeline for discovering Biosynthetic Gene Clusters (BGCs) in microbial genomes by chaining three analytical phases: **Pangenome Analysis → HGT Detection → AI BGC Prediction**. Designed for soil MAGs and other environmental metagenome-assembled genomes.

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Flowchart](#workflow-flowchart)
3. [Phase 1 — Pangenome Miner](#phase-1--pangenome-miner)
4. [Phase 2 — HGT Detective](#phase-2--hgt-detective)
5. [Phase 3 — AI BGC Predictor (BGC-Prophet)](#phase-3--ai-bgc-predictor-bgc-prophet)
6. [ESM2 Model Selection](#esm2-model-selection)
7. [Output Files](#output-files)
8. [Installation](#installation)
9. [Usage & Examples](#usage--examples)
10. [Project Structure](#project-structure)
11. [Testing](#testing)

---

## Overview

PanAdapt-BGC Miner answers one central question:

> **Which genes in an accessory/adaptive genome are horizontally transferred biosynthetic gene clusters of biotechnological interest?**

It ingests raw genome assemblies (FASTA) and their NCBI PGAP or Prokka annotations (GFF3), then pipes them through three phases, each building on the previous one. Final output is a set of publication-ready plots and self-contained HTML reports — one per phase.

### Key Features

- **BGC-Prophet integration**: Trained TransformerEncoder models (annotator + classifier) with ESM2 protein language model embeddings for biologically meaningful BGC predictions
- **Flexible ESM2 model selection**: Choose from 6 ESM2 variants (8M to 15B parameters) to trade off speed vs. embedding quality
- **Automatic fallback**: Graceful degradation to mock inference when BGC-Prophet or ESM2 is unavailable
- **Three-phase pipeline**: Pangenome → HGT → BGC, each phase producing independent visualizations and reports
- **Publication-ready output**: Self-contained HTML reports, heatmaps, distribution plots, and CSV prediction matrices

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
║  PHASE 3 — BGC-Prophet AI Predictor (bgc_predictor.py)                 ║
║                                                                         ║
║  Input: HGT-flagged alien gene records from Phase 2                    ║
║                                                                         ║
║  Step 1 — Protein translation:                                          ║
║    └── CDS DNA → amino acid sequences (BioPython Seq.translate)        ║
║                                                                         ║
║  Step 2 — ESM2 protein embeddings:                                      ║
║    └── ESM2 model (configurable, default: esm2_t6_8M_UR50D)           ║
║        Per-protein mean-pooled embedding (320-dim after projection)     ║
║                                                                         ║
║  Step 3 — BGC-Prophet Annotator (TransformerEncoder):                   ║
║    └── 128-gene windows → per-gene BGC probability (sigmoid)           ║
║        Genes with P(BGC) ≥ 0.50 → flagged as biosynthetic              ║
║                                                                         ║
║  Step 4 — BGC-Prophet Classifier (TransformerEncoder):                  ║
║    └── 7-class sigmoid classification of BGC-positive windows          ║
║        Classes: Alkaloid | Terpene | NRP | Polyketide | RiPP |         ║
║                 Saccharide | Other                                      ║
║                                                                         ║
║  Step 5 — Keyword boost (optional):                                     ║
║    └── Product annotation keywords boost matching class scores          ║
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
| **GC deviation** | `\|gene_GC − host_GC\| / host_GC`. High values (>5%) indicate foreign origin. |
| **K-mer deviation** | `1 − cosine_similarity(gene_4mer_vector, host_4mer_vector)`. Captures compositional mismatch beyond GC alone. |
| **MGE proximity** | Scans all annotated genes for 25+ MGE keywords (transposase, integrase, IS element, phage, recombinase, resolvase, conjugative, mobilization, CRISPR-associated, etc.). Any gene within **10 kbp** of an MGE is flagged `mge_proximity=True`. |
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

## Phase 3 — AI BGC Predictor (BGC-Prophet)

**Module:** `pipeline/bgc_predictor.py`

### What it does

Phase 3 asks: *"Among the alien genes found by Phase 2, which ones are part of a Biosynthetic Gene Cluster?"* It uses the **BGC-Prophet** trained model — a TransformerEncoder architecture powered by ESM2 protein language model embeddings — to perform gene-level BGC annotation and type classification.

### BGC-Prophet Model

[BGC-Prophet](https://github.com/HUST-NingKang-Lab/BGC-Prophet) (MIT License) uses two trained TransformerEncoder models:

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| **Annotator** | PositionalEncoding → TransformerEncoder (2 layers, 5 heads) → per-position MLP → Sigmoid | 128-gene windows of 320-dim embeddings | Per-gene BGC probability |
| **Classifier** | PositionalEncoding → TransformerEncoder (2 layers, 5 heads) → mean pool → Linear(320→7) → Sigmoid | 128-gene windows + padding mask | 7-class BGC type probabilities |

### BGC Classes

| Class | Description |
|-------|-------------|
| **Alkaloid** | Alkaloid biosynthesis clusters |
| **Terpene** | Terpenoid biosynthesis — geosmin, hopanoids |
| **NRP** | Non-Ribosomal Peptide — produced by NRPS enzymes |
| **Polyketide** | Polyketide biosynthesis — macrolides, Type I/II PKS |
| **RiPP** | Ribosomally synthesised & Post-translationally modified Peptides — lasso peptides, bacteriocins |
| **Saccharide** | Saccharide/sugar-based natural product clusters |
| **Other** | Other BGC types not in the above categories |
| **NonBGC** | No BGC signature detected |

### Pipeline Flow

```
CDS DNA sequences (from Phase 1)
        │
        ▼
┌─────────────────────────────┐
│  BioPython Seq.translate()  │  DNA → Protein
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  ESM2 Protein Embeddings    │  Per-protein → 320-dim vector
│  (configurable model)       │  (auto-projection if dim ≠ 320)
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  128-Gene Windowing         │  Group by contig, pad/split
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  BGC-Prophet Annotator      │  Per-gene: BGC or Non-BGC
│  (TransformerEncoder)       │  threshold ≥ 0.50
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  BGC-Prophet Classifier     │  Per-window: 7-class BGC type
│  (TransformerEncoder)       │  + keyword boost scoring
└────────────┬────────────────┘
             ▼
   BGCGeneRecord per gene
   BGCResult with statistics
```

### Key classes

```python
BGCPredictor(
    seed=42,
    min_confidence=0.25,             # minimum score to call a BGC hit
    use_keyword_boost=True,          # boost scores from product annotation keywords
    model_dir=None,                  # path to BGC-Prophet weights (auto-detected)
    esm_model_name="esm2_t6_8M_UR50D",  # ESM2 model variant (see table below)
)
bgc_result: BGCResult = predictor.run(hgt_result)
```

### Fallback Mode

When BGC-Prophet dependencies are unavailable (`fair-esm`, model weights), the predictor gracefully falls back to a mock MLP with random weights. This allows the pipeline structure to run without the trained model, but predictions will not be biologically meaningful.

---

## ESM2 Model Selection

The ESM2 protein language model used for embedding extraction is **configurable at runtime** via the `--esm-model` argument. Larger models produce richer protein representations but require more memory and compute time.

| Model Name | Layers | Embed Dim | BGC-Prophet d_model | Approx. Size | Notes |
|------------|--------|-----------|---------------------|-------------|-------|
| `esm2_t6_8M_UR50D` | 6 | 320 | 320 | ~30 MB | **Default.** Fastest; good for most use cases. |
| `esm2_t12_35M_UR50D` | 12 | 480 | 480 | ~140 MB | Moderate upgrade. Needs model-specific weights. |
| `esm2_t30_150M_UR50D` | 30 | 640 | 640 | ~600 MB | Richer embeddings. ~2 GB RAM. |
| `esm2_t33_650M_UR50D` | 33 | 1280 | 1280 | ~2.5 GB | Large model. GPU recommended. |
| `esm2_t36_3B_UR50D` | 36 | 2560 | 2560 | ~11 GB | Very large. GPU required. |
| `esm2_t48_15B_UR50D` | 48 | 5120 | 5120 | ~60 GB | Largest ESM2. Multi-GPU recommended. |

### Architecture — No PCA, Native Dimensions

**PCA projection is never used.** Each ESM2 model uses its own BGC-Prophet weights
at the model's native embedding dimension (`d_model`).

| Model | Embed dim | nhead | Weight source |
|-------|-----------|-------|---------------|
| `esm2_t6_8M_UR50D` (default) | 320 | 5 | Official upstream `models/model/` root |
| `esm2_t12_35M_UR50D` | 480 | 5 | User-trained/seeded subfolder |
| `esm2_t30_150M_UR50D` | 640 | 5 | User-trained/seeded subfolder |
| `esm2_t33_650M_UR50D` | 1280 | 5 | User-trained/seeded subfolder |

The default 8M model is loaded from `models/model/annotator.pt` and `classifier.pt`
(the official BGC-Prophet pre-trained weights, `nhead=5`). Larger models are loaded
from `models/model/{esm_model_name}/` and use `nhead=5`.

### Notes

- **8M default**: uses the official pre-trained weights shipped with this repo.
  No training or seeding required.
- **Larger models (35M/150M/650M)**: require dedicated weights at
  `models/model/{esm_model_name}/`. Seed in ~30s or train on real MIBiG data.
- ESM2 model weights (the protein language model) are auto-downloaded from the
  `fair-esm` package on first use and cached locally.

### Usage

```bash
# Default ESM2-8M (recommended)
python main.py --genomes data/genomes --annotations data/annotations --output output/

# Use a larger model (35M parameters)
python main.py --genomes data/genomes --annotations data/annotations --output output/ \
  --esm-model esm2_t12_35M_UR50D

# Use 150M parameter model (needs ~2 GB RAM)
python main.py --genomes data/genomes --annotations data/annotations --output output/ \
  --esm-model esm2_t30_150M_UR50D
```

### Programmatic Access

```python
from pipeline.bgc_predictor import BGCPredictor, ESM2_REGISTRY

# List all supported models
for name, spec in ESM2_REGISTRY.items():
    print(f"{name}: {spec['layers']} layers, {spec['embed_dim']}d, ~{spec['params']} params")

# Use a specific model
predictor = BGCPredictor(esm_model_name="esm2_t12_35M_UR50D")
result = predictor.run(hgt_result)
```

---

## Training Custom BGC-Prophet Models

**The default ESM2-8M model ships with official pre-trained weights** — no training
needed to run the pipeline out of the box:
```bash
python main.py --genomes data/genomes --annotations data/annotations --output output/
```

Training is **only needed for larger models** (35M, 150M, 650M), which require
their own BGC-Prophet weights at the native embedding dimension.

| ESM2 Model | Embed Dim | nhead | Needs training? |
|------------|-----------|-------|-----------------|
| `esm2_t6_8M_UR50D` (default) | 320 | 3 | **No** — official weights included |
| `esm2_t12_35M_UR50D` | 480 | 5 | Yes — seed or train |
| `esm2_t30_150M_UR50D` | 640 | 5 | Yes — seed or train |
| `esm2_t33_650M_UR50D` | 1280 | 5 | Yes — seed or train |
### Prerequisites

- Python 3.9+
- GPU recommended (NVIDIA with CUDA, or Apple Silicon with MPS)
- Internet connection (downloads [MIBiG v3.1](https://mibig.secondarymetabolites.org/) on first run)
- ~2 GB disk space for MIBiG data + ESM2 embeddings cache

### Seed Weights Instantly (No Download Required)

Generate valid model weights for 35M/150M/650M in ~30 seconds on CPU/MPS/GPU.
Uses 100 synthetic training windows — no MIBiG download, no internet required.

```bash
# Seed weights for 35M (480-dim)
python scripts/seed_weights.py --model esm2_t12_35M_UR50D

# Seed weights for 150M (640-dim)
python scripts/seed_weights.py --model esm2_t30_150M_UR50D

# Seed weights for 650M (1280-dim)
python scripts/seed_weights.py --model esm2_t33_650M_UR50D
```

Each command writes to `models/model/{esm_model_name}/annotator.pt` and `classifier.pt`.

Each command writes to `models/model/{esm_model_name}/annotator.pt` and `classifier.pt`.

Then run the full pipeline with any model:

```bash
# 8M (default, fastest) — no weights needed, official weights included
python main.py --genomes data/genomes --annotations data/annotations --output output/

# 35M (seed first with seed_weights.py)
python main.py --genomes data/genomes --annotations data/annotations --output output/ \
  --esm-model esm2_t12_35M_UR50D

# 150M
python main.py --genomes data/genomes --annotations data/annotations --output output/ \
  --esm-model esm2_t30_150M_UR50D

# 650M
python main.py --genomes data/genomes --annotations data/annotations --output output/ \
  --esm-model esm2_t33_650M_UR50D
```

### Quick Start (Full MIBiG Training)

```bash
# Train for a specific larger ESM2 model (downloads MIBiG v3.1 on first run)
python train_prophet.py --esm-model esm2_t12_35M_UR50D --device auto
```

To iterate quickly on a small slice of MIBiG data:

```bash
# Use --max-entries to cap the number of BGC entries (e.g. 50 for a smoke-test)
python train_prophet.py --esm-model esm2_t12_35M_UR50D --max-entries 50 --epochs 5
```

### Train All Larger Models

```bash
for model in esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D; do
    echo "=== Training $model ==="
    python train_prophet.py --esm-model "$model" --epochs 50 --device auto
done
```

### Google Colab

```python
!pip install fair-esm bgc-prophet
!python train_prophet.py --esm-model esm2_t30_150M_UR50D --device auto --epochs 50
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--esm-model` | `esm2_t6_8M_UR50D` | ESM2 model variant to train for |
| `--data-dir` | `data/training` | Directory for MIBiG data and embedding cache |
| `--output-dir` | `models/model` | Output directory for trained weights |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `16` | Training batch size |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--device` | `auto` | Device selection (`auto` / `cuda` / `mps` / `cpu`) |
| `--val-split` | `0.1` | Fraction of data used for validation |
| `--num-negatives` | `2000` | Number of synthetic negative training examples |
| `--seed` | `42` | Random seed for reproducibility |
| `--max-entries` | `0` | Limit MIBiG entries used (0 = all). Use for fast smoke-tests, e.g. `--max-entries 50` |

### Output Structure

```
models/model/
├── annotator.pt                    # Official BGC-Prophet 8M weights (nhead=5, 320-dim)
├── classifier.pt                   # Official BGC-Prophet 8M weights (nhead=5, 320-dim)
├── esm2_t12_35M_UR50D/             # User-trained 35M weights (nhead=5, 480-dim)
│   ├── annotator.pt
│   ├── classifier.pt
│   └── training_meta.json
├── esm2_t30_150M_UR50D/            # User-trained 150M weights (nhead=5, 640-dim)
│   ├── annotator.pt
│   ├── classifier.pt
│   └── training_meta.json
└── esm2_t33_650M_UR50D/            # User-trained 650M weights (nhead=5, 1280-dim)
    ├── annotator.pt
    ├── classifier.pt
    └── training_meta.json
```

### Using Trained Models

After seeding or training, the pipeline auto-detects weights — no configuration needed.
The 8M default always uses the official root-level weights (no subfolder).
Larger models are loaded from their dedicated subfolder:
```
8M:   models/model/annotator.pt + classifier.pt  (nhead=5, official)
35M:  models/model/esm2_t12_35M_UR50D/           (nhead=5, user-trained)
150M: models/model/esm2_t30_150M_UR50D/          (nhead=5, user-trained)
650M: models/model/esm2_t33_650M_UR50D/          (nhead=5, user-trained)
```
- No PCA projection — ever
- Missing weights raise a clear error with the exact command to run


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
    └── prediction_matrix.csv           # full per-gene prediction scores (8 classes)
```

---

## Installation

### Prerequisites

- Python 3.10+
- conda (recommended) or pip
- ~200 MB disk for model weights (BGC-Prophet + ESM2-8M)

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

# 4. Download BGC-Prophet model weights
mkdir -p models/model
wget -q "https://github.com/HUST-NingKang-Lab/BGC-Prophet/files/12733164/model.tar.gz" \
  -O models/model.tar.gz
tar -xzf models/model.tar.gz -C models/
rm models/model.tar.gz

# 5. (Optional) Install MMseqs2 for faster ortholog clustering at scale
conda install -c bioconda mmseqs2
```

### Verify Installation

```bash
# Check all dependencies
python -c "
from pipeline.bgc_predictor import BGCPredictor, ESM2_REGISTRY, _PROPHET_AVAILABLE
print(f'BGC-Prophet available: {_PROPHET_AVAILABLE}')
print(f'Supported ESM2 models: {list(ESM2_REGISTRY.keys())}')
"

# Run tests
python -m pytest tests/ -v
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `fair-esm` | ≥2.0.0 | ESM2 protein language models |
| `bgc-prophet` | ≥0.1.2 | BGC-Prophet annotator/classifier models |
| `lmdb` | ≥1.4.0 | Required by bgc-prophet internals |
| `biopython` | ≥1.80 | GFF/FASTA parsing, DNA→protein translation |
| `scikit-learn` | ≥1.2 | Isolation Forest (Phase 2) |
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical operations |
| `matplotlib` | ≥3.6 | Plotting |
| `seaborn` | ≥0.12 | Statistical plots |
| `plotly` | ≥5.0 | Interactive plots |
| `jinja2` | ≥3.1 | HTML report templating |
| `loguru` | ≥0.7 | Structured logging |
| `tqdm` | ≥4.60 | Progress bars |

---

## Usage & Examples

### Quick start — bundled test data (5 Streptomyces abikoensis MAGs)

```bash
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     tests/output \
  --verbose
```

Expected output:
```
═══ Phase 1: Pangenome Analysis ═══
  Strains analyzed       : 5
  Total CDS genes        : ~32,000
  Ortholog clusters      : ~5,050
  Accessory genes        : ~2,021

═══ Phase 2: HGT Detection ═══
  Genes screened         : ~2,021
  Alien HGT genes        : ~606 (~30%)

═══ Phase 3: BGC Prediction ═══
  Alien genes scored     : ~606
  BGC hits               : ~43 (~7%)
  High-confidence hits   : ~27
  Top BGC class          : RiPP
  Inference engine       : BGC-Prophet
  ESM2 model             : esm2_t6_8M_UR50D
```

Expected runtime: ~4–6 minutes (Phase 1 k-mer clustering dominates; Phase 3 ESM2 embedding ~30 seconds).

### Minimal run

```bash
python main.py \
  --genomes    data/genomes \
  --annotations data/annotations \
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
  --model-dir            models/model \
  --esm-model            esm2_t6_8M_UR50D \
  --verbose
```

### Using a larger ESM2 model

```bash
# 35M parameter model — richer embeddings, ~4x slower
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     output_35M/ \
  --esm-model  esm2_t12_35M_UR50D \
  --verbose

# 150M parameter model — needs ~2 GB RAM
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     output_150M/ \
  --esm-model  esm2_t30_150M_UR50D \
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

### CLI Reference

```
usage: panadapt-bgc-miner [-h]
       [--genomes GENOMES] [--annotations ANNOTATIONS]
       [--output OUTPUT]
       [--core-threshold CORE_THRESHOLD]
       [--accessory-threshold ACCESSORY_THRESHOLD]
       [--identity IDENTITY]
       [--model-dir MODEL_DIR]
       [--esm-model ESM_MODEL]
       [--mock] [--verbose]

options:
  --genomes              Directory of FASTA genome files (.fna/.fasta/.fa)
  --annotations          Directory of GFF3 annotation files (.gff/.gff3)
  --output               Root output directory  [default: output/]
  --core-threshold       Fraction of strains for Core genes  [default: 0.95]
  --accessory-threshold  Max fraction of strains for Accessory  [default: 0.10]
  --identity             Ortholog clustering Jaccard threshold  [default: 0.80]
  --model-dir            Directory with BGC-Prophet model weights  [default: models/model/]
  --esm-model            ESM2 model variant for protein embeddings  [default: esm2_t6_8M_UR50D]
                         Options: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D,
                         esm2_t30_150M_UR50D, esm2_t33_650M_UR50D,
                         esm2_t36_3B_UR50D, esm2_t48_15B_UR50D
  --mock                 Use auto-generated synthetic data (ignores --genomes/--annotations)
  --verbose, -v          Enable DEBUG logging
```

---

## Project Structure

```
proj_6_magsanalysis/
├── main.py                        # Pipeline entry point & CLI
├── requirements.txt               # Python dependencies
├── README.md
├── .gitignore
│
├── pipeline/
│   ├── __init__.py
│   ├── pangenome_miner.py         # Phase 1 — ortholog clustering & pangenome partition
│   ├── phase1_visualizer.py       # Phase 1 — heatmap, summary plot, HTML report
│   ├── hgt_detective.py           # Phase 2 — HGT scoring & Isolation Forest
│   ├── phase2_visualizer.py       # Phase 2 — genomic island map, distributions, HTML
│   ├── bgc_predictor.py           # Phase 3 — BGC-Prophet AI predictor (ESM2 + TransformerEncoder)
│   └── phase3_visualizer.py       # Phase 3 — class distribution, heatmap, HTML
│
├── models/
│   └── model/
│       ├── annotator.pt           # BGC-Prophet annotator weights (~10 MB)
│       └── classifier.pt          # BGC-Prophet classifier weights (~10 MB)
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

---

## Acknowledgments

- **BGC-Prophet** — [HUST-NingKang-Lab/BGC-Prophet](https://github.com/HUST-NingKang-Lab/BGC-Prophet) (MIT License) for the trained BGC annotator and classifier models
- **ESM2** — [facebookresearch/esm](https://github.com/facebookresearch/esm) for protein language models
- Test data: 5 *Streptomyces abikoensis* soil MAG assemblies from NCBI
