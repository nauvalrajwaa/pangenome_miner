"""
bgc_predictor.py — Phase 3: AI-assisted BGC Type Prediction

Implements a PyTorch-based skeleton/mock-inference pipeline for scoring
HGT-flagged (Alien_HGT) genes for Biosynthetic Gene Cluster (BGC) activity,
with particular focus on NRPS (Non-Ribosomal Peptide Synthetase) and
PKS (Polyketide Synthase) biosynthetic classes.

Architecture
------------
  • BGCFeatureExtractor  – converts HGTGeneRecord → feature tensor
  • BGCClassifier        – lightweight 4-layer MLP (mock weights, no training)
  • BGCPredictor         – orchestrator: batch-scores alien_records, returns BGCResult
  • BGCGeneRecord        – per-gene dataclass extending HGTGeneRecord scores
  • BGCResult            – full Phase 3 output dataclass

Mock-Inference Design
---------------------
The model is intentionally untrained ("skeleton" per spec). Weights are
initialised with a reproducible seed so output scores are deterministic and
biologically plausible-looking.  In a production system these weights would
be replaced by a trained checkpoint (.pt file).

BGC Classes modelled
--------------------
  0  Non-BGC     – no biosynthetic signal
  1  NRPS        – Non-Ribosomal Peptide Synthetase
  2  PKS-I       – Type I Polyketide Synthase (modular)
  3  PKS-II      – Type II Polyketide Synthase (iterative)
  4  Terpene     – Terpene synthase / cyclase cluster
  5  RiPP        – Ribosomally synthesised and Post-translationally modified Peptides
  6  Other-BGC   – Siderophore, ectoine, nucleoside, etc.

Usage
-----
    from pipeline.bgc_predictor import BGCPredictor
    from pipeline.hgt_detective import HGTResult

    predictor = BGCPredictor(seed=42)
    bgc_result = predictor.run(hgt_result)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PyTorch import — gracefully degrade to numpy mock if unavailable
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logging.warning(
        "PyTorch not installed — BGCPredictor will use NumPy mock inference. "
        "Install with: pip install torch"
    )

from pipeline.hgt_detective import HGTGeneRecord, HGTResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BGC_CLASSES: List[str] = [
    "Non-BGC",
    "NRPS",
    "PKS-I",
    "PKS-II",
    "Terpene",
    "RiPP",
    "Other-BGC",
]

N_CLASSES = len(BGC_CLASSES)

# Feature vector dimensionality
# [gc_content, gc_deviation, kmer_deviation, anomaly_score,
#  mge_proximity, gene_length_log, upstream_gene_count, downstream_gene_count,
#  gc_bin_0..9 (10 bins), kmer_bin_0..9 (10 bins)]
N_FEATURES = 4 + 1 + 1 + 2 + 10 + 10  # = 28

# BGC confidence thresholds
HIGH_CONF_THRESHOLD = 0.65
MED_CONF_THRESHOLD  = 0.40

# Heuristic keyword → BGC class boosts (applied to mock scores)
# Maps product/note field keywords to (class_index, boost_weight)
BGC_KEYWORD_MAP: Dict[str, Tuple[int, float]] = {
    "nrps":           (1, 0.30),
    "non-ribosomal":  (1, 0.30),
    "condensation":   (1, 0.25),
    "adenylation":    (1, 0.25),
    "pks":            (2, 0.30),
    "polyketide":     (2, 0.25),
    "ketosynthase":   (2, 0.25),
    "acyltransferase":(2, 0.20),
    "type ii pks":    (3, 0.30),
    "cyclase":        (3, 0.20),
    "terpene":        (4, 0.35),
    "geranyl":        (4, 0.20),
    "sesquiterpene":  (4, 0.25),
    "lanthipeptide":  (5, 0.30),
    "ripp":           (5, 0.30),
    "bacteriocin":    (5, 0.25),
    "siderophore":    (6, 0.25),
    "ectoine":        (6, 0.25),
    "nucleoside":     (6, 0.20),
}


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class BGCGeneRecord:
    """Phase 3 output: an HGT-flagged gene annotated with BGC prediction."""

    hgt_record: HGTGeneRecord
    bgc_class: str              # Predicted BGC class label (e.g. "NRPS")
    bgc_class_idx: int          # Integer index into BGC_CLASSES
    confidence: float           # Softmax probability of predicted class [0,1]
    class_scores: List[float]   # Full softmax distribution over all BGC classes
    is_bgc: bool                # True if bgc_class != "Non-BGC" and confidence >= MED
    confidence_tier: str        # "High", "Medium", "Low"
    keyword_hits: List[str]     # Product/note keywords that boosted the score

    # Convenience passthrough accessors
    @property
    def gene_id(self) -> str:
        return self.hgt_record.gene_record.gene_id

    @property
    def strain_id(self) -> str:
        return self.hgt_record.gene_record.strain_id

    @property
    def contig_id(self) -> str:
        return self.hgt_record.gene_record.contig


@dataclass
class BGCResult:
    """Full Phase 3 output — passed to Phase 3 visualizer and final report."""

    bgc_records: List[BGCGeneRecord]        # All scored alien records
    bgc_hits: List[BGCGeneRecord]           # Subset: is_bgc=True
    class_distribution: Dict[str, int]      # {class_label: count}
    strain_bgc_counts: Dict[str, int]       # {strain_id: bgc_count}
    feature_matrix: pd.DataFrame            # genes × input features
    prediction_matrix: pd.DataFrame         # genes × class scores
    stats: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Feature extraction
# ===========================================================================

class BGCFeatureExtractor:
    """
    Converts a list of HGTGeneRecord objects into a normalised NumPy feature
    matrix suitable for the BGCClassifier.

    Features (N_FEATURES = 28 per gene):
        [0]  gc_content          – raw GC fraction
        [1]  gc_deviation        – normalised |gene_gc - host_gc|
        [2]  kmer_deviation      – tetranucleotide distance from host profile
        [3]  anomaly_score       – IsolationForest outlier score
        [4]  mge_proximity       – 1.0 if near MGE, else 0.0
        [5]  gene_length_log     – log10(gene_length_bp), 0 if unknown
        [6]  upstream_neighbors  – placeholder (0.0; extend in production)
        [7]  downstream_nbrs     – placeholder (0.0; extend in production)
        [8-17]  gc_bin_0..9     – histogram of GC across all records (context)
        [18-27] kmer_bin_0..9   – histogram of kmer_deviation (context)
    """

    def __init__(self) -> None:
        self._fitted = False
        self._gc_bins: np.ndarray = np.zeros(10)
        self._kmer_bins: np.ndarray = np.zeros(10)

    def fit_transform(self, records: List[HGTGeneRecord]) -> np.ndarray:
        """Fit context histograms and return (n_records, N_FEATURES) float32 array."""
        if not records:
            return np.zeros((0, N_FEATURES), dtype=np.float32)

        # --- Collect per-gene scalars ---
        gc_vals    = np.array([r.gc_content     for r in records], dtype=np.float32)
        gc_dev     = np.array([r.gc_deviation   for r in records], dtype=np.float32)
        kmer_dev   = np.array([r.kmer_deviation for r in records], dtype=np.float32)
        anom       = np.array([r.anomaly_score  for r in records], dtype=np.float32)
        mge_prox   = np.array([float(r.mge_proximity) for r in records], dtype=np.float32)

        # Gene length: derive from GeneRecord coordinates
        gene_len_log = np.array(
            [
                math.log10(max(abs(r.gene_record.end - r.gene_record.start) + 1, 1))
                for r in records
            ],
            dtype=np.float32,
        )

        # Placeholder neighborhood features
        upstream   = np.zeros(len(records), dtype=np.float32)
        downstream = np.zeros(len(records), dtype=np.float32)

        # --- Context histograms (same value for all records — population context) ---
        self._gc_bins   = np.histogram(gc_vals,  bins=10, range=(0.0, 1.0))[0].astype(np.float32)
        self._kmer_bins = np.histogram(kmer_dev, bins=10, range=(0.0, 1.0))[0].astype(np.float32)
        # Normalise to sum=1
        _norm = lambda x: x / (x.sum() + 1e-9)
        gc_hist   = _norm(self._gc_bins)
        kmer_hist = _norm(self._kmer_bins)

        # Clip deviations to [0,1] for numerical stability
        gc_dev   = np.clip(gc_dev,  0.0, 1.0)
        kmer_dev = np.clip(kmer_dev, 0.0, 1.0)
        # Normalise anomaly score to [0,1]
        anom_min, anom_max = anom.min(), anom.max()
        if anom_max > anom_min:
            anom_norm = (anom - anom_min) / (anom_max - anom_min)
        else:
            anom_norm = np.zeros_like(anom)

        # --- Assemble feature matrix ---
        per_gene = np.column_stack([
            gc_vals, gc_dev, kmer_dev, anom_norm,
            mge_prox, gene_len_log,
            upstream, downstream,
        ])  # shape (n, 8)

        # Broadcast histograms as context features (same row for all genes)
        n = len(records)
        gc_hist_mat   = np.tile(gc_hist,   (n, 1))  # (n, 10)
        kmer_hist_mat = np.tile(kmer_hist, (n, 1))  # (n, 10)

        X = np.concatenate([per_gene, gc_hist_mat, kmer_hist_mat], axis=1)  # (n, 28)
        self._fitted = True
        return X.astype(np.float32)


# ===========================================================================
# PyTorch Model (skeleton)
# ===========================================================================

if _TORCH_AVAILABLE:
    class BGCClassifier(nn.Module):
        """
        4-layer MLP for BGC-type classification.

        Architecture:
            FC(28 → 64) → LayerNorm → ReLU → Dropout(0.3)
            FC(64 → 128) → LayerNorm → ReLU → Dropout(0.3)
            FC(128 → 64) → LayerNorm → ReLU → Dropout(0.2)
            FC(64 → 7)   → Softmax

        This is a skeleton model: weights are randomly initialised with a
        fixed seed. In production, load a pre-trained checkpoint with
        BGCClassifier.load_state_dict(torch.load("bgc_model.pt")).
        """

        def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(p=0.3),

                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(p=0.2),

                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns softmax probabilities, shape (batch, N_CLASSES)."""
            logits = self.net(x)
            return F.softmax(logits, dim=-1)


def _build_torch_model(seed: int = 42) -> "BGCClassifier":
    """Construct BGCClassifier with deterministic mock weights."""
    torch.manual_seed(seed)
    model = BGCClassifier()
    model.eval()  # eval mode: dropout disabled
    return model


def _torch_inference(model: "BGCClassifier", X: np.ndarray) -> np.ndarray:
    """Run forward pass; return (n, N_CLASSES) float32 probability array."""
    with torch.no_grad():
        tensor = torch.from_numpy(X)
        probs  = model(tensor)
        return probs.numpy()


# ===========================================================================
# NumPy mock inference (fallback when PyTorch unavailable)
# ===========================================================================

def _numpy_mock_inference(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Deterministic NumPy mock of the MLP forward pass.
    Applies a fixed random linear projection followed by softmax.
    Produces plausible-looking probability distributions without PyTorch.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((N_FEATURES, 64)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((64, 128)).astype(np.float32) * 0.1
    W3 = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    W4 = rng.standard_normal((64, N_CLASSES)).astype(np.float32) * 0.1

    h = np.maximum(X @ W1, 0)
    h = np.maximum(h @ W2, 0)
    h = np.maximum(h @ W3, 0)
    logits = h @ W4

    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp    = np.exp(logits)
    probs  = exp / exp.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


# ===========================================================================
# Keyword boost (heuristic annotation signal)
# ===========================================================================

def _apply_keyword_boosts(
    records: List[HGTGeneRecord],
    probs: np.ndarray,
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    For each gene, scan the product/note fields for BGC keywords.
    Boost corresponding class logit before final re-normalisation.

    Returns
    -------
    boosted_probs : np.ndarray (n, N_CLASSES)
    keyword_hits  : List[List[str]] — per-gene keyword matches
    """
    boosted = probs.copy()
    all_hits: List[List[str]] = []

    for i, rec in enumerate(records):
        hits: List[str] = []
        product = (rec.gene_record.product or "").lower()
        text    = product

        for kw, (cls_idx, boost) in BGC_KEYWORD_MAP.items():
            if kw in text:
                boosted[i, cls_idx] += boost
                hits.append(kw)

        all_hits.append(hits)

    # Re-normalise rows to sum=1
    row_sums = boosted.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    boosted /= row_sums

    return boosted, all_hits


# ===========================================================================
# BGCPredictor orchestrator
# ===========================================================================

class BGCPredictor:
    """
    Phase 3 orchestrator: score HGT alien genes for BGC activity.

    Parameters
    ----------
    seed : int
        Random seed for reproducible mock weights.
    min_confidence : float
        Minimum confidence to flag a gene as is_bgc=True (default: MED_CONF_THRESHOLD).
    use_keyword_boost : bool
        Apply heuristic keyword boost from product/note annotations (default: True).

    Example
    -------
        predictor = BGCPredictor(seed=42)
        bgc_result = predictor.run(hgt_result)
        print(f"BGC hits: {len(bgc_result.bgc_hits)}")
    """

    def __init__(
        self,
        seed: int = 42,
        min_confidence: float = MED_CONF_THRESHOLD,
        use_keyword_boost: bool = True,
    ) -> None:
        self.seed = seed
        self.min_confidence = min_confidence
        self.use_keyword_boost = use_keyword_boost

        # Build model
        if _TORCH_AVAILABLE:
            self._model = _build_torch_model(seed=seed)
            logger.info("BGCPredictor: PyTorch BGCClassifier loaded (mock weights, seed=%d)", seed)
        else:
            self._model = None
            logger.info("BGCPredictor: NumPy mock inference active (PyTorch not installed)")

        self._extractor = BGCFeatureExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, hgt_result: HGTResult) -> BGCResult:
        """
        Score all alien_records from Phase 2 and return a BGCResult.

        Parameters
        ----------
        hgt_result : HGTResult
            Output from Phase 2 (HGTDetective.run()).

        Returns
        -------
        BGCResult
        """
        t0 = time.perf_counter()
        alien_records = hgt_result.alien_records

        logger.info("Phase 3 | BGCPredictor | %d alien genes to score", len(alien_records))

        if not alien_records:
            logger.warning("Phase 3 | No alien genes received — returning empty BGCResult")
            return self._empty_result()

        # 1. Feature extraction
        X = self._extractor.fit_transform(alien_records)
        logger.debug("  Feature matrix shape: %s", X.shape)

        # 2. Model inference
        if _TORCH_AVAILABLE and self._model is not None:
            probs = _torch_inference(self._model, X)
        else:
            probs = _numpy_mock_inference(X, seed=self.seed)

        # 3. Optional keyword boost
        if self.use_keyword_boost:
            probs, keyword_hits_list = _apply_keyword_boosts(alien_records, probs)
        else:
            keyword_hits_list = [[] for _ in alien_records]

        # 4. Assemble BGCGeneRecord objects
        bgc_records: List[BGCGeneRecord] = []
        for i, (rec, kw_hits) in enumerate(zip(alien_records, keyword_hits_list)):
            class_scores  = probs[i].tolist()
            cls_idx        = int(np.argmax(probs[i]))
            cls_label      = BGC_CLASSES[cls_idx]
            conf           = float(probs[i, cls_idx])

            is_bgc = cls_label != "Non-BGC" and conf >= self.min_confidence

            if conf >= HIGH_CONF_THRESHOLD:
                tier = "High"
            elif conf >= MED_CONF_THRESHOLD:
                tier = "Medium"
            else:
                tier = "Low"

            bgc_records.append(BGCGeneRecord(
                hgt_record       = rec,
                bgc_class        = cls_label,
                bgc_class_idx    = cls_idx,
                confidence       = conf,
                class_scores     = class_scores,
                is_bgc           = is_bgc,
                confidence_tier  = tier,
                keyword_hits     = kw_hits,
            ))

        bgc_hits = [r for r in bgc_records if r.is_bgc]

        # 5. Summary statistics
        class_distribution = {cls: 0 for cls in BGC_CLASSES}
        strain_bgc_counts: Dict[str, int] = {}
        for r in bgc_hits:
            class_distribution[r.bgc_class] += 1
            strain_bgc_counts[r.strain_id] = strain_bgc_counts.get(r.strain_id, 0) + 1

        elapsed = time.perf_counter() - t0
        stats = {
            "n_alien_scored":    len(alien_records),
            "n_bgc_hits":        len(bgc_hits),
            "bgc_hit_rate":      len(bgc_hits) / max(len(alien_records), 1),
            "n_high_confidence": sum(1 for r in bgc_hits if r.confidence_tier == "High"),
            "n_med_confidence":  sum(1 for r in bgc_hits if r.confidence_tier == "Medium"),
            "top_class":         max(class_distribution, key=class_distribution.get),
            "elapsed_s":         round(elapsed, 3),
            "torch_used":        _TORCH_AVAILABLE,
        }

        logger.info(
            "Phase 3 | Done in %.2fs | %d BGC hits (%.1f%%) | top class: %s",
            elapsed, len(bgc_hits), stats["bgc_hit_rate"] * 100, stats["top_class"],
        )

        # 6. DataFrames for inspection / export
        feature_df     = self._make_feature_df(alien_records, X)
        prediction_df  = self._make_prediction_df(bgc_records)

        return BGCResult(
            bgc_records         = bgc_records,
            bgc_hits            = bgc_hits,
            class_distribution  = class_distribution,
            strain_bgc_counts   = strain_bgc_counts,
            feature_matrix      = feature_df,
            prediction_matrix   = prediction_df,
            stats               = stats,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> BGCResult:
        """Return an empty BGCResult when no alien genes exist."""
        return BGCResult(
            bgc_records        = [],
            bgc_hits           = [],
            class_distribution = {cls: 0 for cls in BGC_CLASSES},
            strain_bgc_counts  = {},
            feature_matrix     = pd.DataFrame(),
            prediction_matrix  = pd.DataFrame(),
            stats              = {"n_alien_scored": 0, "n_bgc_hits": 0},
        )

    def _make_feature_df(
        self,
        records: List[HGTGeneRecord],
        X: np.ndarray,
    ) -> pd.DataFrame:
        """Return feature matrix as labelled DataFrame."""
        col_names = (
            ["gc_content", "gc_deviation", "kmer_deviation", "anomaly_score",
             "mge_proximity", "gene_length_log", "upstream_nbrs", "downstream_nbrs"]
            + [f"gc_bin_{i}"   for i in range(10)]
            + [f"kmer_bin_{i}" for i in range(10)]
        )
        df = pd.DataFrame(X, columns=col_names)
        df.insert(0, "gene_id",  [r.gene_record.gene_id  for r in records])
        df.insert(1, "strain_id", [r.gene_record.strain_id for r in records])
        return df

    def _make_prediction_df(self, bgc_records: List[BGCGeneRecord]) -> pd.DataFrame:
        """Return prediction matrix as labelled DataFrame."""
        rows = []
        for r in bgc_records:
            row = {
                "gene_id":        r.gene_id,
                "strain_id":      r.strain_id,
                "bgc_class":      r.bgc_class,
                "confidence":     round(r.confidence, 4),
                "confidence_tier": r.confidence_tier,
                "is_bgc":         r.is_bgc,
                "keyword_hits":   "|".join(r.keyword_hits),
            }
            for cls, score in zip(BGC_CLASSES, r.class_scores):
                row[f"score_{cls}"] = round(score, 4)
            rows.append(row)
        return pd.DataFrame(rows)
