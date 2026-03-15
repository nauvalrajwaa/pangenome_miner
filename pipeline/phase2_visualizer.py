"""
PanAdapt-BGC Miner — Phase 2 Visualizer
=========================================
Module  : pipeline/phase2_visualizer.py
Purpose : Generate Phase 2 output artefacts:
            1. Genomic Island Architecture Plot (matplotlib)
            2. HGT Feature Distribution Plots
            3. Phase 2 HTML report section (Jinja2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from pipeline.hgt_detective import HGTGeneRecord, HGTResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Genomic Island Architecture Plot
# ---------------------------------------------------------------------------

def plot_genomic_island_architecture(
    hgt_result: HGTResult,
    strain_id: str,
    output_path: Path,
    max_genes: int = 120,
) -> Optional[Path]:
    """
    Linear genomic map for *strain_id*:
      - Core/shell genes → dull gray arrows
      - HGT genes → dark red arrows
      - Self-labelled MGE genes → orange
      - Product labels drawn inside arrows where space allows

    Parameters
    ----------
    hgt_result  : HGTResult from Phase 2
    strain_id   : Which strain to plot
    output_path : Save path (PNG)
    max_genes   : Cap on genes shown per contig for readability
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter records for this strain
    strain_records = [r for r in hgt_result.hgt_records
                      if r.gene_record.strain_id == strain_id]
    if not strain_records:
        logger.warning("No HGT records found for strain %s", strain_id)
        return None

    # Group by contig, sort by start
    contig_map: Dict[str, List[HGTGeneRecord]] = {}
    for r in strain_records:
        contig_map.setdefault(r.gene_record.contig, []).append(r)
    for c in contig_map:
        contig_map[c].sort(key=lambda x: x.gene_record.start)

    # Pick top 3 contigs by gene count
    top_contigs = sorted(contig_map, key=lambda c: len(contig_map[c]), reverse=True)[:3]

    n_contigs = len(top_contigs)
    fig_height = max(4, n_contigs * 3.5)
    fig, axes = plt.subplots(n_contigs, 1, figsize=(18, fig_height))
    if n_contigs == 1:
        axes = [axes]

    fig.suptitle(
        f"Genomic Island Architecture — {strain_id}",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, contig in zip(axes, top_contigs):
        records = contig_map[contig][:max_genes]
        contig_start = records[0].gene_record.start
        contig_end   = records[-1].gene_record.end
        span = contig_end - contig_start + 1

        ax.set_xlim(contig_start - span * 0.02, contig_end + span * 0.02)
        ax.set_ylim(-0.8, 1.2)
        ax.axhline(0, color="#bdc3c7", lw=1.5, zorder=1)  # backbone

        for rec in records:
            gene = rec.gene_record
            s, e = gene.start, gene.end
            length = e - s + 1
            y_center = 0.0
            arrow_height = 0.45

            # Choose colour
            if rec.is_hgt:
                color = "#922b21"    # dark red — Alien HGT
                zorder = 4
            elif rec.mge_proximity and any(
                kw in gene.product.lower()
                for kw in ("transposase", "integrase", "phage", "insertion")
            ):
                color = "#d35400"    # orange — MGE
                zorder = 3
            else:
                color = "#aab7b8"    # gray — core/shell
                zorder = 2

            # Draw arrow
            direction = 1 if gene.strand in ("+", ".") else -1
            ax.annotate(
                "",
                xy=(e if direction == 1 else s, y_center),
                xytext=(s if direction == 1 else e, y_center),
                arrowprops=dict(
                    arrowstyle=f"simple,head_width={arrow_height*2},tail_width={arrow_height}",
                    color=color,
                    alpha=0.85,
                ),
                zorder=zorder,
            )

            # Label HGT genes (skip tiny ones)
            if rec.is_hgt and length > span * 0.015:
                label = gene.product[:22] if gene.product else gene.gene_id[:18]
                ax.text(
                    (s + e) / 2, y_center + 0.58,
                    label,
                    ha="center", va="bottom",
                    fontsize=6.5, color="#922b21",
                    fontweight="bold",
                    clip_on=True,
                )

        # Axes labels
        n_hgt = sum(1 for r in records if r.is_hgt)
        ax.set_title(
            f"Contig: {contig}  |  {len(records)} genes shown  |  "
            f"{n_hgt} Alien HGT ([HGT])  |  pos {contig_start:,}\u2013{contig_end:,} bp",
            fontsize=9, loc="left",
        )
        ax.set_xlabel("Genomic position (bp)", fontsize=8)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

        # x-axis: nice Mbp/kbp formatting
        xticks = ax.get_xticks()
        from matplotlib.ticker import FixedLocator
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xticklabels([f"{x/1000:.0f} kb" for x in xticks])

    # Legend
    legend_patches = [
        mpatches.Patch(color="#aab7b8", label="Core / Shell gene"),
        mpatches.Patch(color="#d35400", label="Mobile Genetic Element (MGE)"),
        mpatches.Patch(color="#922b21", label="Alien HGT Region (Phase 2 flag)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Genomic island plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. HGT Feature Distribution
# ---------------------------------------------------------------------------

def plot_hgt_feature_distributions(
    hgt_result: HGTResult,
    output_path: Path,
) -> Path:
    """
    Four-panel figure showing feature distributions for HGT vs normal genes:
      - GC content histogram
      - GC deviation
      - K-mer cosine deviation
      - Anomaly score
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hgt_recs = [r for r in hgt_result.hgt_records if r.is_hgt]
    nrm_recs = [r for r in hgt_result.hgt_records if not r.is_hgt]

    if not hgt_recs:
        logger.warning("No HGT genes to plot feature distributions.")
        return output_path

    def _get(recs, attr):
        return [getattr(r, attr) for r in recs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Phase 2 — HGT Feature Distributions\n(Alien HGT Regions vs Normal Accessory Genes)",
                 fontsize=12, fontweight="bold")

    panels = [
        ("gc_content",    "GC Content",             "GC Fraction"),
        ("gc_deviation",  "GC Deviation from Host",  "| Gene GC − Host GC | / Host GC"),
        ("kmer_deviation","K-mer Cosine Deviation",  "1 − Cosine Similarity to Host"),
        ("anomaly_score", "Anomaly Score",           "IsolationForest Score (higher=more alien)"),
    ]

    for ax, (attr, title, xlabel) in zip(axes.flat, panels):
        nrm_vals = _get(nrm_recs, attr)
        hgt_vals = _get(hgt_recs, attr)

        bins = 35
        ax.hist(nrm_vals, bins=bins, alpha=0.6, color="#27ae60",
                label=f"Normal ({len(nrm_recs):,})", density=True)
        ax.hist(hgt_vals, bins=bins, alpha=0.7, color="#922b21",
                label=f"HGT ({len(hgt_recs):,})", density=True)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("HGT feature distribution plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Phase 2 HTML Report (Jinja2)
# ---------------------------------------------------------------------------

PHASE2_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PanAdapt-BGC Miner — Phase 2 Report</title>
  <style>
    :root { --hgt: #922b21; --normal: #1a5276; --accent: #1a7a5e; --mge: #d35400; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #2c3e50; }
    header { background: var(--hgt); color: white; padding: 2rem 3rem; }
    header h1 { font-size: 1.8rem; }
    header p  { opacity: 0.85; margin-top: 0.4rem; font-size: 0.95rem; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 2rem; }
    .card { background: white; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 1.8rem; margin-bottom: 1.8rem; }
    .card h2 { font-size: 1.2rem; color: var(--hgt); border-bottom: 2px solid var(--hgt);
               padding-bottom: 0.5rem; margin-bottom: 1.2rem; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }
    .stat-box { text-align: center; padding: 1.2rem; border-radius: 8px; }
    .stat-box .value { font-size: 2rem; font-weight: 700; }
    .stat-box .label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }
    .hgt-box  { background: #fdedec; color: var(--hgt); }
    .norm-box { background: #eaf4fb; color: var(--normal); }
    .mge-box  { background: #fef9e7; color: var(--mge); }
    .gc-box   { background: #e9f7ef; color: var(--accent); }
    table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
    th { background: #f2f3f4; text-align: left; padding: 0.6rem 0.7rem;
         font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.4px; }
    td { padding: 0.5rem 0.7rem; border-bottom: 1px solid #ecf0f1; }
    tr:hover td { background: #fdfefe; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px;
             font-size: 0.76rem; font-weight: 600; }
    .badge-hgt  { background: #fdedec; color: var(--hgt); }
    .badge-mge  { background: #fef5ec; color: var(--mge); }
    .badge-norm { background: #eaf4fb; color: var(--normal); }
    .evidence   { font-size: 0.78rem; color: #7f8c8d; font-style: italic; }
    .plot-img   { width: 100%; border-radius: 6px; border: 1px solid #dde; }
    .alert-hgt  { background: #fdedec; border-left: 4px solid var(--hgt);
                  padding: 1rem 1.2rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.92rem; }
    .gc-table td, .gc-table th { padding: 0.4rem 0.7rem; }
    footer { text-align: center; padding: 2rem; color: #95a5a6; font-size: 0.82rem; }
  </style>
</head>
<body>
<header>
  <h1>🔬 PanAdapt-BGC Miner</h1>
  <p>Phase 2 Report — The HGT Detective (Horizontal Gene Transfer Analysis)</p>
  <p style="margin-top:0.6rem; font-size:0.85rem; opacity:0.7;">Generated: {{ generated_at }}</p>
</header>

<div class="container">

  <div class="stats-grid" style="margin-bottom:1.8rem;">
    <div class="stat-box hgt-box">
      <div class="value">{{ "{:,}".format(stats.n_alien_hgt) }}</div>
      <div class="label">Alien HGT Regions</div>
    </div>
    <div class="stat-box norm-box">
      <div class="value">{{ "{:,}".format(stats.n_normal) }}</div>
      <div class="label">Normal Accessory</div>
    </div>
    <div class="stat-box mge-box">
      <div class="value">{{ "{:,}".format(stats.n_mge_proximal) }}</div>
      <div class="label">MGE-proximal genes</div>
    </div>
    <div class="stat-box gc-box">
      <div class="value">{{ "%.1f"|format(stats.hgt_fraction * 100) }}%</div>
      <div class="label">HGT Fraction</div>
    </div>
  </div>

  <div class="alert-hgt">
    <strong>🚨 Alert:</strong>
    {{ "{:,}".format(stats.n_alien_hgt) }} of {{ "{:,}".format(stats.n_accessory_input) }}
    accessory genes show significant HGT signatures based on k-mer frequency deviation,
    GC content skew, and/or Mobile Genetic Element proximity.
    These are flagged as <strong>Alien_HGT_Regions</strong> and will be passed to
    <strong>Phase 3 (AI BGC Predictor)</strong> for biosynthetic gene cluster analysis.
  </div>

  <!-- Per-strain GC profiles -->
  <div class="card">
    <h2>🧬 Host Genome GC Content Profiles</h2>
    <table class="gc-table">
      <thead><tr><th>Strain</th><th>Host Genome GC</th></tr></thead>
      <tbody>
        {% for strain, gc in stats.strain_gc_profiles.items() %}
        <tr>
          <td><code>{{ strain }}</code></td>
          <td>{{ "%.2f"|format(gc * 100) }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Top HGT records table -->
  <div class="card">
    <h2>🛸 Top Alien HGT Regions (ranked by anomaly score)</h2>
    <p style="font-size:0.85rem;color:#7f8c8d;margin-bottom:1rem;">
      Showing top {{ top_hgt|length }} of {{ "{:,}".format(stats.n_alien_hgt) }} HGT-flagged genes.
    </p>
    <table>
      <thead>
        <tr>
          <th>Gene ID</th><th>Strain</th><th>Contig</th>
          <th>Position</th><th>GC</th><th>GC Dev</th><th>k-mer Dev</th>
          <th>MGE</th><th>Anomaly</th><th>Product / Evidence</th>
        </tr>
      </thead>
      <tbody>
        {% for r in top_hgt %}
        <tr>
          <td><code style="font-size:0.8rem;">{{ r.gene_record.gene_id[:28] }}</code></td>
          <td><code style="font-size:0.8rem;">{{ r.gene_record.strain_id[:25] }}</code></td>
          <td><code style="font-size:0.78rem;">{{ r.gene_record.contig[:18] }}</code></td>
          <td style="font-size:0.8rem;">{{ "{:,}".format(r.gene_record.start) }}–{{ "{:,}".format(r.gene_record.end) }}</td>
          <td>{{ "%.1f"|format(r.gc_content * 100) }}%</td>
          <td>{{ "%.3f"|format(r.gc_deviation) }}</td>
          <td>{{ "%.4f"|format(r.kmer_deviation) }}</td>
          <td>{% if r.mge_proximity %}<span class="badge badge-mge">MGE</span>{% else %}—{% endif %}</td>
          <td><strong>{{ "%.3f"|format(r.anomaly_score) }}</strong></td>
          <td>
            <span style="font-size:0.8rem;">{{ r.gene_record.product[:40] if r.gene_record.product else "—" }}</span>
            {% if r.evidence %}
            <br><span class="evidence">{{ r.evidence | join("; ") | truncate(80) }}</span>
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Plots -->
  {% for strain_id, plot_rel in genomic_plots.items() %}
  <div class="card">
    <h2>🗺 Genomic Island Architecture — {{ strain_id }}</h2>
    <p style="font-size:0.85rem;color:#7f8c8d;margin-bottom:0.8rem;">
      Dark red = Alien HGT Region &nbsp;|&nbsp; Orange = MGE &nbsp;|&nbsp; Gray = Core/Shell gene.
      Gene labels shown for HGT-flagged regions.
    </p>
    <img class="plot-img" src="{{ plot_rel }}" alt="Genomic island plot {{ strain_id }}">
  </div>
  {% endfor %}

  <div class="card">
    <h2>📊 HGT Feature Distributions</h2>
    <img class="plot-img" src="{{ feature_dist_rel }}" alt="HGT feature distributions">
  </div>

</div>

<footer>
  PanAdapt-BGC Miner v0.1 &mdash; Phase 2 complete.
  {{ "{:,}".format(stats.n_alien_hgt) }} Alien HGT Regions ready for Phase 3 (AI BGC Prediction).
</footer>
</body>
</html>
"""


def render_phase2_html_report(
    hgt_result: HGTResult,
    genomic_plot_paths: Dict[str, Path],
    feature_dist_path: Path,
    output_path: Path,
) -> Path:
    """Render the Phase 2 HTML report."""
    try:
        from jinja2 import Environment, BaseLoader
    except ImportError:
        logger.warning("jinja2 not installed — skipping Phase 2 HTML report")
        return output_path

    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output_path.parent

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(report_dir))
        except ValueError:
            return str(p)

    # Sort HGT records by anomaly score descending
    top_hgt = sorted(hgt_result.alien_records, key=lambda r: r.anomaly_score, reverse=True)[:80]

    env = Environment(loader=BaseLoader())
    env.filters["truncate"] = lambda s, n: s[:n] + "…" if len(s) > n else s
    template = env.from_string(PHASE2_HTML_TEMPLATE)

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stats={**hgt_result.stats, "n_accessory_input": len(hgt_result.hgt_records)},
        top_hgt=top_hgt,
        genomic_plots={sid: _rel(p) for sid, p in genomic_plot_paths.items()},
        feature_dist_rel=_rel(feature_dist_path),
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Phase 2 HTML report saved → %s", output_path)
    return output_path
