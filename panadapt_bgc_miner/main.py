"""
PanAdapt-BGC Miner — Main Pipeline Entry Point
===============================================
Module  : main.py
Purpose : CLI entry point for the full PanAdapt-BGC Miner pipeline.
          Currently wires Phase 1 (PangenomeMiner) and is scaffolded
          for Phase 2 (HGT Detective) and Phase 3 (AI BGC Predictor).

Usage
-----
  # Run with real data:
  python main.py --genomes genomes/ --annotations annotations/ --output output/

  # Run with auto-generated mock data (for testing):
  python main.py --mock --output output/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline.pangenome_miner import PangenomeMiner, PangenomeResult

logger = logging.getLogger("panadapt_bgc_miner")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panadapt-bgc-miner",
        description=(
            "PanAdapt-BGC Miner — Pangenome Adaptive & Alien Biosynthetic Gene "
            "Cluster Miner. Integrates pangenome analysis, HGT detection, and "
            "AI-driven BGC prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--genomes", type=Path, default=Path("genomes"),
        help="Directory containing FASTA genome files (.fasta / .fna / .fa).",
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("annotations"),
        help="Directory containing GFF3 annotation files (.gff / .gff3).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"),
        help="Root output directory for all pipeline artefacts.",
    )
    parser.add_argument(
        "--core-threshold", type=float, default=0.95,
        help="Fraction of strains a gene must appear in to be Core (default: 0.95).",
    )
    parser.add_argument(
        "--accessory-threshold", type=float, default=0.10,
        help="Max fraction of strains for a gene to be Accessory (default: 0.10).",
    )
    parser.add_argument(
        "--identity", type=float, default=0.80,
        help="Ortholog clustering sequence-identity threshold (default: 0.80).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Generate and use synthetic mock data for testing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Phase 1 runner
# ---------------------------------------------------------------------------
def run_phase1(args: argparse.Namespace) -> PangenomeResult:
    logger.info("=" * 60)
    logger.info("PHASE 1 — The Comparatist Engine (Pangenome Analysis)")
    logger.info("=" * 60)

    miner = PangenomeMiner(
        core_threshold=args.core_threshold,
        accessory_threshold=args.accessory_threshold,
        identity_threshold=args.identity,
    )

    result: PangenomeResult = miner.run(args.genomes, args.annotations)

    # Persist matrix for downstream inspection
    matrix_out = args.output / "phase1_presence_absence_matrix.csv"
    miner.save_matrix(matrix_out)

    # Summary print
    stats = result.stats
    print(
        f"\n{'─' * 55}\n"
        f"  Phase 1 Summary\n"
        f"{'─' * 55}\n"
        f"  Strains analysed   : {stats['n_strains']}\n"
        f"  Total clusters     : {stats['n_total_clusters']}\n"
        f"  Core genome        : {stats['n_core']} clusters\n"
        f"  Shell genome       : {stats['n_shell']} clusters\n"
        f"  Accessory genome   : {stats['n_accessory']} clusters "
        f"({stats['n_accessory_records']} individual genes)\n"
        f"{'─' * 55}\n"
        f"  Matrix saved → {matrix_out}\n"
    )
    return result


# ---------------------------------------------------------------------------
# Phase 2 stub
# ---------------------------------------------------------------------------
def run_phase2(phase1_result: PangenomeResult, args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 2 — The HGT Detective (placeholder — coming soon)")
    logger.info("=" * 60)
    logger.info(
        "Will analyse %d accessory gene records for HGT signatures …",
        len(phase1_result.accessory_records),
    )
    # TODO: Import and call HGTDetective(phase1_result).run()


# ---------------------------------------------------------------------------
# Phase 3 stub
# ---------------------------------------------------------------------------
def run_phase3(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 3 — The AI Discoverer (placeholder — coming soon)")
    logger.info("=" * 60)
    # TODO: Import and call BGCPredictor(hgt_result).run()


# ---------------------------------------------------------------------------
# Mock data helper
# ---------------------------------------------------------------------------
def prepare_mock_data(args: argparse.Namespace) -> None:
    mock_dir = Path(__file__).parent / "mock_data"
    gen_script = mock_dir / "generate_mock_data.py"

    logger.info("Generating mock data …")
    result = subprocess.run(
        [sys.executable, str(gen_script)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Mock data generation failed.")

    # Override paths to point at generated data
    args.genomes = mock_dir / "genomes"
    args.annotations = mock_dir / "annotations"
    logger.info("Mock data ready: %s / %s", args.genomes, args.annotations)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging level
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Optional: generate mock data
    if args.mock:
        prepare_mock_data(args)

    # --- Run pipeline phases ---
    phase1_result = run_phase1(args)
    run_phase2(phase1_result, args)
    run_phase3(args)

    logger.info("Pipeline complete. All outputs in: %s", args.output)


if __name__ == "__main__":
    main()
