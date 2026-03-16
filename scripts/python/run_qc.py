"""
run_qc.py — Run QC metrics across completed segmentation results for a single sample.

Called by the generated SLURM script:
    python run_qc.py --config CONFIG --sample-dir /path/to/output-XETG... \
                     --sample-id XETG... --slide-dir /path/to/slide_folder
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config, get_output_base_override
from utils.data_io import configure_threads, save_run_metadata, timed


@timed("Compute morphological QC")
def compute_morph_qc(adata, method_name: str) -> pd.DataFrame:
    """Compute basic morphological and transcript QC metrics."""
    metrics = {}
    metrics["n_cells"] = adata.n_obs
    metrics["n_genes"] = adata.n_vars

    counts = np.asarray(adata.X.sum(axis=1)).flatten()
    genes = np.asarray((adata.X > 0).sum(axis=1)).flatten()

    metrics["median_genes_per_cell"] = float(np.median(genes))
    metrics["median_counts_per_cell"] = float(np.median(counts))
    metrics["mean_counts_per_cell"] = float(np.mean(counts))

    if hasattr(adata.X, "toarray"):
        total = adata.n_obs * adata.n_vars
        metrics["sparsity"] = round(1 - (adata.X.nnz / total), 4)
    metrics["pct_cells_lt_10_counts"] = round(float(np.mean(counts < 10)) * 100, 2)
    metrics["pct_cells_lt_50_counts"] = round(float(np.mean(counts < 50)) * 100, 2)
    metrics["pct_cells_lt_5_genes"] = round(float(np.mean(genes < 5)) * 100, 2)

    if "area" in adata.obs.columns:
        metrics["median_cell_area"] = float(adata.obs["area"].median())

    metrics["method"] = method_name
    return pd.DataFrame([metrics])


@timed("Generate QC plots")
def generate_qc_plots(adata, method_name: str, output_dir: Path):
    """Generate standard QC violin/scatter plots."""
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"QC — {method_name}", fontsize=14)

    for ax, metric, label in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
        ["Genes per Cell", "Total Counts", "% in Top 50 Genes"],
    ):
        if metric in adata.obs.columns:
            ax.violinplot(adata.obs[metric].values, showmedians=True)
            ax.set_title(label)
            ax.set_ylabel(label)

    plt.tight_layout()
    fig.savefig(output_dir / f"qc_violin_{method_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"], s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Total Counts")
    ax.set_ylabel("Genes Detected")
    ax.set_title(f"{method_name}: Counts vs Genes")
    fig.savefig(output_dir / f"qc_scatter_{method_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run segmentation QC (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--slide-dir", required=True, help="Slide folder containing {method}_reseg/ dirs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "cellspa_qc")
    params = method_cfg["params"]
    output_base = get_output_base_override(cfg)

    # QC output goes alongside the method results
    slide_dir = Path(args.slide_dir)
    if output_base:
        # When using override, slide_dir arg is the slide name, resolve under override
        qc_dir = Path(output_base) / slide_dir.name / "qc" / args.sample_id
    else:
        qc_dir = slide_dir / "qc" / args.sample_id
    qc_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  QC — {args.sample_id}")
    print("=" * 60)

    methods_to_qc = params.get("methods_to_qc", ["proseg", "baysor", "cellpose"])
    all_metrics = []

    import scanpy as sc

    for method in methods_to_qc:
        # Find method results for this sample
        if output_base:
            method_dir = Path(output_base) / slide_dir.name / f"{method}_reseg" / args.sample_id
        else:
            method_dir = slide_dir / f"{method}_reseg" / args.sample_id

        h5ad_files = list(method_dir.glob("*.h5ad")) if method_dir.exists() else []

        if not h5ad_files:
            print(f"[WARN] No h5ad for {method} at {method_dir}, skipping")
            continue

        print(f"\n── QC: {method} ──")
        adata = sc.read_h5ad(h5ad_files[0])
        print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

        metrics_df = compute_morph_qc(adata, method)
        all_metrics.append(metrics_df)
        generate_qc_plots(adata, method, qc_dir)

    if all_metrics:
        comparison = pd.concat(all_metrics, ignore_index=True)
        comparison.to_csv(qc_dir / "method_comparison.csv", index=False)
        print(f"\n{comparison.to_string(index=False)}")

    elapsed = time.time() - t_start
    save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
    print(f"[DONE] QC — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
