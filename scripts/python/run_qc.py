"""
run_qc.py — CellSPA QC across all completed segmentation results for a single sample.

Auto-discovers completed methods by scanning *_reseg/ dirs for h5ad files.
Runs CellSPA (R) for reference-free metrics + Python plots per method.

Called by the generated SLURM script:
    python run_qc.py --config CONFIG --sample-id XETG... --slide-dir /path/to/slide_folder
                     [--sample-dir /path/to/raw/output-XETG...]
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config, get_output_base_override
from utils.data_io import configure_threads, save_run_metadata, timed


# ── CellSPA R script (reference-free: per-method metrics) ──────────────────────
# Reads pre-exported CSV/MTX files from Python — no zellkonverter needed.
CELLSPA_R_SCRIPT = """\
suppressPackageStartupMessages({
    library(CellSPA)
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(Matrix)
})

args          <- commandArgs(trailingOnly = TRUE)
counts_path   <- args[1]
coords_path   <- args[2]
meta_path     <- args[3]
method_name   <- args[4]
output_dir    <- args[5]
cellseg_path  <- if (length(args) >= 6 && nchar(args[6]) > 0) args[6] else NULL

cat(sprintf("[INFO] CellSPA QC: %s\\n", method_name))

counts  <- readMM(counts_path)
coords  <- as.matrix(read.csv(coords_path)[, c("x", "y")])
meta_df <- read.csv(meta_path)

cat(sprintf("[INFO] Loaded: %d cells x %d genes\\n", ncol(counts), nrow(counts)))

spe <- SpatialExperiment(
    assays        = list(counts = counts),
    colData       = meta_df,
    spatialCoords = coords
)

spe <- tryCatch(processingSPE(spe), error = function(e) {
    cat(sprintf("[WARN] processingSPE: %s\\n", e$message)); spe
})
cat(sprintf("[INFO] After filtering: %d cells\\n", ncol(spe)))

# ── Load cell boundary vertices for morphological metrics ──
if (!is.null(cellseg_path) && file.exists(cellseg_path)) {
    cellseg <- read.csv(cellseg_path)
    cellseg$cell_id <- as.character(cellseg$cell_id)
    spe@metadata$CellSegOutput <- cellseg
    cat(sprintf("[INFO] CellSegOutput loaded: %d points\\n", nrow(cellseg)))

    spe <- tryCatch(
        calBaselineAllMetrics(spe, verbose = FALSE),
        error = function(e) {
            cat(sprintf("[WARN] calBaselineAllMetrics: %s\\n", e$message)); spe
        }
    )
    cat("[INFO] calBaselineAllMetrics complete\\n")
} else {
    cat("[WARN] No boundary vertices — skipping morphological metrics\\n")
}

# ── Spatial diversity metrics ──
spatial_metrics <- tryCatch(
    calSpatialMetricsDiversity(spe),
    error = function(e) { cat(sprintf("[WARN] Spatial metrics: %s\\n", e$message)); NULL }
)

# ── Build summary (medians for counts/genes, means for everything else) ──
summary_df <- data.frame(
    method        = method_name,
    n_cells       = ncol(spe),
    n_genes       = nrow(spe),
    median_counts = median(colSums(counts(spe))),
    median_genes  = median(colSums(counts(spe) > 0)),
    stringsAsFactors = FALSE
)

# Append any CellSPA baseline metrics now sitting in colData
baseline_cols <- c("cell_area", "elongation", "compactness", "eccentricity",
                   "sphericity", "solidity", "convexity", "circularity", "density")
for (col in baseline_cols) {
    if (col %in% names(colData(spe))) {
        summary_df[[col]] <- median(colData(spe)[[col]], na.rm = TRUE)
    }
}

# Append spatial diversity metrics
if (!is.null(spatial_metrics) && is.data.frame(spatial_metrics)) {
    for (col in colnames(spatial_metrics))
        summary_df[[col]] <- mean(spatial_metrics[[col]], na.rm = TRUE)
}

out_path <- file.path(output_dir, sprintf("cellspa_%s.csv", method_name))
write.csv(summary_df, out_path, row.names = FALSE)
cat(sprintf("[INFO] Saved: %s\\n", out_path))
print(summary_df)
"""

# ── PDF report R script (runs once after all methods complete) ──────────────────
CELLSPA_REPORT_R_SCRIPT = """\
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(Matrix)
})

args       <- commandArgs(trailingOnly = TRUE)
comparison <- read.csv(args[1])
coords_dir <- args[2]
output_pdf <- args[3]
sample_id  <- args[4]

# Load per-cell count data for distribution plots
all_cells <- list()
for (method in comparison$method) {
    counts_path <- file.path(coords_dir, sprintf("counts_%s.mtx", method))
    if (!file.exists(counts_path)) next
    counts <- readMM(counts_path)
    all_cells[[method]] <- data.frame(
        method       = method,
        total_counts = colSums(counts),
        n_genes      = colSums(counts > 0)
    )
}
cells_df <- bind_rows(all_cells)

tt <- theme_minimal() + theme(plot.title = element_text(size = 13, face = "bold"))

# ── Page 1: Cells detected ──
p_ncells <- ggplot(comparison, aes(x = method, y = n_cells, fill = method)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = format(n_cells, big.mark = ",")), vjust = -0.3, size = 3.5) +
    labs(title = sprintf("Segmentation QC — %s", sample_id),
         subtitle = "Cells Detected per Method", x = NULL, y = "# Cells") +
    tt

# ── Page 2: Summary metrics bar chart ──
summary_cols <- intersect(c("n_cells", "median_counts", "median_genes",
                             "pct_transcripts_captured"), colnames(comparison))
p_summary <- comparison %>%
    select(method, all_of(summary_cols)) %>%
    pivot_longer(-method, names_to = "metric", values_to = "value") %>%
    ggplot(aes(x = method, y = value, fill = method)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~metric, scales = "free_y") +
    labs(title = "Summary Metrics by Method", x = NULL, y = NULL) +
    tt + theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ── Page 3: Count distributions ──
p_counts <- ggplot(cells_df, aes(x = method, y = total_counts, fill = method)) +
    geom_violin(show.legend = FALSE, trim = TRUE) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, show.legend = FALSE) +
    coord_cartesian(ylim = c(0, quantile(cells_df$total_counts, 0.99, na.rm = TRUE))) +
    labs(title = "Total Counts per Cell", x = NULL, y = "Total Counts") + tt

# ── Page 4: Gene distributions ──
p_genes <- ggplot(cells_df, aes(x = method, y = n_genes, fill = method)) +
    geom_violin(show.legend = FALSE, trim = TRUE) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, show.legend = FALSE) +
    coord_cartesian(ylim = c(0, quantile(cells_df$n_genes, 0.99, na.rm = TRUE))) +
    labs(title = "Genes Detected per Cell", x = NULL, y = "# Genes") + tt

# ── Page 5: Counts vs Genes scatter ──
p_scatter <- ggplot(cells_df, aes(x = total_counts, y = n_genes, color = method)) +
    geom_point(size = 0.3, alpha = 0.2) +
    coord_cartesian(
        xlim = c(0, quantile(cells_df$total_counts, 0.99, na.rm = TRUE)),
        ylim = c(0, quantile(cells_df$n_genes,      0.99, na.rm = TRUE))
    ) +
    labs(title = "Counts vs Genes (all methods)", x = "Total Counts", y = "# Genes") +
    tt + guides(color = guide_legend(override.aes = list(size = 2, alpha = 1)))

# ── Page 6 (optional): % transcripts captured ──
p_pct <- NULL
if ("pct_transcripts_captured" %in% colnames(comparison)) {
    p_pct <- ggplot(comparison, aes(x = method, y = pct_transcripts_captured, fill = method)) +
        geom_col(show.legend = FALSE) +
        geom_text(aes(label = sprintf("%.1f%%", pct_transcripts_captured)), vjust = -0.3) +
        labs(title = "% Transcripts Captured per Method", x = NULL, y = "% Captured") +
        ylim(0, 100) + tt
}

# ── Page 7 (optional): CellSPA morphological metrics ──
morph_cols <- intersect(
    c("cell_area", "elongation", "compactness", "eccentricity",
      "sphericity", "solidity", "convexity", "circularity", "density"),
    colnames(comparison)
)
p_morph <- NULL
if (length(morph_cols) > 0) {
    p_morph <- comparison %>%
        select(method, all_of(morph_cols)) %>%
        pivot_longer(-method, names_to = "metric", values_to = "value") %>%
        ggplot(aes(x = method, y = value, fill = method)) +
        geom_col(show.legend = FALSE) +
        facet_wrap(~metric, scales = "free_y") +
        labs(title = "CellSPA Morphological Metrics (Median per Cell)", x = NULL, y = NULL) +
        tt + theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

# ── Write PDF ──
pdf(output_pdf, width = 11, height = 8.5)
    print(p_ncells)
    print(p_summary)
    print(p_counts)
    print(p_genes)
    print(p_scatter)
    if (!is.null(p_pct))   print(p_pct)
    if (!is.null(p_morph)) print(p_morph)
dev.off()

cat(sprintf("[INFO] PDF report saved: %s\\n", output_pdf))
"""


def count_total_transcripts(sample_dir: Path) -> Optional[int]:
    """Count total transcripts in the raw Xenium sample directory."""
    tx_parquet = sample_dir / "transcripts.parquet"
    tx_csv_gz  = sample_dir / "transcripts.csv.gz"
    try:
        if tx_parquet.exists():
            return len(pd.read_parquet(tx_parquet, columns=["transcript_id"]))
        elif tx_csv_gz.exists():
            return len(pd.read_csv(tx_csv_gz, usecols=["transcript_id"]))
    except Exception as e:
        print(f"[WARN] Could not count transcripts: {e}")
    return None



def export_for_r(adata, method: str, qc_dir: Path, method_output_dir: Path) -> tuple:
    """Export counts, coords, obs metadata, and boundary vertices for R.

    Returns (counts_path, coords_path, meta_path, cellseg_path).
    cellseg_path is None if cell_boundaries.parquet is unavailable.
    """
    from scipy.io import mmwrite
    import scipy.sparse as sp

    counts_path  = qc_dir / f"counts_{method}.mtx"
    coords_path  = qc_dir / f"coords_{method}.csv"
    meta_path    = qc_dir / f"meta_{method}.csv"
    cellseg_path = None

    # Write genes x cells MTX
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    mmwrite(str(counts_path), X.T)

    # Extract spatial coords — sopa stores them in obsm['spatial']
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"][:, :2]
    else:
        xy_cols = [c for c in adata.obs.columns if c in ("x", "y", "spatial_x", "spatial_y")]
        coords = adata.obs[xy_cols[:2]].values if len(xy_cols) >= 2 else None

    if coords is not None:
        pd.DataFrame(coords, columns=["x", "y"]).to_csv(coords_path, index=False)
    else:
        print(f"[WARN] No spatial coordinates found for {method} — spatial metrics will be skipped")
        coords_path = None

    # Export obs metadata (numeric columns only — R colData)
    numeric_obs = adata.obs.select_dtypes(include="number")
    numeric_obs.to_csv(meta_path, index=False)

    # Export boundary polygon vertices as CellSegOutput for CellSPA generatePolygon()
    boundary_parquet = method_output_dir / "cell_boundaries.parquet"
    if boundary_parquet.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_parquet(boundary_parquet)
            # Align to cells that survived sopa filtering
            gdf.index = gdf.index.astype(str)
            cell_ids = set(adata.obs_names.astype(str))
            gdf = gdf[gdf.index.isin(cell_ids)]

            # Extract exterior ring vertices → (x, y, cell_id) rows
            rows = []
            for cell_id, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                for x, y in geom.exterior.coords[:-1]:  # skip duplicate closing point
                    rows.append({"x": float(x), "y": float(y), "cell_id": cell_id})

            if rows:
                cellseg_path = qc_dir / f"cellseg_{method}.csv"
                pd.DataFrame(rows).to_csv(cellseg_path, index=False)
                print(f"[INFO] Boundary vertices exported: {len(rows):,} points, "
                      f"{len(gdf):,} cells → {cellseg_path.name}")
        except Exception as e:
            print(f"[WARN] Could not export boundary vertices: {e}")

    return counts_path, coords_path, meta_path, cellseg_path


def run_cellspa(adata, method: str, qc_dir: Path, method_output_dir: Path) -> bool:
    """Export data, write and run the CellSPA R script for one method. Returns True on success."""
    counts_path, coords_path, meta_path, cellseg_path = export_for_r(
        adata, method, qc_dir, method_output_dir
    )
    if coords_path is None:
        return False

    r_script = qc_dir / f"run_cellspa_{method}.R"
    r_script.write_text(CELLSPA_R_SCRIPT)

    cmd = [
        "Rscript", str(r_script),
        str(counts_path), str(coords_path), str(meta_path),
        method, str(qc_dir),
        str(cellseg_path) if cellseg_path else "",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] CellSPA R failed for {method}:\n{result.stderr}")
        return False
    return True


def generate_pdf_report(comparison_csv: Path, qc_dir: Path, sample_id: str):
    """Generate a multi-page ggplot PDF report comparing all methods."""
    pdf_path = qc_dir / "qc_report.pdf"
    r_script = qc_dir / "run_cellspa_report.R"
    r_script.write_text(CELLSPA_REPORT_R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_script), str(comparison_csv), str(qc_dir), str(pdf_path), sample_id],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] PDF report failed:\n{result.stderr}")
    return pdf_path


@timed("Generate QC plots")
def generate_qc_plots(adata, method_name: str, output_dir: Path):
    """Generate violin + scatter QC plots."""
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


def discover_completed_methods(slide_dir: Path, sample_id: str, output_base: str) -> dict:
    """Scan *_reseg/ dirs and return {method: h5ad_path} for those with results."""
    base = Path(output_base) / slide_dir.name if output_base else slide_dir
    found = {}
    for reseg_dir in sorted(base.glob("*_reseg")):
        method = reseg_dir.name.replace("_reseg", "")
        sample_dir = reseg_dir / sample_id
        h5ads = list(sample_dir.glob("*.h5ad")) if sample_dir.exists() else []
        if h5ads:
            found[method] = h5ads[0]
    return found


def main():
    parser = argparse.ArgumentParser(description="CellSPA QC — all completed segmentation methods")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--slide-dir", required=True, help="Slide folder containing {method}_reseg/ dirs")
    parser.add_argument("--sample-dir", default=None, help="Raw sample dir (for % transcripts captured)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "cellspa_qc")
    output_base = get_output_base_override(cfg)

    slide_dir = Path(args.slide_dir)
    qc_dir = (Path(output_base) / slide_dir.name if output_base else slide_dir) / "qc" / args.sample_id
    qc_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  CellSPA QC — {args.sample_id}")
    print("=" * 60)

    # Count total transcripts once for % capture calculation
    total_transcripts = None
    if args.sample_dir:
        total_transcripts = count_total_transcripts(Path(args.sample_dir))
        if total_transcripts:
            print(f"[INFO] Total transcripts in sample: {total_transcripts:,}")
        else:
            print("[WARN] Could not count total transcripts — % capture will be skipped")

    # Auto-discover completed methods
    discovered = discover_completed_methods(slide_dir, args.sample_id, output_base)
    if not discovered:
        print(f"[WARN] No completed segmentation results found under {slide_dir}")
        elapsed = time.time() - t_start
        save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
        sys.exit(0)

    print(f"[INFO] Found results for: {', '.join(discovered.keys())}\n")

    import scanpy as sc
    import scipy.sparse as sp
    cellspa_results = []

    for method, h5ad_path in discovered.items():
        print(f"\n── {method} ──")
        adata = sc.read_h5ad(h5ad_path)
        print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

        # CellSPA R metrics (includes morphological via calBaselineAllMetrics)
        @timed(f"CellSPA metrics: {method}")
        def _cellspa():
            return run_cellspa(adata, method, qc_dir, h5ad_path.parent)
        success = _cellspa()

        if success:
            csv = qc_dir / f"cellspa_{method}.csv"
            if csv.exists():
                df = pd.read_csv(csv)

                # % transcripts captured
                if total_transcripts:
                    X = adata.X
                    assigned = int(X.sum()) if sp.issparse(X) else int(np.sum(X))
                    df["pct_transcripts_captured"] = round(100.0 * assigned / total_transcripts, 2)
                    print(f"[INFO] Transcripts captured: {assigned:,} / {total_transcripts:,} "
                          f"({df['pct_transcripts_captured'].iloc[0]:.1f}%)")
                    df.to_csv(csv, index=False)

                cellspa_results.append(df)

        # Python QC plots
        generate_qc_plots(adata, method, qc_dir)

    # Combined CellSPA comparison table + PDF report
    if cellspa_results:
        comparison = pd.concat(cellspa_results, ignore_index=True)
        comparison_csv = qc_dir / "cellspa_comparison.csv"
        comparison.to_csv(comparison_csv, index=False)
        print(f"\n── CellSPA Comparison ──")
        print(comparison.to_string(index=False))

        @timed("Generate PDF report")
        def _pdf():
            return generate_pdf_report(comparison_csv, qc_dir, args.sample_id)
        pdf_path = _pdf()
        print(f"[INFO] Report: {pdf_path}")

    elapsed = time.time() - t_start
    save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
    print(f"\n[DONE] CellSPA QC — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
