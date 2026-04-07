"""
run_qc.py — CellSPA QC across all completed segmentation results for a single sample.

Auto-discovers completed methods by scanning *_reseg/ dirs for h5ad files.
Runs CellSPA (R) for reference-free basic metrics + Python morphological metrics.

Called by the generated SLURM script:
    python run_qc.py --config CONFIG --sample-id XETG... --slide-dir /path/to/slide_folder
                     [--sample-dir /path/to/raw/output-XETG...]
"""

import os
import sys
import time
import math
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config, get_output_base_override
from utils.data_io import configure_threads, save_run_metadata, timed


# ── CellSPA R script (basic QC metrics only) ───────────────────────────────────
# Reads pre-exported CSV/MTX files from Python — no zellkonverter needed.
#
# DEPRECATED: CellSPA morphological metrics (calBaselineAllMetrics / generatePolygon)
#   CellSPA's generatePolygon() consistently returns 0 polygons due to internal
#   bugs incompatible with our boundary format. Morphological metrics are now
#   computed in Python via compute_morphological_metrics() using shapely geometry.
#
# DEPRECATED: CellSPA spatial diversity metrics (calSpatialMetricsDiversity)
#   Spatial diversity metrics are not computed here. If needed, implement
#   directly in Python.
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

cat(sprintf("[INFO] CellSPA QC: %s\\n", method_name))

counts  <- readMM(counts_path)
coords  <- as.matrix(read.csv(coords_path)[, c("x", "y")])
meta_df <- read.csv(meta_path, row.names = 1)

cat(sprintf("[INFO] Loaded: %d cells x %d genes\\n", ncol(counts), nrow(counts)))

spe <- SpatialExperiment(
    assays        = list(counts = counts),
    colData       = meta_df,
    spatialCoords = coords
)

# Add placeholder celltype — required by CellSPA processingSPE
colData(spe)$celltype <- "all"

spe <- tryCatch(processingSPE(spe), error = function(e) {
    cat(sprintf("[WARN] processingSPE: %s\\n", e$message)); spe
})
cat(sprintf("[INFO] After filtering: %d cells\\n", ncol(spe)))

# ── Basic QC summary ──
summary_df <- data.frame(
    method        = method_name,
    n_cells       = ncol(spe),
    n_genes       = nrow(spe),
    median_counts = median(colSums(counts(spe))),
    median_genes  = median(colSums(counts(spe) > 0)),
    stringsAsFactors = FALSE
)

out_path <- file.path(output_dir, sprintf("cellspa_%s.csv", method_name))
write.csv(summary_df, out_path, row.names = FALSE)
cat(sprintf("[INFO] Saved: %s\\n", out_path))
print(summary_df)
"""

# ── PDF report R script path (standalone file — see scripts/r/qc_report.R) ────
_QC_REPORT_R_SCRIPT = Path(__file__).resolve().parent.parent / "r" / "qc_report.R"

# Legacy inline heredoc kept below as fallback reference only — NOT used.
CELLSPA_REPORT_R_SCRIPT_UNUSED = """\
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(Matrix)
    library(patchwork)
})

args              <- commandArgs(trailingOnly = TRUE)
comparison        <- read.csv(args[1])
coords_dir        <- args[2]
qc_page_pdf       <- args[3]
sample_id         <- args[4]
morpho_page_pdf   <- args[5]
celltype_page_pdf <- if (length(args) >= 6) args[6] else file.path(coords_dir, "_temp_celltype_page.pdf")

# Preserve CSV row order in all plots (xenium first, then reseg methods)
method_levels <- comparison$method
comparison$method <- factor(comparison$method, levels = method_levels)

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
cells_df$method <- factor(cells_df$method, levels = method_levels)

# dittoSeq color palette (colorblind-friendly)
ditto_colors <- c(
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#666666",
    "#AD7700", "#1C91D4", "#007756", "#D5C711", "#005685", "#A04700", "#B14380", "#4D4D4D",
    "#FFBE2D", "#80C7EF", "#00F6B3", "#F4EB71", "#06A5FF", "#FF8320", "#D99BBD", "#8C8C8C",
    "#FFCB57", "#9AD2F2", "#2CFFC6", "#F6EF8E", "#38B7FF", "#FF9B4D", "#E0AFCA", "#A3A3A3",
    "#8A5F00", "#1674A9", "#005F45", "#AA9F0D", "#00446B", "#803800", "#8D3666", "#3D3D3D"
)
fill_scale  <- scale_fill_manual(values  = ditto_colors)
color_scale <- scale_color_manual(values = ditto_colors)

tt <- theme_minimal(base_size = 9) +
    theme(plot.title = element_text(size = 9, face = "bold"),
          legend.position = "none")

# ── Page 1 panels ──
p_ncells <- ggplot(comparison, aes(x = method, y = n_cells, fill = method)) +
    geom_col() +
    geom_text(aes(label = format(n_cells, big.mark = ",")), vjust = -0.3, size = 3) +
    labs(title = "Cells Detected", x = NULL, y = "# Cells") + fill_scale + tt

p_pct <- NULL
if ("pct_transcripts_captured" %in% colnames(comparison)) {
    p_pct <- ggplot(comparison, aes(x = method, y = pct_transcripts_captured, fill = method)) +
        geom_col() +
        geom_text(aes(label = sprintf("%.1f%%", pct_transcripts_captured)), vjust = -0.3, size = 3) +
        labs(title = "% Transcripts Captured", x = NULL, y = "% Captured") +
        ylim(0, 100) + fill_scale + tt
}

p_med <- comparison %>%
    select(method, median_counts, median_genes) %>%
    pivot_longer(-method, names_to = "metric", values_to = "value") %>%
    ggplot(aes(x = method, y = value, fill = method)) +
    geom_col() +
    facet_wrap(~metric, scales = "free_y") +
    labs(title = "Median per Cell", x = NULL, y = NULL) +
    fill_scale + tt + theme(axis.text.x = element_text(angle = 25, hjust = 1))

p_counts <- ggplot(cells_df, aes(x = method, y = total_counts, fill = method)) +
    geom_violin(trim = TRUE) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
    coord_cartesian(ylim = c(0, quantile(cells_df$total_counts, 0.99, na.rm = TRUE))) +
    labs(title = "Counts / Cell", x = NULL, y = "Total Counts") +
    fill_scale + tt + theme(aspect.ratio = 1)

p_genes <- ggplot(cells_df, aes(x = method, y = n_genes, fill = method)) +
    geom_violin(trim = TRUE) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
    coord_cartesian(ylim = c(0, quantile(cells_df$n_genes, 0.99, na.rm = TRUE))) +
    labs(title = "Unique Genes / Cell", x = NULL, y = "# Unique Genes") +
    fill_scale + tt + theme(aspect.ratio = 1)

p_scatter <- ggplot(cells_df, aes(x = total_counts, y = n_genes, color = method)) +
    geom_point(size = 0.3, alpha = 0.2) +
    coord_cartesian(
        xlim = c(0, quantile(cells_df$total_counts, 0.99, na.rm = TRUE)),
        ylim = c(0, quantile(cells_df$n_genes,      0.99, na.rm = TRUE))
    ) +
    labs(title = "Counts vs Unique Genes", x = "Total Counts", y = "# Unique Genes") +
    theme_minimal(base_size = 9) + theme(aspect.ratio = 1) +
    color_scale +
    guides(color = guide_legend(override.aes = list(size = 2, alpha = 1), title = NULL))

# ── Load morpho CSVs (needed for both Page 1 nucleus metric and Page 2 violins) ──
morpho_metrics <- c("cell_area", "elongation", "circularity", "compactness",
                    "eccentricity", "solidity", "convexity", "density", "nuclear_ratio")

all_morpho <- list()
for (method in levels(comparison$method)) {
    morpho_path <- file.path(coords_dir, sprintf("morpho_%s.csv", method))
    if (!file.exists(morpho_path)) next
    df <- read.csv(morpho_path)
    df$method <- as.character(method)
    all_morpho[[method]] <- df
}

# Compute % cells without a matched nucleus and add to comparison table
if (length(all_morpho) > 0 && "nuclear_ratio" %in% colnames(bind_rows(all_morpho))) {
    nuc_stats <- bind_rows(lapply(all_morpho, function(df) {
        data.frame(
            method           = df$method[1],
            pct_no_nucleus   = if ("nuclear_ratio" %in% colnames(df))
                                   100 * sum(is.na(df$nuclear_ratio)) / nrow(df)
                               else NA_real_
        )
    }))
    comparison <- left_join(comparison, nuc_stats, by = "method")
}

p_no_nucleus <- NULL
if ("pct_no_nucleus" %in% colnames(comparison) && any(!is.na(comparison$pct_no_nucleus))) {
    p_no_nucleus <- ggplot(comparison, aes(x = method, y = pct_no_nucleus, fill = method)) +
        geom_col() +
        geom_text(aes(label = sprintf("%.1f%%", pct_no_nucleus)), vjust = -0.3, size = 3) +
        labs(title = "% Cells Without Nucleus", x = NULL, y = "% Cells") +
        ylim(0, 100) + fill_scale + tt
}

top_row <- if (!is.null(p_pct) && !is.null(p_no_nucleus)) {
    p_ncells | p_pct | p_no_nucleus | p_med
} else if (!is.null(p_pct)) {
    p_ncells | p_pct | p_med
} else if (!is.null(p_no_nucleus)) {
    p_ncells | p_no_nucleus | p_med
} else {
    p_ncells | p_med
}
bottom_row <- p_counts | p_genes | p_scatter

# ── Cell type annotation CSVs (optional — written by classifier step) ──
all_annot <- list()
for (method in as.character(method_levels)) {
    annot_path <- file.path(coords_dir, sprintf("annotations_%s.csv", method))
    if (!file.exists(annot_path)) next
    df <- read.csv(annot_path, stringsAsFactors = FALSE)
    if (!"predicted_cell_type" %in% colnames(df)) next
    df$cell_id <- as.character(df$cell_id)
    df$method <- as.character(method)
    all_annot[[method]] <- df
}
has_annotations <- length(all_annot) > 0

bottom_annot <- plot_spacer()
if (has_annotations) {
    annot_df <- bind_rows(all_annot)
    annot_df$method <- factor(annot_df$method, levels = method_levels)
    ct_order <- annot_df %>%
        group_by(predicted_cell_type) %>%
        summarise(med = median(predicted_cell_type_confidence, na.rm = TRUE), .groups = "drop") %>%
        arrange(desc(med)) %>%
        pull(predicted_cell_type)
    annot_df$predicted_cell_type <- factor(annot_df$predicted_cell_type, levels = ct_order)

    bottom_annot <- ggplot(annot_df, aes(x = predicted_cell_type,
                                          y = predicted_cell_type_confidence,
                                          fill = predicted_cell_type)) +
        geom_boxplot(outlier.shape = NA, width = 0.6, linewidth = 0.3) +
        facet_wrap(~method, ncol = length(levels(annot_df$method))) +
        coord_cartesian(ylim = c(0, 1)) +
        labs(title = "Prediction Confidence by Cell Type", x = NULL, y = "Confidence") +
        scale_fill_manual(values = ditto_colors) +
        theme_minimal(base_size = 8) +
        theme(axis.text.x   = element_text(angle = 40, hjust = 1),
              legend.position = "none",
              strip.text    = element_text(size = 8, face = "bold"),
              plot.title    = element_text(size = 9, face = "bold"))
}

page1 <- (plot_spacer() / top_row / bottom_row / bottom_annot) +
    plot_layout(heights = c(0.05, 1, 1, 0.4)) +
    plot_annotation(
        title = sprintf("Segmentation QC Report — %s", sample_id),
        theme = theme(plot.title = element_text(size = 11, face = "bold"))
    )

p_morpho_plots <- NULL
if (length(all_morpho) > 0) {
    morpho_df <- bind_rows(all_morpho)
    morpho_df$method <- factor(morpho_df$method, levels = method_levels)
    available_morpho <- intersect(morpho_metrics, colnames(morpho_df))

    if (length(available_morpho) > 0) {
        morpho_long <- morpho_df %>%
            select(method, all_of(available_morpho)) %>%
            pivot_longer(-method, names_to = "metric", values_to = "value") %>%
            filter(is.finite(value))

        p_morpho_plots <- lapply(available_morpho, function(m) {
            sub_df <- morpho_long[morpho_long$metric == m, ]
            q99 <- quantile(sub_df$value, 0.99, na.rm = TRUE)
            ggplot(sub_df, aes(x = method, y = value, fill = method)) +
                geom_violin(trim = TRUE) +
                geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
                coord_cartesian(ylim = c(0, q99)) +
                labs(title = m, x = NULL, y = NULL) +
                fill_scale + tt +
                theme(axis.text.x = element_text(angle = 25, hjust = 1),
                      aspect.ratio = 1.0)
        })
    }
}

# Write each dynamic page as its own file so Python can interleave guide pages.
pdf(qc_page_pdf, width = 8.5, height = 11)
    print(page1)
dev.off()
cat(sprintf("[INFO] QC page saved: %s\\n", qc_page_pdf))

if (!is.null(p_morpho_plots) && length(p_morpho_plots) > 0) {
    morpho_page <- (plot_spacer() / wrap_plots(p_morpho_plots, ncol = 3) / plot_spacer()) +
        plot_layout(heights = c(0.05, 1, 0.05)) +
        plot_annotation(
            title = "Morphological Metrics",
            subtitle = "Per-cell distributions computed from segmentation boundary geometry",
            theme = theme(plot.title = element_text(size = 11, face = "bold"))
        )
    pdf(morpho_page_pdf, width = 8.5, height = 11)
        print(morpho_page)
    dev.off()
    cat(sprintf("[INFO] Morpho page saved: %s\\n", morpho_page_pdf))
}

# ── Cell type page (only if annotation CSVs were loaded) ──
if (has_annotations) {
    # Cell type composition: stacked % and absolute count side by side
    comp_df <- annot_df %>%
        group_by(method, predicted_cell_type) %>%
        summarise(n = n(), .groups = "drop") %>%
        group_by(method) %>%
        mutate(pct = 100 * n / sum(n)) %>%
        ungroup()

    p_comp_pct <- ggplot(comp_df, aes(x = method, y = pct, fill = predicted_cell_type)) +
        geom_col(position = "stack", width = 0.65) +
        scale_fill_manual(values = ditto_colors) +
        labs(title = "Cell Type Composition (%)", x = NULL, y = "% Cells", fill = "Cell Type") +
        theme_minimal(base_size = 9) +
        theme(plot.title     = element_text(size = 9, face = "bold"),
              legend.text    = element_text(size = 7),
              legend.key.size = unit(0.35, "cm"),
              axis.text.x    = element_text(angle = 20, hjust = 1))

    p_comp_n <- ggplot(comp_df, aes(x = predicted_cell_type, y = n, fill = method)) +
        geom_col(position = "dodge", width = 0.7) +
        scale_fill_manual(values = ditto_colors) +
        labs(title = "Cell Count by Type", x = NULL, y = "# Cells", fill = "Method") +
        theme_minimal(base_size = 9) +
        theme(plot.title     = element_text(size = 9, face = "bold"),
              axis.text.x    = element_text(angle = 40, hjust = 1),
              legend.text    = element_text(size = 7),
              legend.key.size = unit(0.35, "cm"))

    # Per-cell-type confidence violin by method
    p_conf_violin <- ggplot(annot_df,
                            aes(x = method, y = predicted_cell_type_confidence, fill = method)) +
        geom_violin(trim = TRUE, scale = "width") +
        geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA) +
        coord_cartesian(ylim = c(0, 1)) +
        facet_wrap(~predicted_cell_type, ncol = 4) +
        labs(title = "Prediction Confidence per Cell Type", x = NULL, y = "Confidence") +
        scale_fill_manual(values = ditto_colors) +
        theme_minimal(base_size = 8) +
        theme(axis.text.x    = element_text(angle = 25, hjust = 1),
              legend.position = "none",
              strip.text      = element_text(size = 7, face = "bold"),
              plot.title      = element_text(size = 9, face = "bold"))

    top_comp <- (p_comp_pct | p_comp_n) + plot_layout(widths = c(0.38, 0.62))

    celltype_page <- (plot_spacer() / top_comp / p_conf_violin / plot_spacer()) +
        plot_layout(heights = c(0.05, 1, 2, 0.05)) +
        plot_annotation(
            title    = "Cell Type Annotations",
            subtitle = "XGBoost rank-gene classifier predictions",
            theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                             plot.subtitle = element_text(size = 9, color = "gray40"))
        )

    pdf(celltype_page_pdf, width = 8.5, height = 11)
        print(celltype_page)
    dev.off()
    cat(sprintf("[INFO] Cell type page saved: %s\\n", celltype_page_pdf))
}
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


def load_boundary_geodataframe(method_output_dir: Path, adata) -> Optional[object]:
    """Load cell boundary geometries as a GeoDataFrame, indexed by cell_id.

    Handles both GeoParquet format (proseg/baysor sopa output) and Xenium native
    format (regular parquet with vertex_x, vertex_y, cell_id columns).
    Returns None if boundaries are unavailable.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    boundary_parquet = next(
        (p for p in [
            method_output_dir / "cell_boundaries.parquet",
            method_output_dir / "cell_segmentation" / "cell_boundaries.parquet",
        ] if p.exists()),
        None,
    )
    if boundary_parquet is None:
        print(f"[INFO] No cell_boundaries.parquet found in {method_output_dir}")
        return None

    cell_ids = set(adata.obs_names.astype(str))

    # Try GeoParquet first (proseg/baysor sopa output)
    try:
        gdf = gpd.read_parquet(boundary_parquet)
        gdf.index = gdf.index.astype(str)
        gdf = gdf[gdf.index.isin(cell_ids)].copy()

        # Resolve MultiPolygon: keep largest component
        def _resolve(geom):
            if isinstance(geom, MultiPolygon):
                return max(geom.geoms, key=lambda p: p.area)
            return geom
        gdf["geometry"] = gdf["geometry"].apply(_resolve)

        print(f"[INFO] Loaded GeoParquet boundaries: {len(gdf)} shapes")
        return gdf
    except Exception as e:
        if "geo metadata" not in str(e).lower() and "missing geo" not in str(e).lower():
            print(f"[WARN] Could not load boundary GeoParquet: {e}")
            return None

    # Xenium native format: regular parquet with vertex_x, vertex_y, cell_id
    try:
        df = pd.read_parquet(boundary_parquet)
        if not all(c in df.columns for c in ["vertex_x", "vertex_y", "cell_id"]):
            print(f"[WARN] Unexpected Xenium boundary parquet columns: {list(df.columns)}")
            return None

        df["cell_id"] = df["cell_id"].astype(str)
        df = df[df["cell_id"].isin(cell_ids)]

        polys = {}
        for cell_id, group in df.groupby("cell_id"):
            coords = list(zip(group["vertex_x"], group["vertex_y"]))
            if len(coords) >= 3:
                polys[cell_id] = Polygon(coords)

        gdf = gpd.GeoDataFrame(
            {"geometry": list(polys.values())},
            index=list(polys.keys()),
        )
        print(f"[INFO] Loaded Xenium native boundaries: {len(gdf)} shapes")
        return gdf
    except Exception as e:
        print(f"[WARN] Could not load Xenium native boundary parquet: {e}")
        return None


def load_nucleus_geodataframe(sample_dir: Path) -> Optional[object]:
    """Load all nucleus boundary geometries from a Xenium raw sample directory.

    Loads all nuclei (unfiltered) so they can be spatially joined against any
    reseg method's cell boundaries regardless of cell ID scheme.
    Returns None if not found.
    """
    nucleus_parquet = next(
        (p for p in [
            sample_dir / "nucleus_segmentation" / "nucleus_boundaries.parquet",
            sample_dir / "nucleus_boundaries.parquet",
        ] if p.exists()),
        None,
    )
    if nucleus_parquet is None:
        return None

    try:
        from shapely.geometry import Polygon
        import geopandas as gpd

        df = pd.read_parquet(nucleus_parquet)
        if not all(c in df.columns for c in ["vertex_x", "vertex_y", "cell_id"]):
            return None

        df["cell_id"] = df["cell_id"].astype(str)

        polys = {}
        for cell_id, group in df.groupby("cell_id"):
            coords = list(zip(group["vertex_x"], group["vertex_y"]))
            if len(coords) >= 3:
                polys[cell_id] = Polygon(coords)

        gdf = gpd.GeoDataFrame(
            {"geometry": list(polys.values())},
            index=list(polys.keys()),
        )
        print(f"[INFO] Loaded nucleus boundaries: {len(gdf)} shapes")
        return gdf
    except Exception as e:
        print(f"[WARN] Could not load nucleus boundaries: {e}")
        return None


def compute_morphological_metrics(gdf, adata, nucleus_gdf=None) -> Optional[pd.DataFrame]:
    """Compute per-cell morphological metrics from shapely polygon geometry.

    Metrics:
        cell_area    : polygon area
        perimeter    : polygon perimeter
        elongation   : major_axis / minor_axis (minimum rotated rectangle; 1 = circular)
        circularity  : 4π * area / perimeter²  (1 = perfect circle)
        compactness  : √(4π * area) / perimeter (isoperimetric quotient square root)
        eccentricity : √(1 - (minor/major)²)   (0 = circular, 1 = line)
        solidity     : area / convex_hull_area
        convexity    : convex_hull_perimeter / perimeter
        density      : area / bounding_box_area
        nuclear_ratio: nucleus_area / cell_area  (spatial join — works across any cell ID scheme)
    """
    # Pre-compute nuclear areas via spatial join so all methods get nuclear_ratio,
    # regardless of whether their cell IDs match the Xenium nucleus IDs.
    # Strategy: project each nucleus centroid into whatever cell polygon contains it,
    # then use that nucleus's area. Cells with multiple nuclei keep the largest.
    nuc_area_by_cell = {}
    if nucleus_gdf is not None and len(nucleus_gdf) > 0:
        try:
            import geopandas as gpd
            nuc_cents = nucleus_gdf.copy()
            nuc_cents["_nuc_area"] = nucleus_gdf.geometry.area
            nuc_cents["geometry"] = nucleus_gdf.geometry.centroid

            joined = gpd.sjoin(
                nuc_cents[["geometry", "_nuc_area"]],
                gdf[["geometry"]],
                how="inner",
                predicate="within",
            )
            nuc_area_by_cell = (
                joined["_nuc_area"]
                .groupby(joined["index_right"])
                .max()
                .to_dict()
            )
            print(f"[INFO] Nuclear areas matched to {len(nuc_area_by_cell)} / {len(gdf)} cells")
        except Exception as e:
            print(f"[WARN] Nuclear area spatial join failed: {e}")

    rows = []
    for cell_id, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            area = geom.area
            peri = geom.length

            # Minimum rotated rectangle → major / minor axes
            rect = geom.minimum_rotated_rectangle
            rc   = list(rect.exterior.coords)
            s0   = math.sqrt((rc[1][0] - rc[0][0])**2 + (rc[1][1] - rc[0][1])**2)
            s1   = math.sqrt((rc[2][0] - rc[1][0])**2 + (rc[2][1] - rc[1][1])**2)
            major = max(s0, s1)
            minor = min(s0, s1)

            elongation   = major / minor if minor > 0 else float("nan")
            eccentricity = math.sqrt(1 - (minor / major) ** 2) if major > 0 else float("nan")
            circularity  = (4 * math.pi * area / peri ** 2) if peri > 0 else float("nan")
            compactness  = (math.sqrt(4 * math.pi * area) / peri) if peri > 0 else float("nan")

            hull      = geom.convex_hull
            solidity  = area / hull.area if hull.area > 0 else float("nan")
            convexity = hull.length / peri if peri > 0 else float("nan")

            b        = geom.bounds  # (minx, miny, maxx, maxy)
            bbox_a   = (b[2] - b[0]) * (b[3] - b[1])
            density  = area / bbox_a if bbox_a > 0 else float("nan")

            centroid = geom.centroid
            rec = {
                "cell_id":          str(cell_id),
                "cell_area":        area,
                "perimeter":        peri,
                "elongation":       elongation,
                "circularity":      circularity,
                "compactness":      compactness,
                "eccentricity":     eccentricity,
                "solidity":         solidity,
                "convexity":        convexity,
                "density":          density,
                "cell_centroid_x":  centroid.x,
                "cell_centroid_y":  centroid.y,
            }

            if nuc_area_by_cell:
                nuc_area = nuc_area_by_cell.get(cell_id)
                if nuc_area is not None:
                    rec["nuclear_ratio"] = nuc_area / area if area > 0 else float("nan")

            rows.append(rec)
        except Exception:
            continue

    if not rows:
        print("[WARN] compute_morphological_metrics: no valid geometries processed")
        return None
    df = pd.DataFrame(rows)
    print(f"[INFO] Morphological metrics computed: {len(df)} cells, "
          f"columns: {[c for c in df.columns if c != 'cell_id']}")
    return df


def export_for_r(adata, method: str, qc_dir: Path) -> tuple:
    """Export counts, coords, and obs metadata for the CellSPA R script.

    Returns (counts_path, coords_path, meta_path).
    """
    from scipy.io import mmwrite
    import scipy.sparse as sp

    counts_path = qc_dir / f"counts_{method}.mtx"
    coords_path = qc_dir / f"coords_{method}.csv"
    meta_path   = qc_dir / f"meta_{method}.csv"

    # Write genes × cells MTX
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

    # Export numeric obs metadata as R colData.
    # Write cell IDs as row names so SPE colnames match expected cell identifiers.
    numeric_obs = adata.obs.select_dtypes(include="number")
    if numeric_obs.shape[1] == 0:
        numeric_obs = pd.DataFrame({"placeholder": np.zeros(len(adata), dtype=np.float32)},
                                   index=adata.obs_names)
    else:
        numeric_obs = numeric_obs.copy()
        numeric_obs.index = adata.obs_names
    numeric_obs.to_csv(meta_path, index=True)

    return counts_path, coords_path, meta_path


def run_cellspa(adata, method: str, qc_dir: Path) -> bool:
    """Export data, write and run the CellSPA R script for one method. Returns True on success."""
    counts_path, coords_path, meta_path = export_for_r(adata, method, qc_dir)
    if coords_path is None:
        return False

    r_script = qc_dir / f"run_cellspa_{method}.R"
    r_script.write_text(CELLSPA_R_SCRIPT)

    cmd = [
        "Rscript", str(r_script),
        str(counts_path), str(coords_path), str(meta_path),
        method, str(qc_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] CellSPA R failed for {method}:\n{result.stderr}")
        return False
    return True


def _stitch_pdfs(pages: list, output: Path):
    """Combine a list of PDF paths into one file. Skips paths that don't exist."""
    existing = [p for p in pages if Path(p).exists()]
    if not existing:
        return
    try:
        from pypdf import PdfWriter, PdfReader
        writer = PdfWriter()
        for p in existing:
            writer.append(PdfReader(str(p)))
        with open(output, "wb") as f:
            writer.write(f)
    except ImportError:
        # Fallback: ghostscript (usually available on HPC)
        result = subprocess.run(
            ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
             f"-sOutputFile={output}", *[str(p) for p in existing]],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ghostscript PDF stitch failed: {result.stderr}")


def generate_pdf_report(comparison_csv: Path, qc_dir: Path, sample_id: str, guide_dir: Path,
                        cv_metrics_path: Optional[Path] = None):
    """Generate the QC report by interleaving guide pages with R-generated plots.

    Page order:
        1. segmentation_qc_guide_pg1.pdf  (static guide)
        2. R-generated QC summary
        3. morpholgical_metrics_pg3.pdf   (static guide, if morpho data available)
        4. R-generated morphological plots (if morpho data available)
        5. R-generated cell type composition page (if annotation CSVs available)
        6. R-generated prediction confidence by cell type page (if annotation CSVs available)
        7. R-generated segger metrics page (if segger CSVs available)
    """
    pdf_path      = qc_dir / "qc_report.pdf"
    qc_page        = qc_dir / "_temp_qc_page.pdf"
    morpho_page    = qc_dir / "_temp_morpho_page.pdf"
    morpho_ct_pages = [qc_dir / f"_temp_morpho_page_morpho_ct_{i}.pdf" for i in range(1, 4)]
    celltype_page  = qc_dir / "_temp_celltype_page.pdf"
    conf_page      = qc_dir / "_temp_celltype_page_conf.pdf"
    segger_page    = qc_dir / "_temp_celltype_page_segger.pdf"

    if not _QC_REPORT_R_SCRIPT.exists():
        print(f"[ERROR] R script not found: {_QC_REPORT_R_SCRIPT}")
        return pdf_path

    r_args = ["Rscript", str(_QC_REPORT_R_SCRIPT),
              str(comparison_csv), str(qc_dir),
              str(qc_page), sample_id, str(morpho_page), str(celltype_page)]
    if cv_metrics_path and cv_metrics_path.exists():
        r_args.append(str(cv_metrics_path))
    result = subprocess.run(r_args, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] PDF report failed:\n{result.stderr}")
        return pdf_path

    guide_pg1 = guide_dir / "segmentation_qc_guide_pg1.pdf"
    guide_pg3 = guide_dir / "morpholgical_metrics_pg3.pdf"

    pages = [guide_pg1, qc_page]
    if morpho_page.exists():
        pages += [guide_pg3, morpho_page]
    for p in morpho_ct_pages:
        if p.exists():
            pages.append(p)
    if celltype_page.exists():
        pages.append(celltype_page)
    if conf_page.exists():
        pages.append(conf_page)
    if segger_page.exists():
        pages.append(segger_page)

    try:
        _stitch_pdfs(pages, pdf_path)
    except Exception as e:
        print(f"[WARN] PDF stitch failed ({e}) — falling back to R output only")
        if qc_page.exists():
            import shutil
            shutil.copy(qc_page, pdf_path)

    # Compress final PDF via ghostscript (reduces file size for email)
    compressed = qc_dir / "_temp_compressed.pdf"
    try:
        result = subprocess.run(
            ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
             "-dPDFSETTINGS=/ebook", f"-sOutputFile={compressed}", str(pdf_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and compressed.exists() and compressed.stat().st_size > 0:
            compressed.replace(pdf_path)
            print(f"[INFO] PDF compressed: {pdf_path.stat().st_size / 1e6:.1f} MB")
        else:
            compressed.unlink(missing_ok=True)
    except Exception:
        pass  # ghostscript not available — use uncompressed PDF

    for p in [qc_page, morpho_page, celltype_page, conf_page, segger_page, *morpho_ct_pages]:
        if p.exists():
            p.unlink()

    return pdf_path


def _run_multi_sample_qc(args):
    """Run QC on concatenated h5ads from multiple samples under a single slide_dir."""
    import scanpy as sc
    import scipy.sparse as sp

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "cellspa_qc")
    output_base = get_output_base_override(cfg)

    slide_dir = Path(args.slide_dir)
    sample_ids = args.sample_ids or []
    sample_dirs_map = {}  # sample_id → raw sample dir Path
    if args.sample_dirs:
        for sid, sdir in zip(sample_ids, args.sample_dirs):
            sample_dirs_map[sid] = Path(sdir)

    base_dir = Path(output_base) / slide_dir.name if output_base else slide_dir
    qc_dir = base_dir / "qc" / "combined"
    qc_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    n = len(sample_ids)
    label = ", ".join(sample_ids)
    print("=" * 60)
    print(f"  Multi-Sample QC — {n} samples: {label}")
    print(f"  Slide: {slide_dir}")
    print(f"  Output: {qc_dir}")
    print("=" * 60)

    # Discover completed methods across all samples
    methods_multi = discover_completed_methods_multi(slide_dir, sample_ids, output_base)
    print(f"[INFO] Methods found: {', '.join(methods_multi.keys()) or 'none'}")

    # Build method_data: {method: (concatenated_adata, representative_output_dir)}
    # Also track per-sample info for morphological metrics.
    method_data = {}
    # {method: [(sid, per_sample_adata, output_dir), ...]} — used for morpho
    method_per_sample: dict = {}

    # --- Xenium baselines (concatenate across samples) ---
    baseline_per_sample = []
    for sid in sample_ids:
        raw_dir = sample_dirs_map.get(sid)
        if raw_dir:
            bl = load_xenium_baseline(raw_dir)
            if bl is not None:
                bl.obs["sample_id"] = sid
                baseline_per_sample.append((sid, bl, raw_dir))
    if baseline_per_sample:
        if len(baseline_per_sample) == 1:
            combined_bl = baseline_per_sample[0][1]
        else:
            combined_bl = sc.concat(
                [a for _, a, _ in baseline_per_sample],
                keys=[sid for sid, _, _ in baseline_per_sample],
                label="sample_id",
                index_unique="-",
            )
        method_data["xenium"] = (combined_bl, sample_dirs_map.get(sample_ids[0]))
        method_per_sample["xenium"] = baseline_per_sample
        print(f"[INFO] Xenium baseline: {combined_bl.n_obs} cells (combined)")

    # --- Reseg methods (concatenate across samples) ---
    for method, sample_h5ads in methods_multi.items():
        per_sample = []
        for sid in sample_ids:
            if sid not in sample_h5ads:
                print(f"[WARN] {method}: no h5ad for {sid} — skipping that sample")
                continue
            try:
                a = sc.read_h5ad(sample_h5ads[sid])
                a.obs["sample_id"] = sid
                per_sample.append((sid, a, sample_h5ads[sid].parent))
            except Exception as e:
                print(f"[WARN] {method}/{sid}: could not load h5ad: {e}")

        if not per_sample:
            continue

        if len(per_sample) == 1:
            combined = per_sample[0][1]
        else:
            combined = sc.concat(
                [a for _, a, _ in per_sample],
                keys=[sid for sid, _, _ in per_sample],
                label="sample_id",
                index_unique="-",
            )
        method_data[method] = (combined, per_sample[0][2])
        method_per_sample[method] = per_sample
        print(f"[INFO] {method}: {combined.n_obs} cells (combined from {len(per_sample)} samples)")

    if not method_data:
        print("[WARN] No data found for any method — nothing to QC")
        save_run_metadata(qc_dir, "qc", method_cfg, time.time() - t_start)
        return

    # --- Annotation CSVs: concatenate from each sample (with sample-prefixed cell IDs) ---
    import shutil as _shutil
    for method, (adata, output_dir) in method_data.items():
        all_annot_rows = []
        for sid in sample_ids:
            if method == "xenium":
                annot_csv = base_dir / "xenium_export_reseg" / sid / f"{sid}_predicted_celltypes.csv"
            else:
                method_reseg = base_dir / f"{method}_reseg" / sid
                annot_csv = method_reseg / f"{sid}_predicted_celltypes.csv"
            if not annot_csv.exists():
                continue
            try:
                df = pd.read_csv(annot_csv, index_col=0)
                df.index = (sid + "-" + df.index.astype(str)).astype(str)
                all_annot_rows.append(df)
            except Exception as e:
                print(f"[WARN] {method}/{sid} annotation CSV: {e}")

        if all_annot_rows:
            merged_annot = pd.concat(all_annot_rows)
            merged_annot.to_csv(qc_dir / f"annotations_{method}.csv")
            print(f"[INFO] Annotations merged for {method}: {len(merged_annot)} cells")
            # Merge into adata.obs
            adata.obs.index = adata.obs.index.astype(str)
            merged_annot.index = merged_annot.index.astype(str)
            for col in ["predicted_cell_type", "predicted_cell_type_confidence"]:
                if col in merged_annot.columns:
                    adata.obs[col] = merged_annot[col].reindex(adata.obs.index)

    # Total transcripts (sum across all samples)
    total_transcripts = None
    _total = 0
    for sid, raw_dir in sample_dirs_map.items():
        t = count_total_transcripts(raw_dir)
        if t:
            _total += t
    if _total > 0:
        total_transcripts = _total
        print(f"[INFO] Total transcripts (all samples): {total_transcripts:,}")

    # Nucleus GDF is per-sample and spatial — skip for multi-sample concatenated QC
    nucleus_gdf = None

    cellspa_results = []
    title_label = f"{n} samples"

    for method, (adata, output_dir) in method_data.items():
        print(f"\n── {method} ──")
        print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

        @timed(f"CellSPA metrics: {method}")
        def _cellspa():
            return run_cellspa(adata, method, qc_dir)
        success = _cellspa()

        if success:
            csv = qc_dir / f"cellspa_{method}.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if total_transcripts:
                    X = adata.X
                    assigned = int(X.sum()) if sp.issparse(X) else int(np.sum(X))
                    df["pct_transcripts_captured"] = round(100.0 * assigned / total_transcripts, 2)
                    df.to_csv(csv, index=False)
                cellspa_results.append(df)

        @timed(f"Morphological metrics: {method}")
        def _morpho(method=method):
            per_sample = method_per_sample.get(method, [])
            all_morpho_dfs = []
            for sid, sample_adata, out_dir in per_sample:
                gdf = load_boundary_geodataframe(out_dir, sample_adata)
                if gdf is None or len(gdf) == 0:
                    print(f"[INFO] {method}/{sid}: no boundaries — skipping morpho")
                    continue
                # Load nucleus boundaries for this sample independently (no spatial overlap)
                raw_dir = sample_dirs_map.get(sid)
                nuc_gdf = load_nucleus_geodataframe(raw_dir) if raw_dir else None
                mdf = compute_morphological_metrics(gdf, sample_adata, nucleus_gdf=nuc_gdf)
                if mdf is not None:
                    # Prefix cell_id to match concatenated obs_names (sample_id-cellid)
                    mdf["cell_id"] = sid + "-" + mdf["cell_id"].astype(str)
                    all_morpho_dfs.append(mdf)
            if all_morpho_dfs:
                combined_morpho = pd.concat(all_morpho_dfs, ignore_index=True)
                morpho_path = qc_dir / f"morpho_{method}.csv"
                combined_morpho.to_csv(morpho_path, index=False)
                print(f"[INFO] Morpho saved: {morpho_path.name} ({len(combined_morpho)} cells)")
        _morpho()

        generate_qc_plots(adata, method, qc_dir)

    print("\n── Segger Metrics ──")
    compute_segger_metrics(method_data, qc_dir, base_dir,
                           reference_path=args.reference_path or "")

    if cellspa_results:
        comparison = pd.concat(cellspa_results, ignore_index=True)
        comparison_csv = qc_dir / "cellspa_comparison.csv"
        comparison.to_csv(comparison_csv, index=False)
        print(f"\n── CellSPA Comparison ──")
        print(comparison.to_string(index=False))

        guide_dir = Path(__file__).parents[2] / "guide_pgs"
        ref_path = args.reference_path or ""
        if ref_path:
            _ref_stem = Path(ref_path).stem
            _cv_cache = base_dir / f"classifier_cache_{_ref_stem}"
        else:
            _cv_cache = base_dir / "classifier_cache"
        cv_metrics_path = _cv_cache / "cv_metrics.json"

        @timed("Generate PDF report")
        def _pdf():
            return generate_pdf_report(
                comparison_csv, qc_dir,
                f"Combined ({title_label})",
                guide_dir,
                cv_metrics_path=cv_metrics_path,
            )
        pdf_path = _pdf()
        print(f"[INFO] Report: {pdf_path}")

    elapsed = time.time() - t_start
    save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
    print(f"\n[DONE] Multi-Sample QC — {title_label} — {elapsed / 60:.1f} min")


@timed("Generate QC plots")
def generate_qc_plots(adata, method_name: str, output_dir: Path):
    """Generate violin + scatter QC plots."""
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_genes = adata.n_vars
    percent_top = [n for n in (50, 100, 200, 500) if n <= n_genes]
    sc.pp.calculate_qc_metrics(adata, percent_top=percent_top or None, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"QC — {method_name}", fontsize=14)
    for ax, metric, label in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
        ["Unique Genes per Cell", "Total Counts", "% in Top 50 Genes"],
    ):
        if metric in adata.obs.columns:
            ax.violinplot(adata.obs[metric].values, showmedians=True)
            ax.set_title(label)
            ax.set_ylabel(label)
    plt.tight_layout()
    fig.savefig(output_dir / f"qc_violin_{method_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_df = adata.obs[["total_counts", "n_genes_by_counts"]].dropna()
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(5000, random_state=42)
    ax.scatter(scatter_df["total_counts"], scatter_df["n_genes_by_counts"], s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Total Counts")
    ax.set_ylabel("Genes Detected")
    ax.set_title(f"{method_name}: Counts vs Genes (n={len(scatter_df):,})")
    fig.savefig(output_dir / f"qc_scatter_{method_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def compute_segger_metrics(method_data: dict, qc_dir: Path, base_dir: Path,
                           reference_path: str = ""):
    """Compute segger QC metrics (contamination, MECR, sensitivity, quantized MECR area)
    for each method that has predicted_cell_type annotations.

    Requires markers.pkl and gene_pairs.pkl in classifier_cache/ (generated by the
    classifier script). Results are saved as CSVs for the R report to plot.
    """
    import pickle

    # Remove stale segger CSVs from previous runs so R never reads empty/outdated files
    for stale in qc_dir.glob("segger_*.csv"):
        stale.unlink()

    try:
        from segger_functions.metrics import (
            compute_MECR, compute_quantized_mecr_area,
            calculate_contamination, calculate_sensitivity,
        )
    except ImportError as e:
        print(f"[WARN] segger_functions not importable — skipping segger metrics: {e}")
        return

    # Marker cache sits at base_dir (slide/experiment dir level),
    # named after the reference h5ad stem to support multiple references.
    if reference_path:
        ref_stem  = Path(reference_path).stem
        cache_dir = base_dir / f"classifier_cache_{ref_stem}"
    else:
        cache_dir = base_dir / "classifier_cache"
    m_path  = cache_dir / "markers.pkl"
    gp_path = cache_dir / "gene_pairs.pkl"
    if not (m_path.exists() and gp_path.exists()):
        print(f"[INFO] Segger marker cache not found at {cache_dir} — skipping segger metrics")
        return

    with open(m_path, "rb") as f:
        markers = pickle.load(f)
    with open(gp_path, "rb") as f:
        gene_pairs = pickle.load(f)
    print(f"[INFO] Segger markers loaded: {len(markers)} cell types, {len(gene_pairs)} gene pairs")

    for method, (adata, output_dir) in method_data.items():
        if "predicted_cell_type" not in adata.obs.columns:
            continue

        print(f"[INFO] Computing segger metrics: {method}")
        # Alias predicted_cell_type to celltype_major (used by segger functions)
        adata = adata.copy()
        adata.obs["celltype_major"] = adata.obs["predicted_cell_type"].astype(str)

        # If annotation merge produced mostly NaNs (cell ID mismatch), skip segger for this method
        valid_frac = (adata.obs["celltype_major"] != "nan").mean()
        if valid_frac < 0.2:
            print(f"[WARN] {method}: only {valid_frac:.0%} cells have valid cell types "
                  f"(annotation cell ID mismatch?) — skipping segger metrics")
            continue

        # Ensure a "raw" layer exists (contamination function requires it)
        if "raw" not in adata.layers:
            import scipy.sparse as _sp
            adata.layers["raw"] = adata.X.copy() if not _sp.issparse(adata.X) else adata.X

        # Merge cell_area and centroids from morpho CSV if available
        morpho_csv = qc_dir / f"morpho_{method}.csv"
        if morpho_csv.exists():
            try:
                morpho_df = pd.read_csv(morpho_csv)
                if "cell_id" in morpho_df.columns:
                    morpho_df = morpho_df.set_index("cell_id")
                    morpho_df.index = morpho_df.index.astype(str)  # match adata.obs_names string type
                    for col in ["cell_area", "cell_centroid_x", "cell_centroid_y"]:
                        if col in morpho_df.columns:
                            adata.obs[col] = morpho_df[col].reindex(adata.obs_names).values
            except Exception as e:
                print(f"[WARN] Could not merge morpho data for {method}: {e}")

        # Add centroids from boundary geometries if still missing
        if "cell_centroid_x" not in adata.obs.columns:
            try:
                import geopandas as gpd
                boundary_parquet = next(
                    (p for p in [
                        output_dir / "cell_boundaries.parquet",
                        output_dir / "cell_segmentation" / "cell_boundaries.parquet",
                    ] if p.exists()), None)
                if boundary_parquet:
                    gdf = gpd.read_parquet(boundary_parquet)
                    gdf["cell_centroid_x"] = gdf.geometry.centroid.x
                    gdf["cell_centroid_y"] = gdf.geometry.centroid.y
                    gdf.index = gdf.index.astype(str)
                    adata.obs["cell_centroid_x"] = gdf["cell_centroid_x"].reindex(adata.obs_names).values
                    adata.obs["cell_centroid_y"] = gdf["cell_centroid_y"].reindex(adata.obs_names).values
            except Exception as e:
                print(f"[WARN] Could not add centroids for {method}: {e}")

        # Filter gene_pairs to genes present in this adata (reference may have extra genes)
        method_genes = set(adata.var_names)
        filtered_pairs = [p for p in gene_pairs
                          if (isinstance(p, (list, tuple)) and len(p) >= 2
                              and p[0] in method_genes and p[1] in method_genes)
                          or (isinstance(p, str) and p in method_genes)]
        active_pairs = filtered_pairs if filtered_pairs else gene_pairs

        # ── MECR ──
        try:
            mecr_dict = compute_MECR(adata, active_pairs)
            mecr_df = pd.DataFrame([
                {"gene1": g1, "gene2": g2, "mecr": v, "method": method}
                for (g1, g2), v in mecr_dict.items()
            ])
            if len(mecr_df) > 0:
                mecr_df.to_csv(qc_dir / f"segger_mecr_{method}.csv", index=False)
                print(f"[INFO]   MECR: mean={mecr_df['mecr'].mean():.4f}")
        except Exception as e:
            print(f"[WARN]   MECR failed for {method}: {e}")

        # ── Contamination ──
        try:
            contam_df = calculate_contamination(adata, markers, celltype_column="celltype_major")
            if len(contam_df) > 0:
                contam_df.to_csv(qc_dir / f"segger_contamination_{method}.csv")
                print(f"[INFO]   Contamination saved")
        except Exception as e:
            print(f"[WARN]   Contamination failed for {method}: {e}")

        # ── Sensitivity ──
        try:
            sens = calculate_sensitivity(adata, markers)
            rows = [{"cell_type": ct, "sensitivity": v, "method": method}
                    for ct, vals in sens.items() for v in vals]
            if rows:
                pd.DataFrame(rows).to_csv(qc_dir / f"segger_sensitivity_{method}.csv", index=False)
                print(f"[INFO]   Sensitivity saved")
        except Exception as e:
            print(f"[WARN]   Sensitivity failed for {method}: {e}")

        # ── Quantized MECR by area ──
        if "cell_area" in adata.obs.columns and adata.obs["cell_area"].notna().any():
            try:
                qmecr_df = compute_quantized_mecr_area(adata, active_pairs)
                qmecr_df["method"] = method
                if len(qmecr_df) > 0:
                    qmecr_df.to_csv(qc_dir / f"segger_mecr_area_{method}.csv", index=False)
                    print(f"[INFO]   Quantized MECR area saved")
            except Exception as e:
                print(f"[WARN]   Quantized MECR area failed for {method}: {e}")


def discover_completed_methods(slide_dir: Path, sample_id: str, output_base: str) -> dict:
    """Scan *_reseg/ dirs and return {method: h5ad_path} for those with results."""
    base = Path(output_base) / slide_dir.name if output_base else slide_dir
    found = {}
    for reseg_dir in sorted(base.glob("*_reseg")):
        method = reseg_dir.name.replace("_reseg", "")
        if method == "xenium_export":
            continue  # xenium baseline is loaded separately via load_xenium_baseline
        sample_dir = reseg_dir / sample_id
        h5ads = list(sample_dir.glob("*.h5ad")) if sample_dir.exists() else []
        if h5ads:
            found[method] = h5ads[0]
    return found


def discover_completed_methods_multi(slide_dir: Path, sample_ids: list, output_base: str) -> dict:
    """Multi-sample: return {method: {sample_id: h5ad_path}} for all found results."""
    base = Path(output_base) / slide_dir.name if output_base else slide_dir
    result = {}
    for reseg_dir in sorted(base.glob("*_reseg")):
        method = reseg_dir.name.replace("_reseg", "")
        if method == "xenium_export":
            continue
        for sample_id in sample_ids:
            sd = reseg_dir / sample_id
            h5ads = list(sd.glob("*.h5ad")) if sd.exists() else []
            if h5ads:
                result.setdefault(method, {})[sample_id] = h5ads[0]
    return result


def load_xenium_baseline(sample_dir: Path):
    """Load the original Xenium segmentation as a baseline AnnData.

    Reads cell_feature_matrix/ (10x MTX format) and attaches spatial
    coordinates from cells.csv.gz.  Returns None if the data is missing.
    """
    import scanpy as sc

    mtx_path = sample_dir / "cell_feature_matrix"
    if not mtx_path.exists():
        print("[WARN] Xenium baseline: cell_feature_matrix/ not found — skipping")
        return None

    try:
        adata = sc.read_10x_mtx(str(mtx_path), var_names="gene_symbols", cache=False)
    except Exception as e:
        print(f"[WARN] Xenium baseline: could not load cell_feature_matrix: {e}")
        return None

    # Attach spatial coordinates from cells.csv.gz
    for cells_file in [sample_dir / "cells.csv.gz", sample_dir / "cells.csv"]:
        if cells_file.exists():
            try:
                cells_df = pd.read_csv(cells_file)
                id_col = next((c for c in ["cell_id", "barcode"] if c in cells_df.columns), None)
                if id_col:
                    cells_df = cells_df.set_index(id_col)
                xy = [c for c in ["x_centroid", "y_centroid"] if c in cells_df.columns]
                if len(xy) == 2:
                    coords = cells_df.reindex(adata.obs_names)[xy].values.astype(float)
                    adata.obsm["spatial"] = coords
            except Exception as e:
                print(f"[WARN] Xenium baseline: could not attach coordinates: {e}")
            break

    return adata


def main():
    parser = argparse.ArgumentParser(description="CellSPA QC — all completed segmentation methods")
    parser.add_argument("--config", required=True)
    parser.add_argument("--slide-dir", required=True, help="Slide folder containing {method}_reseg/ dirs")
    parser.add_argument("--reference-path", default=None, help="Reference h5ad path (to locate classifier cache)")
    # Single-sample args
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-dir", default=None, help="Raw sample dir (for % transcripts captured)")
    # Multi-sample args
    parser.add_argument("--multi-sample", action="store_true",
                        help="Concatenate h5ads across all samples before QC")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                        help="Sample IDs to include (multi-sample mode)")
    parser.add_argument("--sample-dirs", nargs="*", default=None,
                        help="Raw sample dirs, parallel to --sample-ids (for xenium baseline)")
    args = parser.parse_args()

    if args.multi_sample:
        _run_multi_sample_qc(args)
        return

    if not args.sample_id:
        parser.error("--sample-id is required for single-sample mode (or use --multi-sample)")

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

    # Auto-discover completed reseg methods
    discovered = discover_completed_methods(slide_dir, args.sample_id, output_base)
    if discovered:
        print(f"[INFO] Reseg results found: {', '.join(discovered.keys())}")
    else:
        print(f"[INFO] No reseg results found under {slide_dir}")

    import scanpy as sc
    import scipy.sparse as sp

    # Build method → (adata, output_dir) — xenium baseline first, then reseg methods
    method_data = {}

    if args.sample_dir:
        baseline = load_xenium_baseline(Path(args.sample_dir))
        if baseline is not None:
            method_data["xenium"] = (baseline, Path(args.sample_dir))
            print(f"[INFO] Xenium baseline: {baseline.n_obs} cells × {baseline.n_vars} genes")

    for method, h5ad_path in discovered.items():
        try:
            method_data[method] = (sc.read_h5ad(h5ad_path), h5ad_path.parent)
        except Exception as e:
            print(f"[WARN] Could not load {method}: {e}")

    if not method_data:
        print("[WARN] No data to compare (no reseg results and no Xenium baseline found)")
        elapsed = time.time() - t_start
        save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
        sys.exit(0)

    print(f"[INFO] Methods to compare: {', '.join(method_data.keys())}\n")

    # Copy annotation CSVs from classifier output into qc_dir so the R report can find them.
    # The classifier writes {sample_id}_predicted_celltypes.csv alongside the h5ad.
    import shutil as _shutil
    base_dir = Path(output_base) / slide_dir.name if output_base else slide_dir
    for method, (adata, output_dir) in method_data.items():
        if method == "xenium":
            # Annotations for xenium baseline come from xenium_export_reseg (not the raw dir)
            annot_csv = base_dir / "xenium_export_reseg" / args.sample_id / f"{args.sample_id}_predicted_celltypes.csv"
        else:
            annot_csv = output_dir / f"{args.sample_id}_predicted_celltypes.csv"
        if annot_csv.exists():
            _shutil.copy(annot_csv, qc_dir / f"annotations_{method}.csv")
            print(f"[INFO] Annotation CSV found for {method}: {annot_csv.name}")
            # Also merge annotations into adata.obs so segger metrics can use them
            annot_df = pd.read_csv(annot_csv, index_col=0)
            annot_df.index = annot_df.index.astype(str)
            adata.obs.index = adata.obs.index.astype(str)
            for col in ["predicted_cell_type", "predicted_cell_type_confidence"]:
                if col in annot_df.columns:
                    adata.obs[col] = annot_df[col].reindex(adata.obs.index)
            # Xenium: fix cell ID mismatch between baseline and annotation CSV.
            # Baseline obs_names = barcode strings (from cell_feature_matrix/barcodes.tsv.gz).
            # Annotation CSV indices = integer positions (from spatialdata_io xenium_export h5ad).
            # Fix: read barcodes.tsv.gz in positional order; annotation index i → barcodes[i].
            if method == "xenium":
                valid_frac = adata.obs.get("predicted_cell_type", pd.Series(dtype=str)).notna().mean()
                if valid_frac < 0.05 and args.sample_dir:
                    _barcodes_path = Path(args.sample_dir) / "cell_feature_matrix" / "barcodes.tsv.gz"
                    if _barcodes_path.exists():
                        try:
                            _barcodes = pd.read_csv(_barcodes_path, header=None, compression="gzip")[0].values
                            _idx_ints = annot_df.index.astype(int)
                            _remapped = annot_df.copy()
                            _remapped.index = _barcodes[_idx_ints].astype(str)
                            for col in ["predicted_cell_type", "predicted_cell_type_confidence"]:
                                if col in _remapped.columns:
                                    adata.obs[col] = _remapped[col].reindex(adata.obs.index)
                            new_frac = adata.obs.get("predicted_cell_type", pd.Series(dtype=str)).notna().mean()
                            print(f"[INFO] Xenium annotation: mapped via barcodes.tsv.gz ({new_frac:.0%} matched)")
                        except Exception as _e:
                            print(f"[WARN] Xenium barcode mapping failed: {_e}")
                    else:
                        print("[WARN] Xenium annotation: barcodes.tsv.gz not found — xenium segger metrics will be skipped")

    # Load Xenium nucleus boundaries once — spatially joined against every method's
    # cell polygons so nuclear_ratio is available for proseg/baysor too.
    nucleus_gdf = None
    if args.sample_dir:
        nucleus_gdf = load_nucleus_geodataframe(Path(args.sample_dir))
        if nucleus_gdf is None:
            print("[INFO] No nucleus boundaries found — nuclear_ratio will be skipped")

    cellspa_results = []

    for method, (adata, output_dir) in method_data.items():
        print(f"\n── {method} ──")
        print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

        # ── CellSPA R basic QC metrics ──
        @timed(f"CellSPA metrics: {method}")
        def _cellspa():
            return run_cellspa(adata, method, qc_dir)
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

        # ── Python morphological metrics ──
        @timed(f"Morphological metrics: {method}")
        def _morpho():
            gdf = load_boundary_geodataframe(output_dir, adata)
            if gdf is None or len(gdf) == 0:
                print(f"[INFO] No boundaries available for {method} — skipping morpho metrics")
                return

            morpho_df = compute_morphological_metrics(gdf, adata, nucleus_gdf=nucleus_gdf)
            if morpho_df is not None:
                morpho_path = qc_dir / f"morpho_{method}.csv"
                morpho_df.to_csv(morpho_path, index=False)
                print(f"[INFO] Morphological metrics saved: {morpho_path.name} ({len(morpho_df)} cells)")
        _morpho()

        # Python QC plots
        generate_qc_plots(adata, method, qc_dir)

    # ── Segger QC metrics (requires classifier to have run first) ──
    print("\n── Segger Metrics ──")
    compute_segger_metrics(method_data, qc_dir, base_dir,
                           reference_path=args.reference_path or "")

    # Combined CellSPA comparison table + PDF report
    if cellspa_results:
        comparison = pd.concat(cellspa_results, ignore_index=True)
        comparison_csv = qc_dir / "cellspa_comparison.csv"
        comparison.to_csv(comparison_csv, index=False)
        print(f"\n── CellSPA Comparison ──")
        print(comparison.to_string(index=False))

        guide_dir = Path(__file__).parents[2] / "guide_pgs"

        # Discover cv_metrics.json from classifier cache (same logic as compute_segger_metrics)
        ref_path = args.reference_path or ""
        if ref_path:
            _ref_stem = Path(ref_path).stem
            _cv_cache = base_dir / f"classifier_cache_{_ref_stem}"
        else:
            _cv_cache = base_dir / "classifier_cache"
        cv_metrics_path = _cv_cache / "cv_metrics.json"

        @timed("Generate PDF report")
        def _pdf():
            return generate_pdf_report(comparison_csv, qc_dir, args.sample_id, guide_dir,
                                       cv_metrics_path=cv_metrics_path)
        pdf_path = _pdf()
        print(f"[INFO] Report: {pdf_path}")

    elapsed = time.time() - t_start
    save_run_metadata(qc_dir, "qc", method_cfg, elapsed)
    print(f"\n[DONE] CellSPA QC — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
