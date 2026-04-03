#!/usr/bin/env Rscript
# qc_report.R — Generate QC PDF pages for one sample.
#
# Usage:
#   Rscript qc_report.R <comparison_csv> <qc_dir> <qc_page_pdf> <sample_id>
#                       <morpho_page_pdf> <celltype_page_pdf>
#
# All segger metric CSVs (segger_mecr_<method>.csv, etc.) are auto-discovered
# from qc_dir if present — segger pages are only generated when those files exist.

suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(Matrix)
    library(patchwork)
    library(magick)
})

args              <- commandArgs(trailingOnly = TRUE)
comparison        <- read.csv(args[1])
coords_dir        <- args[2]
qc_page_pdf       <- args[3]
sample_id         <- args[4]
morpho_page_pdf   <- args[5]
celltype_page_pdf <- args[6]

# ── Shared helpers ──────────────────────────────────────────────────────────────

# Preserve CSV row order in all plots (xenium first, then reseg methods)
method_levels <- comparison$method
comparison$method <- factor(comparison$method, levels = method_levels)

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
    theme(plot.title  = element_text(size = 9, face = "bold"),
          legend.position = "none",
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))


# ── Page 1: CellSPA summary ─────────────────────────────────────────────────────

# Per-cell count distributions
all_cells <- list()
for (method in as.character(method_levels)) {
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

# Save scatter as PNG then embed as raster to keep PDF size small
scatter_png <- sub("\\.pdf$", "_scatter.png", qc_page_pdf)
ggsave(scatter_png, p_scatter, width = 4, height = 4, dpi = 150)
p_scatter_raster <- wrap_elements(
    grid::rasterGrob(
        as.raster(image_read(scatter_png)),
        interpolate = TRUE
    )
)

# Load morpho CSVs (needed for nucleus metric on page 1 and morpho page)
morpho_metrics <- c("cell_area", "elongation", "circularity", "compactness",
                    "eccentricity", "solidity", "convexity", "density", "nuclear_ratio")

all_morpho <- list()
for (method in as.character(method_levels)) {
    morpho_path <- file.path(coords_dir, sprintf("morpho_%s.csv", method))
    if (!file.exists(morpho_path)) next
    df <- read.csv(morpho_path)
    df$cell_id <- as.character(df$cell_id)   # ensure string for bind_rows compatibility
    df$method <- as.character(method)
    all_morpho[[method]] <- df
}

# % cells without nucleus → top row
if (length(all_morpho) > 0 && "nuclear_ratio" %in% colnames(bind_rows(all_morpho))) {
    nuc_stats <- bind_rows(lapply(all_morpho, function(df) {
        data.frame(
            method         = df$method[1],
            pct_no_nucleus = if ("nuclear_ratio" %in% colnames(df))
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
    p_ncells | p_pct | p_no_nucleus
} else if (!is.null(p_pct)) {
    p_ncells | p_pct
} else if (!is.null(p_no_nucleus)) {
    p_ncells | p_no_nucleus
} else {
    p_ncells
}
bottom_row <- p_counts | p_genes | p_scatter_raster

page1 <- (plot_spacer() / top_row / bottom_row) +
    plot_layout(heights = c(0.05, 1, 1)) +
    plot_annotation(
        title = sprintf("Segmentation QC Report — %s", sample_id),
        theme = theme(plot.title = element_text(size = 11, face = "bold"))
    )

pdf(qc_page_pdf, width = 8.5, height = 11)
    print(page1)
dev.off()
cat(sprintf("[INFO] QC page saved: %s\n", qc_page_pdf))


# ── Page 2: Morphological metrics ───────────────────────────────────────────────

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
                theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
        })

        morpho_page <- (plot_spacer() / wrap_plots(p_morpho_plots, ncol = 3) / plot_spacer()) +
            plot_layout(heights = c(0.02, 1, 0.02)) +
            plot_annotation(
                title    = "Morphological Metrics",
                subtitle = "Per-cell distributions computed from segmentation boundary geometry",
                theme    = theme(plot.title = element_text(size = 11, face = "bold"))
            )
        pdf(morpho_page_pdf, width = 11, height = 14)
            print(morpho_page)
        dev.off()
        cat(sprintf("[INFO] Morpho page saved: %s\n", morpho_page_pdf))
    }
}


# ── Pages 2b/2c/2d: Morphological metrics by cell type (3 pages) ─────────────────

all_annot_morpho <- list()
for (method in as.character(method_levels)) {
    annot_path  <- file.path(coords_dir, sprintf("annotations_%s.csv",  method))
    morpho_path <- file.path(coords_dir, sprintf("morpho_%s.csv", method))
    if (!file.exists(annot_path) || !file.exists(morpho_path)) next
    ann <- read.csv(annot_path,  stringsAsFactors = FALSE)
    mor <- read.csv(morpho_path, stringsAsFactors = FALSE)
    if (!"predicted_cell_type" %in% colnames(ann)) next
    if (!"cell_id" %in% colnames(ann)) ann$cell_id <- as.character(rownames(ann))
    if (!"cell_id" %in% colnames(mor)) mor$cell_id <- as.character(rownames(mor))
    ann$cell_id <- as.character(ann$cell_id)
    mor$cell_id <- as.character(mor$cell_id)
    merged <- inner_join(ann[, c("cell_id", "predicted_cell_type")],
                         mor, by = "cell_id")
    merged$method <- method
    all_annot_morpho[[method]] <- merged
}

if (length(all_annot_morpho) > 0) {
    am_df <- bind_rows(all_annot_morpho)
    am_df$method <- factor(am_df$method, levels = method_levels)

    # Method color mapping
    method_colors <- setNames(
        ditto_colors[seq_along(method_levels)],
        as.character(method_levels)
    )

    # Fixed metric groupings across 3 pages
    morpho_ct_groups <- list(
        list(metrics = c("cell_area", "perimeter", "elongation"),
             subtitle = "Size & Shape"),
        list(metrics = c("circularity", "compactness", "eccentricity"),
             subtitle = "Roundness & Compactness"),
        list(metrics = c("solidity", "convexity", "density"),
             subtitle = "Solidity & Density")
    )

    for (pg_i in seq_along(morpho_ct_groups)) {
        grp     <- morpho_ct_groups[[pg_i]]
        metrics <- intersect(grp$metrics, colnames(am_df))
        if (length(metrics) == 0) next

        # Sort cell types by overall median of first metric in group
        ref_metric <- metrics[1]
        ct_order <- am_df %>%
            group_by(predicted_cell_type) %>%
            summarise(med = median(.data[[ref_metric]], na.rm = TRUE), .groups = "drop") %>%
            arrange(med) %>%
            pull(predicted_cell_type)

        panels <- lapply(metrics, function(metric) {
            sub_df <- am_df %>%
                select(predicted_cell_type, method, value = all_of(metric)) %>%
                filter(!is.na(value)) %>%
                mutate(predicted_cell_type = factor(predicted_cell_type, levels = ct_order))
            # Clip x-axis to outermost whisker bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            q1  <- quantile(sub_df$value, 0.25, na.rm = TRUE)
            q3  <- quantile(sub_df$value, 0.75, na.rm = TRUE)
            iqr <- q3 - q1
            wlo <- max(min(sub_df$value, na.rm = TRUE), q1 - 1.5 * iqr)
            whi <- min(max(sub_df$value, na.rm = TRUE), q3 + 1.5 * iqr)
            ggplot(sub_df, aes(x = value, y = predicted_cell_type, fill = method)) +
                geom_boxplot(outlier.shape = NA, linewidth = 0.35,
                             position = position_dodge(preserve = "single")) +
                coord_cartesian(xlim = c(wlo, whi)) +
                scale_fill_manual(values = method_colors, name = NULL) +
                labs(title = metric, x = NULL, y = NULL) +
                theme_minimal(base_size = 9) +
                theme(plot.title    = element_text(size = 9, face = "bold"),
                      axis.text.y   = element_text(size = 7),
                      legend.position = "right",
                      legend.text   = element_text(size = 7))
        })

        morpho_ct_page <- wrap_plots(panels, ncol = 3) +
            plot_annotation(
                title    = "Morphological Metrics by Cell Type",
                subtitle = grp$subtitle,
                theme    = theme(plot.title = element_text(size = 11, face = "bold"))
            )

        morpho_ct_pdf <- sub("\\.pdf$", sprintf("_morpho_ct_%d.pdf", pg_i), morpho_page_pdf)
        pdf(morpho_ct_pdf, width = 11, height = 14)
            print(morpho_ct_page)
        dev.off()
        cat(sprintf("[INFO] Morpho-by-celltype page %d saved: %s\n", pg_i, morpho_ct_pdf))
    }
}


# ── Page 3: Cell type annotations ───────────────────────────────────────────────

all_annot <- list()
for (method in as.character(method_levels)) {
    annot_path <- file.path(coords_dir, sprintf("annotations_%s.csv", method))
    if (!file.exists(annot_path)) next
    df <- read.csv(annot_path, stringsAsFactors = FALSE)
    if (!"predicted_cell_type" %in% colnames(df)) next
    df$cell_id <- as.character(df$cell_id)
    df$method  <- as.character(method)
    all_annot[[method]] <- df
}
has_annotations <- length(all_annot) > 0

if (has_annotations) {
    annot_df <- bind_rows(all_annot)
    annot_df$method <- factor(annot_df$method, levels = method_levels)
    ct_order <- annot_df %>%
        group_by(predicted_cell_type) %>%
        summarise(med = median(predicted_cell_type_confidence, na.rm = TRUE), .groups = "drop") %>%
        arrange(desc(med)) %>%
        pull(predicted_cell_type)
    annot_df$predicted_cell_type <- factor(annot_df$predicted_cell_type, levels = ct_order)

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
        guides(fill = guide_legend(ncol = 1)) +
        theme_minimal(base_size = 9) +
        theme(plot.title      = element_text(size = 9, face = "bold"),
              legend.text     = element_text(size = 7),
              legend.key.size = unit(0.35, "cm"),
              axis.text.x     = element_text(angle = 20, hjust = 1))

    # Order cell types by total count descending (largest at top of horizontal chart)
    ct_total <- comp_df %>%
        group_by(predicted_cell_type) %>%
        summarise(total = sum(n), .groups = "drop") %>%
        arrange(desc(total))
    comp_df$predicted_cell_type <- factor(comp_df$predicted_cell_type,
                                          levels = rev(ct_total$predicted_cell_type))

    p_comp_n <- ggplot(comp_df, aes(y = predicted_cell_type, x = n, fill = method)) +
        geom_col(position = "dodge", width = 0.7) +
        scale_fill_manual(values = ditto_colors) +
        labs(title = "Cell Count by Type", y = NULL, x = "# Cells", fill = "Method") +
        theme_minimal(base_size = 9) +
        theme(plot.title      = element_text(size = 9, face = "bold"),
              legend.text     = element_text(size = 7),
              legend.key.size = unit(0.35, "cm"))

    top_comp <- (p_comp_pct | p_comp_n) + plot_layout(widths = c(0.38, 0.62))

    celltype_page <- (plot_spacer() / top_comp / plot_spacer()) +
        plot_layout(heights = c(0.05, 1, 0.05)) +
        plot_annotation(
            title    = "Cell Type Annotations",
            subtitle = "XGBoost rank-gene classifier predictions",
            theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                             plot.subtitle = element_text(size = 9, color = "gray40"))
        )

    pdf(celltype_page_pdf, width = 8.5, height = 11)
        print(celltype_page)
    dev.off()
    cat(sprintf("[INFO] Cell type page saved: %s\n", celltype_page_pdf))
}


# ── Page 4: Prediction confidence ───────────────────────────────────────────────
# Page 4a: one plot per method — cell types on x-axis (overview)
# Page 4b: one panel per cell type — methods on x-axis (detail)

if (has_annotations) {
    conf_page_pdf <- sub("\\.pdf$", "_conf.pdf", celltype_page_pdf)
    n_methods     <- length(levels(annot_df$method))

    # ── 4a: method-level overview ──
    method_conf_plots <- lapply(seq_along(levels(annot_df$method)), function(i) {
        m   <- levels(annot_df$method)[i]
        sub <- annot_df[annot_df$method == m, ]
        ggplot(sub, aes(x = predicted_cell_type, y = predicted_cell_type_confidence)) +
            geom_violin(trim = TRUE, scale = "width",
                        fill = ditto_colors[((i - 1) %% length(ditto_colors)) + 1],
                        alpha = 0.8) +
            geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA) +
            coord_cartesian(ylim = c(0, 1)) +
            labs(title = m, x = NULL, y = "Confidence") +
            theme_minimal(base_size = 8) +
            theme(axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5),
                  plot.title   = element_text(size = 9, face = "bold"))
    })

    p_conf_overview <- wrap_plots(method_conf_plots, ncol = 1)
    conf_page_a <- (plot_spacer() / p_conf_overview / plot_spacer()) +
        plot_layout(heights = c(0.03, 1, 0.03)) +
        plot_annotation(
            title    = "Prediction Confidence by Cell Type",
            subtitle = "XGBoost rank-gene classifier — confidence = max class probability",
            theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                             plot.subtitle = element_text(size = 9, color = "gray40"))
        )

    # ── 4b: per-cell-type detail ──
    p_conf_detail <- ggplot(annot_df,
                            aes(x = method, y = predicted_cell_type_confidence, fill = method)) +
        geom_violin(trim = TRUE, scale = "width") +
        geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA) +
        coord_cartesian(ylim = c(0, 1)) +
        facet_wrap(~predicted_cell_type, ncol = 4) +
        labs(title = "Confidence per Cell Type (by Method)", x = NULL, y = "Confidence") +
        scale_fill_manual(values = ditto_colors) +
        theme_minimal(base_size = 8) +
        theme(axis.text.x     = element_text(angle = 90, hjust = 1, vjust = 0.5),
              legend.position = "none",
              strip.text      = element_text(size = 7, face = "bold"),
              plot.title      = element_text(size = 9, face = "bold"))

    conf_page_b <- (plot_spacer() / p_conf_detail / plot_spacer()) +
        plot_layout(heights = c(0.05, 1, 0.05)) +
        plot_annotation(
            title    = "Confidence Detail — Per Cell Type",
            subtitle = "Each panel shows method-by-method confidence distribution for one cell type",
            theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                             plot.subtitle = element_text(size = 9, color = "gray40"))
        )

    pdf(conf_page_pdf, width = 11, height = 14)
        print(conf_page_a)
        print(conf_page_b)
    dev.off()
    cat(sprintf("[INFO] Confidence pages saved: %s\n", conf_page_pdf))
}


# ── Page 5: Segger metrics (auto-discovered) ────────────────────────────────────

# MECR
all_mecr <- list()
for (method in as.character(method_levels)) {
    p <- file.path(coords_dir, sprintf("segger_mecr_%s.csv", method))
    if (!file.exists(p)) next
    df <- read.csv(p, stringsAsFactors = FALSE)
    df$method <- method
    all_mecr[[method]] <- df
}

# Contamination
all_contam <- list()
for (method in as.character(method_levels)) {
    p <- file.path(coords_dir, sprintf("segger_contamination_%s.csv", method))
    if (!file.exists(p)) next
    df <- read.csv(p, row.names = 1, check.names = FALSE)
    all_contam[[method]] <- df
}

# Sensitivity
all_sens <- list()
for (method in as.character(method_levels)) {
    p <- file.path(coords_dir, sprintf("segger_sensitivity_%s.csv", method))
    if (!file.exists(p)) next
    df <- read.csv(p, stringsAsFactors = FALSE)
    df$method <- method
    all_sens[[method]] <- df
}

# Quantized MECR by area
all_mecr_area <- list()
for (method in as.character(method_levels)) {
    p <- file.path(coords_dir, sprintf("segger_mecr_area_%s.csv", method))
    if (!file.exists(p)) next
    df <- read.csv(p, stringsAsFactors = FALSE)
    df$method <- method
    all_mecr_area[[method]] <- df
}

has_segger <- length(all_contam) > 0 || length(all_mecr) > 0 ||
              length(all_sens) > 0   || length(all_mecr_area) > 0

if (has_segger) {
    segger_plots <- list()

    # Consistent cell type order across all heatmaps: union of all types, sorted
    all_ct_names <- sort(unique(unlist(lapply(all_contam, function(m) rownames(m)))))

    # Contamination heatmap — one per method, fixed axis order + shared color scale
    if (length(all_contam) > 0) {
        contam_global_max <- max(sapply(all_contam, function(m) max(m, na.rm = TRUE)), na.rm = TRUE)
        for (method in names(all_contam)) {
            mat <- all_contam[[method]]
            # Pad to the full cell type set so all heatmaps have same axes
            full_mat <- matrix(0, nrow = length(all_ct_names), ncol = length(all_ct_names),
                               dimnames = list(all_ct_names, all_ct_names))
            common_r <- intersect(rownames(mat), all_ct_names)
            common_c <- intersect(colnames(mat), all_ct_names)
            full_mat[common_r, common_c] <- as.matrix(mat[common_r, common_c])
            mat_long <- as.data.frame(as.table(full_mat))
            colnames(mat_long) <- c("source", "target", "contamination")
            mat_long$source <- factor(mat_long$source, levels = all_ct_names)
            mat_long$target <- factor(mat_long$target, levels = all_ct_names)
            p_heatmap <- ggplot(mat_long, aes(x = target, y = source, fill = contamination)) +
                geom_tile(color = "white", linewidth = 0.2) +
                scale_fill_gradient(low = "white", high = "#D55E00", name = "Contam.",
                                    limits = c(0, contam_global_max)) +
                coord_fixed(ratio = 1) +
                labs(title = sprintf("Contamination — %s", method), x = NULL, y = NULL) +
                theme_minimal(base_size = 5) +
                theme(axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 4),
                      axis.text.y  = element_text(size = 4),
                      plot.title   = element_text(size = 7, face = "bold"),
                      legend.key.size = unit(0.3, "cm"),
                      legend.text  = element_text(size = 5))
            segger_plots[[paste0("contam_heatmap_", method)]] <- p_heatmap
        }

        # Contamination distribution boxplot (overall, per method)
        contam_box_list <- lapply(names(all_contam), function(method) {
            mat <- all_contam[[method]]
            vals <- as.vector(as.matrix(mat))
            data.frame(method = method, contamination = vals[is.finite(vals)])
        })
        contam_box_df <- bind_rows(contam_box_list)
        contam_box_df$method <- factor(contam_box_df$method, levels = method_levels)
        contam_ymax <- (quantile(contam_box_df$contamination, 0.75, na.rm = TRUE) +
                        1.5 * IQR(contam_box_df$contamination, na.rm = TRUE)) * 1.2
        p_contam_box <- ggplot(contam_box_df, aes(x = method, y = contamination, fill = method)) +
            geom_violin(trim = TRUE) +
            geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
            coord_cartesian(ylim = c(0, contam_ymax)) +
            labs(title = "Contamination Distribution (Overall)", x = NULL, y = "Contamination") +
            fill_scale + tt
        segger_plots[["contam_box"]] <- p_contam_box

        # Neighborhood contamination boxplot: per source cell type × method
        # For each source cell type, shows distribution of contamination values into all target types
        contam_neigh_list <- lapply(names(all_contam), function(method) {
            mat <- all_contam[[method]]
            df <- as.data.frame(as.table(as.matrix(mat)))
            colnames(df) <- c("source", "target", "contamination")
            df$contamination <- as.numeric(df$contamination)
            df <- df[is.finite(df$contamination), ]
            df$method <- method
            df
        })
        contam_neigh_df <- bind_rows(contam_neigh_list)
        contam_neigh_df$method  <- factor(contam_neigh_df$method,  levels = method_levels)
        contam_neigh_df$source  <- factor(contam_neigh_df$source,  levels = all_ct_names)
        neigh_ymax <- (quantile(contam_neigh_df$contamination, 0.75, na.rm = TRUE) +
                       1.5 * IQR(contam_neigh_df$contamination, na.rm = TRUE)) * 1.2
        p_contam_neigh <- ggplot(contam_neigh_df,
                                 aes(x = source, y = contamination, fill = method)) +
            geom_boxplot(outlier.shape = NA, width = 0.6,
                         position = position_dodge(width = 0.75)) +
            coord_cartesian(ylim = c(0, neigh_ymax)) +
            scale_fill_manual(values = ditto_colors) +
            labs(title = "Neighborhood Contamination by Cell Type",
                 x = NULL, y = "Contamination", fill = "Method") +
            theme_minimal(base_size = 8) +
            theme(axis.text.x   = element_text(angle = 90, hjust = 1, vjust = 0.5),
                  plot.title    = element_text(size = 9, face = "bold"),
                  legend.text   = element_text(size = 7),
                  legend.key.size = unit(0.35, "cm"))
        segger_plots[["contam_neighborhood"]] <- p_contam_neigh
    }

    # Sensitivity boxplot
    if (length(all_sens) > 0) {
        sens_df <- bind_rows(all_sens)
        sens_df$method <- factor(sens_df$method, levels = method_levels)
        p_sens <- ggplot(sens_df, aes(x = cell_type, y = sensitivity, fill = method)) +
            geom_boxplot(outlier.shape = NA, width = 0.6,
                         position = position_dodge(width = 0.75)) +
            coord_cartesian(ylim = c(0, 1)) +
            scale_fill_manual(values = ditto_colors) +
            labs(title = "Sensitivity by Cell Type", x = NULL, y = "Sensitivity",
                 fill = "Method") +
            theme_minimal(base_size = 8) +
            theme(axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5),
                  plot.title   = element_text(size = 9, face = "bold"),
                  legend.text  = element_text(size = 7))
        segger_plots[["sensitivity"]] <- p_sens
    }

    # MECR summary
    if (length(all_mecr) > 0) {
        mecr_df <- bind_rows(all_mecr)
        mecr_df$method <- factor(mecr_df$method, levels = method_levels)
        mecr_ymax <- (quantile(mecr_df$mecr, 0.75, na.rm = TRUE) +
                      1.5 * IQR(mecr_df$mecr, na.rm = TRUE)) * 1.2
        p_mecr <- ggplot(mecr_df, aes(x = method, y = mecr, fill = method)) +
            geom_violin(trim = TRUE) +
            geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA) +
            coord_cartesian(ylim = c(0, mecr_ymax)) +
            labs(title = "MECR Distribution", x = NULL, y = "MECR") +
            fill_scale + tt
        segger_plots[["mecr"]] <- p_mecr
    }

    # Quantized MECR by area — stored separately for its own full page
    p_mecr_area <- NULL
    if (length(all_mecr_area) > 0) {
        mecr_area_df <- bind_rows(all_mecr_area)
        mecr_area_df$method <- factor(mecr_area_df$method, levels = method_levels)
        p_mecr_area <- ggplot(mecr_area_df,
                              aes(x = average_area, y = average_mecr,
                                  color = method, group = method)) +
            geom_line(linewidth = 0.8) +
            geom_point(size = 1.5) +
            geom_ribbon(aes(ymin = average_mecr - sqrt(variance_mecr),
                            ymax = average_mecr + sqrt(variance_mecr),
                            fill = method), alpha = 0.15, color = NA) +
            scale_fill_manual(values = ditto_colors) +
            scale_color_manual(values = ditto_colors) +
            coord_cartesian(xlim = c(0, 250), ylim = c(0, NA)) +
            labs(title = "MECR vs Cell Area",
                 x = "Average Cell Area", y = "Average MECR", color = "Method") +
            theme_minimal(base_size = 11) +
            theme(plot.title   = element_text(size = 11, face = "bold"),
                  legend.text  = element_text(size = 9),
                  aspect.ratio = 1)
    }

    # Entropy boxplot (if present)
    all_entropy <- list()
    for (method in as.character(method_levels)) {
        p <- file.path(coords_dir, sprintf("segger_entropy_%s.csv", method))
        if (!file.exists(p)) next
        df <- read.csv(p, stringsAsFactors = FALSE)
        df$method <- method
        all_entropy[[method]] <- df
    }
    if (length(all_entropy) > 0) {
        ent_df <- bind_rows(all_entropy)
        ent_df$method <- factor(ent_df$method, levels = method_levels)
        p_ent <- ggplot(ent_df, aes(x = method, y = entropy, fill = method)) +
            geom_boxplot(outlier.shape = NA, width = 0.6) +
            labs(title = "Transcript Entropy Distribution", x = NULL, y = "Entropy") +
            fill_scale + tt
        segger_plots[["entropy"]] <- p_ent
    }

    # Layout: all heatmaps on one portrait page (2 per row); other plots on next page
    heatmap_keys <- grep("contam_heatmap", names(segger_plots), value = TRUE)
    other_keys   <- setdiff(names(segger_plots), heatmap_keys)

    segger_page_pdf <- sub("\\.pdf$", "_segger.pdf", celltype_page_pdf)
    pdf(segger_page_pdf, width = 8.5, height = 11)

    # All heatmaps on one page — 2 per row, scales with n_methods
    if (length(heatmap_keys) > 0) {
        n_rows <- ceiling(length(heatmap_keys) / 2)
        p_heatmaps <- wrap_plots(segger_plots[heatmap_keys], ncol = 2, nrow = n_rows) +
            plot_annotation(
                title    = "Contamination Heatmaps",
                subtitle = "Cross-cell-type contamination fraction per method",
                theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                                 plot.subtitle = element_text(size = 9,  color = "gray40"))
            )
        print(p_heatmaps)
    }

    # Other plots — stacked single column
    if (length(other_keys) > 0) {
        n_other <- length(other_keys)
        other_page <- wrap_plots(segger_plots[other_keys], ncol = 1, nrow = n_other) +
            plot_annotation(
                title    = "Segger Segmentation Quality Metrics",
                subtitle = "Computed from reference-derived marker genes",
                theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                                 plot.subtitle = element_text(size = 9, color = "gray40"))
            )
        print(other_page)
    }

    # MECR by area — own full page, square, y-axis from 0
    if (!is.null(p_mecr_area)) {
        mecr_area_page <- (plot_spacer() / wrap_elements(full = p_mecr_area) / plot_spacer()) +
            plot_layout(heights = c(0.1, 1, 0.1)) +
            plot_annotation(
                title    = "MECR vs Cell Area",
                subtitle = "Average MECR per area quantile ± 1 SD; y-axis starts at 0",
                theme    = theme(plot.title    = element_text(size = 11, face = "bold"),
                                 plot.subtitle = element_text(size = 9, color = "gray40"))
            )
        print(mecr_area_page)
    }

    dev.off()
    cat(sprintf("[INFO] Segger page saved: %s\n", segger_page_pdf))
}
