#!/usr/bin/env Rscript
# fastreseg.R — FastReseg post-hoc segmentation refinement.
#
# Usage:
#   Rscript fastreseg.R <inputs_dir> <output_dir> [ref_profiles_csv]
#
# Inputs (in inputs_dir/):
#   counts.mtx       — genes × cells sparse matrix (transposed to cells × genes on load)
#   cells.txt        — cell IDs, one per line
#   genes.txt        — gene names, one per line
#   clust.csv        — cell_id, cell_type
#   transcripts.csv  — CellId, target, x, y, z, transcript_id
#
# Outputs (in output_dir/):
#   updated_counts.mtx       — genes × cells updated count matrix
#   updated_genes.txt
#   updated_cells.txt
#   updated_cells.csv        — per-cell metadata (updated_celltype, reSeg_action, x, y)
#   updated_transcripts.csv  — per-transcript updated assignments

suppressPackageStartupMessages({
    library(Matrix)
    library(FastReseg)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript fastreseg.R <inputs_dir> <output_dir> [ref_profiles_csv]")
}

inputs_dir       <- args[1]
output_dir       <- args[2]
ref_profiles_csv <- if (length(args) >= 3 && nzchar(args[3])) args[3] else NULL

cat("=== FastReseg R pipeline ===\n")
cat("Inputs:", inputs_dir, "\n")
cat("Output:", output_dir, "\n")

# ── Load counts (cells × genes) ──────────────────────────────────────────────
cat("[INFO] Loading count matrix...\n")
counts_mat <- t(readMM(file.path(inputs_dir, "counts.mtx")))  # stored genes×cells → cells×genes
cells <- readLines(file.path(inputs_dir, "cells.txt"))
genes <- readLines(file.path(inputs_dir, "genes.txt"))
rownames(counts_mat) <- cells
colnames(counts_mat) <- genes
cat(sprintf("[INFO] Counts: %d cells × %d genes\n", nrow(counts_mat), ncol(counts_mat)))

# ── Load cluster assignments ──────────────────────────────────────────────────
cat("[INFO] Loading cluster assignments...\n")
clust_df <- read.csv(file.path(inputs_dir, "clust.csv"), row.names = 1,
                     stringsAsFactors = FALSE)
clust <- clust_df[, 1]
names(clust) <- rownames(clust_df)
cat(sprintf("[INFO] %d cells, %d unique cell types\n", length(clust), length(unique(clust))))

# ── Load transcripts ──────────────────────────────────────────────────────────
cat("[INFO] Loading transcripts...\n")
transcript_df <- read.csv(file.path(inputs_dir, "transcripts.csv"),
                          stringsAsFactors = FALSE)
transcript_df$CellId <- as.character(transcript_df$CellId)
cat(sprintf("[INFO] %d transcripts, %d unique genes\n",
            nrow(transcript_df), length(unique(transcript_df$target))))

# ── Load reference profiles (genes × cell types) ──────────────────────────────
refProfiles <- NULL
if (!is.null(ref_profiles_csv) && file.exists(ref_profiles_csv)) {
    cat("[INFO] Loading ref profiles:", ref_profiles_csv, "\n")
    ref_df      <- read.csv(ref_profiles_csv, row.names = 1, check.names = FALSE)
    refProfiles <- as.matrix(ref_df)
    cat(sprintf("[INFO] refProfiles: %d genes × %d cell types\n",
                nrow(refProfiles), ncol(refProfiles)))
}

# ── Subset to common genes ────────────────────────────────────────────────────
spatial_genes <- unique(transcript_df$target)
common_genes  <- intersect(colnames(counts_mat), spatial_genes)
if (!is.null(refProfiles)) {
    common_genes <- intersect(common_genes, rownames(refProfiles))
    refProfiles  <- refProfiles[common_genes, , drop = FALSE]
}
counts_mat <- counts_mat[, common_genes, drop = FALSE]
cat(sprintf("[INFO] Common genes after subsetting: %d\n", length(common_genes)))

# ── Compatibility patch: FastReseg + data.table >= 1.15.0 ─────────────────────
# data.table >= 1.15.0 no longer accepts `by = variable` where variable holds
# column name(s) — it must be `by = eval(variable)`. FastReseg 1.1.1 uses the
# old pattern throughout. Patch every affected function at runtime.
cat(sprintf("[INFO] data.table version: %s\n", as.character(packageVersion("data.table"))))
cat(sprintf("[INFO] FastReseg version:  %s\n", as.character(packageVersion("FastReseg"))))

if (packageVersion("data.table") >= "1.15.0") {
    cat("[INFO] Applying FastReseg/data.table compatibility patch...\n")
    ns        <- getNamespace("FastReseg")
    n_patched <- 0L
    # Match: by = <variable ending in coln or colns> — e.g. cellID_coln, spatLocs_colns
    # colns? matches both "coln" (singular) and "colns" (plural, e.g. spatLocs_colns)
    # No negative lookahead: the regex won't match "eval" since it doesn't end in colns?
    # No "already has eval" guard: gsub is safe to run on partially-patched functions
    # data.table 1.15+ requires explicit c() around character variables in by=
    # e.g. by = cellID_coln  →  by = c(cellID_coln)
    pat_old <- "by\\s*=\\s*([a-zA-Z_][a-zA-Z0-9_]*colns?\\b)"
    pat_new <- "by = c(\\1)"
    for (fn_name in ls(ns, all.names = TRUE)) {
        obj <- tryCatch(get(fn_name, envir = ns), error = function(e) NULL)
        if (!is.function(obj)) next
        src_text <- paste(deparse(body(obj)), collapse = "\n")
        if (grepl(pat_old, src_text, perl = TRUE)) {
            new_text <- gsub(pat_old, pat_new, src_text, perl = TRUE)
            n_hits <- length(gregexpr(pat_old, src_text, perl = TRUE)[[1]])
            tryCatch({
                body(obj) <- parse(text = new_text)[[1]]
                assignInNamespace(fn_name, obj, ns = "FastReseg")
                n_patched <- n_patched + 1L
                cat(sprintf("[PATCH] Fixed %d 'by' call(s) in FastReseg::%s\n", n_hits, fn_name))
            }, error = function(e) {
                cat(sprintf("[PATCH] Skipped %s: %s\n", fn_name, conditionMessage(e)))
            })
        }
    }
    cat(sprintf("[INFO] Patched %d FastReseg function(s)\n", n_patched))
}

# ── Run FastReseg ─────────────────────────────────────────────────────────────
cat("[INFO] Running fastReseg_full_pipeline()...\n")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

result <- fastReseg_full_pipeline(
    counts               = counts_mat,
    clust                = clust,
    refProfiles          = refProfiles,
    transcript_df        = transcript_df,
    transID_coln         = "transcript_id",
    transGene_coln       = "target",
    cellID_coln          = "CellId",
    spatLocs_colns       = c("x", "y", "z"),
    extracellular_cellID = "0",
    pixel_size           = 1.0,   # Xenium coords already in microns
    zstep_size           = 1.0,
    invert_y             = FALSE,
    path_to_output       = file.path(output_dir, "fastreseg_intermediates"),
    save_intermediates   = FALSE,
    return_perCellData   = TRUE,
    percentCores         = 0.75,
)
cat("[INFO] FastReseg complete.\n")

# ── Save outputs ──────────────────────────────────────────────────────────────

# Updated count matrix (genes × cells sparse)
writeMM(result$updated_perCellExprs, file.path(output_dir, "updated_counts.mtx"))
writeLines(rownames(result$updated_perCellExprs), file.path(output_dir, "updated_genes.txt"))
writeLines(colnames(result$updated_perCellExprs), file.path(output_dir, "updated_cells.txt"))
cat("[INFO] Saved updated count matrix.\n")

# Per-cell metadata
cell_dt <- as.data.frame(result$updated_perCellDT)
write.csv(cell_dt, file.path(output_dir, "updated_cells.csv"), row.names = FALSE)
cat(sprintf("[INFO] Saved %d cells metadata.\n", nrow(cell_dt)))

if ("reSeg_action" %in% colnames(cell_dt)) {
    cat("[INFO] Resegmentation actions:\n")
    print(table(cell_dt$reSeg_action))
}

# Updated transcripts (combine all FOVs)
if (!is.null(result$updated_transDF_list)) {
    all_trans <- do.call(rbind, result$updated_transDF_list)
    write.csv(all_trans, file.path(output_dir, "updated_transcripts.csv"), row.names = FALSE)
    cat(sprintf("[INFO] Saved %d updated transcripts.\n", nrow(all_trans)))
}

cat("[DONE] FastReseg R pipeline finished.\n")
