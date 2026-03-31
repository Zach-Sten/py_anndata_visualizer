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
    cat("[INFO] Loading ref profiles:\n", ref_profiles_csv, "\n")
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
# data.table >= 1.15.0 requires explicit c()/eval() around character column-name
# variables in by=.  FastReseg 1.1.1 uses bare symbols and get() calls.
# Approach: AST walk (primary) + deparse/gsub fallback if AST walk finds nothing.
cat(sprintf("[INFO] data.table version: %s\n", as.character(packageVersion("data.table"))))
cat(sprintf("[INFO] FastReseg version:  %s\n", as.character(packageVersion("FastReseg"))))

if (packageVersion("data.table") >= "1.15.0") {
    ns <- getNamespace("FastReseg")

    # ── Diagnostic: print every by= occurrence across all FastReseg functions ──
    cat("[DEBUG] Scanning FastReseg functions for by= calls...\n")
    for (fn_name in ls(ns, all.names = TRUE)) {
        obj <- tryCatch(get(fn_name, envir = ns), error = function(e) NULL)
        if (!is.function(obj)) next
        src_lines <- deparse(body(obj))
        by_lines  <- grep("by\\s*=", src_lines, value = TRUE)
        if (length(by_lines) > 0) {
            cat(sprintf("[BY-SCAN] %s (%d line(s)):\n", fn_name, length(by_lines)))
            for (l in by_lines) cat(sprintf("  %s\n", trimws(l)))
        }
    }

    cat("[INFO] Applying FastReseg/data.table compatibility patch (AST walk)...\n")

    # Walk AST and fix by= patterns that data.table 1.15+ rejects:
    #   by = symbol      (bare variable)  → by = c(symbol)
    #   by = get(symbol) (get() call)     → by = eval(symbol)   [NOT eval(get())]
    fix_by_calls <- function(x) {
        if (!is.call(x)) return(x)
        nms <- names(x)
        for (i in seq_along(x)) {
            nm  <- if (!is.null(nms) && i <= length(nms)) nms[[i]] else ""
            val <- tryCatch(x[[i]], error = function(e) NULL)
            if (is.null(val)) next
            if (!is.na(nm) && nzchar(nm) && nm == "by") {
                if (is.symbol(val) && nzchar(as.character(val))) {
                    # bare symbol: by = cellID_coln → by = c(cellID_coln)
                    cat(sprintf("[AST-PATCH] bare symbol: by = %s\n", as.character(val)))
                    x[[i]] <- call("c", val)
                } else if (is.call(val) &&
                           !is.null(val[[1]]) &&
                           identical(as.character(val[[1]]), "get") &&
                           length(val) == 2) {
                    # get() call: by = get(X) → by = eval(X)
                    # data.table evaluates eval() in the calling R frame.
                    # cellID_coln = "UMI_cellID" (string) → eval(cellID_coln) = "UMI_cellID" ✓
                    # eval(get(cellID_coln)) would look for a variable named UMI_cellID → ERROR
                    cat(sprintf("[AST-PATCH] get() call: by = %s → by = eval(%s)\n",
                                deparse(val), deparse(val[[2]])))
                    x[[i]] <- call("eval", val[[2]])
                }
                # already-ok forms (c, list, ., eval, key, literal) → leave as-is
            } else {
                # Wrap in tryCatch: empty/missing argument symbols cause "arg missing" errors
                x[[i]] <- tryCatch(fix_by_calls(val), error = function(e) val)
            }
        }
        x
    }

    n_patched <- 0L
    for (fn_name in ls(ns, all.names = TRUE)) {
        obj <- tryCatch(get(fn_name, envir = ns), error = function(e) NULL)
        if (!is.function(obj)) next
        old_body <- body(obj)

        # Extra diagnostics for the two known problem functions
        if (fn_name %in% c("choose_distance_cutoff", "transDF_to_perCell_data")) {
            cat(sprintf("[DEBUG-KEY] %s: body class=%s, length=%d\n",
                        fn_name, class(old_body)[1], length(old_body)))
        }

        new_body <- tryCatch(fix_by_calls(old_body), error = function(e) {
            cat(sprintf("[AST-ERR] fix_by_calls failed for %s: %s\n",
                        fn_name, conditionMessage(e)))
            NULL
        })
        if (!is.null(new_body) && !identical(old_body, new_body)) {
            tryCatch({
                body(obj) <- new_body
                assignInNamespace(fn_name, obj, ns = "FastReseg")
                n_patched <- n_patched + 1L
                cat(sprintf("[PATCH] Fixed FastReseg::%s\n", fn_name))
            }, error = function(e) {
                cat(sprintf("[PATCH] Skipped %s: %s\n", fn_name, conditionMessage(e)))
            })
        }
    }
    cat(sprintf("[INFO] AST walk patched %d FastReseg function(s)\n", n_patched))

    # ── Supplemental: deparse/gsub/parse for get() patterns ─────────────────
    # The AST walk handles bare symbols fine but silently skips by=get() calls.
    # Always run this pass to catch remaining get() patterns (safe to double-run:
    # already-fixed c(...) and eval(get(...)) won't match the patterns below).
    {
        cat("[INFO] Running deparse/gsub pass for remaining get() patterns...\n")
        n_gsub <- 0L
        for (fn_name in ls(ns, all.names = TRUE)) {
            obj <- tryCatch(get(fn_name, envir = ns), error = function(e) NULL)
            if (!is.function(obj)) next
            src <- paste(deparse(body(obj)), collapse = "\n")
            orig <- src

            # Fix 1: by = get(varname) → by = eval(varname)
            # data.table evaluates eval() in the calling R frame.
            # cellID_coln = "UMI_cellID" → eval(cellID_coln) = "UMI_cellID" ✓
            # eval(get(cellID_coln)) would do get("UMI_cellID") in calling frame → ERROR
            src <- gsub("\\bby\\s*=\\s*get\\(([^)]+)\\)",
                        "by = eval(\\1)", src, perl = TRUE)
            # Fix 2: by = barevar (not followed by open-paren) → by = c(barevar)
            # \b ensures the WHOLE identifier is matched before the lookahead fires,
            # preventing partial matches like matching "eva" out of "eval(".
            src <- gsub("\\bby\\s*=\\s*([A-Za-z_.][A-Za-z0-9_.]*)\\b(?!\\s*\\()",
                        "by = c(\\1)", src, perl = TRUE)

            if (!identical(src, orig)) {
                new_body <- tryCatch(parse(text = src)[[1]], error = function(e) {
                    cat(sprintf("[GSub-ERR] parse failed for %s: %s\n",
                                fn_name, conditionMessage(e)))
                    NULL
                })
                if (!is.null(new_body)) {
                    tryCatch({
                        body(obj) <- new_body
                        assignInNamespace(fn_name, obj, ns = "FastReseg")
                        n_gsub <- n_gsub + 1L
                        cat(sprintf("[GSub-PATCH] Fixed FastReseg::%s\n", fn_name))
                    }, error = function(e) {
                        cat(sprintf("[GSub-PATCH] Skipped %s: %s\n",
                                    fn_name, conditionMessage(e)))
                    })
                }
            }
        }
        cat(sprintf("[INFO] deparse/gsub fallback patched %d FastReseg function(s)\n", n_gsub))
    }
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
    save_intermediates   = TRUE,   # needed: populates updated_transDF_list for boundaries
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
# fastReseg_full_pipeline sometimes doesn't populate updated_transDF_list in memory
# but always writes {n}_updated_transDF.csv files to the intermediates directory.
if (!is.null(result$updated_transDF_list)) {
    all_trans <- do.call(rbind, result$updated_transDF_list)
    write.csv(all_trans, file.path(output_dir, "updated_transcripts.csv"), row.names = FALSE)
    cat(sprintf("[INFO] Saved %d updated transcripts.\n", nrow(all_trans)))
} else {
    cat("[INFO] updated_transDF_list is NULL — reading from intermediates directory...\n")
    intermediates_dir <- file.path(output_dir, "fastreseg_intermediates")
    trans_files <- sort(list.files(intermediates_dir, pattern = "_updated_transDF\\.csv$",
                                   full.names = TRUE))
    if (length(trans_files) > 0) {
        cat(sprintf("[INFO] Found %d FOV transcript file(s): %s\n",
                    length(trans_files), paste(basename(trans_files), collapse = ", ")))
        all_trans <- do.call(rbind, lapply(trans_files, read.csv, stringsAsFactors = FALSE))
        write.csv(all_trans, file.path(output_dir, "updated_transcripts.csv"), row.names = FALSE)
        cat(sprintf("[INFO] Saved %d updated transcripts from intermediates.\n", nrow(all_trans)))
    } else {
        cat("[WARN] No *_updated_transDF.csv files found in intermediates — transcript output skipped.\n")
    }
}

cat("[DONE] FastReseg R pipeline finished.\n")
