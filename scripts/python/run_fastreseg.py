"""
run_fastreseg.py — FastReseg post-hoc segmentation refinement for a single sample.

Three-stage pipeline:
  Stage 1 (Python): Load source h5ad + raw transcripts → export R-readable inputs
  Stage 2 (R):      Run fastReseg_full_pipeline() → save updated counts + assignments
  Stage 3 (Python): Load R outputs → rebuild h5ad + convex-hull boundaries

Called by the generated SLURM script:
    python run_fastreseg.py --config CONFIG --sample-dir /path/to/output-XETG... \\
                            --output-dir /path/to/fastreseg_reseg/XETG... \\
                            --sample-id XETG... --source-dir /path/to/xenium_export_reseg/XETG...
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import configure_threads, save_run_metadata, timed

_FASTRESEG_R_SCRIPT = Path(__file__).resolve().parent.parent / "r" / "fastreseg.R"


# ── Stage 1: Python → R input preparation ────────────────────────────────────

def prepare_fastreseg_reference(reference_path: str, celltype_col: str, cache_dir: Path) -> Path:
    """Generate ref_profiles.csv (genes × cell types mean expression) for FastReseg."""
    import scanpy as sc
    from scipy import sparse

    profiles_csv = cache_dir / "ref_profiles.csv"
    if profiles_csv.exists():
        print(f"[INFO] FastReseg ref profiles found in cache: {cache_dir}")
        return profiles_csv

    print(f"[INFO] Generating FastReseg ref profiles from: {reference_path}")
    ref = sc.read_h5ad(reference_path)
    if celltype_col not in ref.obs.columns:
        raise ValueError(
            f"Column '{celltype_col}' not found in reference .obs. "
            f"Available: {list(ref.obs.columns)}"
        )
    X = ref.X.toarray() if sparse.issparse(ref.X) else np.array(ref.X)
    cell_types = ref.obs[celltype_col].astype(str)
    means = {ct: X[(cell_types == ct).values].mean(axis=0)
             for ct in sorted(cell_types.unique())}
    profiles_df = pd.DataFrame(means, index=ref.var_names)
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(profiles_csv)
    print(f"[INFO] Saved ref profiles: {profiles_df.shape[0]} genes × {profiles_df.shape[1]} cell types")
    return profiles_csv


@timed("Stage 1: Export inputs for R")
def export_inputs(source_dir: Path, sample_dir: Path, inputs_dir: Path):
    """Load source h5ad and raw transcripts → export R-readable files."""
    import scanpy as sc
    import scipy.sparse as sp
    import scipy.io

    inputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Find source h5ad ──
    h5ads = sorted(source_dir.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No .h5ad found in {source_dir}")
    h5ad_path = h5ads[0]
    print(f"[INFO] Loading: {h5ad_path.name}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] {adata.n_obs} cells × {adata.n_vars} genes")

    # ── counts.mtx (stored as genes × cells so R can readMM and transpose) ──
    X = adata.X
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    X_genes_cells = X.T.tocsc()
    scipy.io.mmwrite(str(inputs_dir / "counts.mtx"), X_genes_cells)
    (inputs_dir / "cells.txt").write_text("\n".join(adata.obs_names.astype(str)))
    (inputs_dir / "genes.txt").write_text("\n".join(adata.var_names.astype(str)))
    print(f"[INFO] Counts exported: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── clust.csv (cell_id → cell_type) ──
    if "predicted_cell_type" in adata.obs.columns:
        clust_df = adata.obs[["predicted_cell_type"]].copy()
        clust_df.columns = ["cell_type"]
        print(f"[INFO] Using predicted_cell_type ({clust_df['cell_type'].nunique()} types)")
    else:
        clust_df = pd.DataFrame({"cell_type": "unknown"}, index=adata.obs_names)
        print("[WARN] predicted_cell_type not found — using 'unknown' for all cells")
    clust_df.index = clust_df.index.astype(str)
    clust_df.index.name = "cell_id"
    clust_df.to_csv(inputs_dir / "clust.csv")

    # ── transcripts.csv from raw Xenium sample_dir ──
    trans_parquet = Path(sample_dir) / "transcripts.parquet"
    trans_csv_gz  = Path(sample_dir) / "transcripts.csv.gz"
    if trans_parquet.exists():
        import pyarrow.parquet as pq
        trans_df = pq.read_table(
            trans_parquet,
            columns=["transcript_id", "cell_id", "feature_name",
                     "x_location", "y_location", "z_location"],
        ).to_pandas()
    elif trans_csv_gz.exists():
        trans_df = pd.read_csv(
            trans_csv_gz,
            usecols=["transcript_id", "cell_id", "feature_name",
                     "x_location", "y_location", "z_location"],
        )
    else:
        raise FileNotFoundError(f"No transcripts file found in {sample_dir}")

    # Rename to FastReseg expected column names
    trans_df = trans_df.rename(columns={
        "feature_name": "target",
        "cell_id":      "CellId",
        "x_location":   "x",
        "y_location":   "y",
        "z_location":   "z",
    })
    # Ensure CellId is string; coerce UNASSIGNED / NaN → "0" (extracellular)
    trans_df["CellId"] = (
        pd.to_numeric(trans_df["CellId"], errors="coerce")
        .fillna(0).astype(int).astype(str)
    )
    trans_df["transcript_id"] = trans_df["transcript_id"].astype(str)

    trans_df.to_csv(inputs_dir / "transcripts.csv", index=False)
    print(f"[INFO] Transcripts exported: {len(trans_df):,} rows")

    return adata


# ── Stage 3: Build output h5ad + boundaries ───────────────────────────────────

@timed("Stage 3: Build output h5ad")
def build_output(output_dir: Path, sample_id: str):
    """Load FastReseg R outputs → AnnData + GeoParquet boundaries."""
    import scanpy as sc
    import scipy.io
    import scipy.sparse as sp
    import geopandas as gpd
    from shapely.geometry import MultiPoint

    counts_mtx  = output_dir / "updated_counts.mtx"
    genes_txt   = output_dir / "updated_genes.txt"
    cells_txt   = output_dir / "updated_cells.txt"
    cells_csv   = output_dir / "updated_cells.csv"
    trans_csv   = output_dir / "updated_transcripts.csv"

    for p in [counts_mtx, genes_txt, cells_txt, cells_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Expected R output missing: {p}")

    # ── Load count matrix (genes × cells from R → transpose to cells × genes) ──
    X = scipy.io.mmread(str(counts_mtx)).tocsr().T   # now cells × genes
    genes = (genes_txt).read_text().strip().splitlines()
    cells = (cells_txt).read_text().strip().splitlines()
    print(f"[INFO] Updated matrix: {X.shape[0]} cells × {X.shape[1]} genes")

    # ── Per-cell metadata ──
    cell_meta = pd.read_csv(cells_csv)
    # Identify cell ID column (first column, or one named CellId/cell_id/updated_cellID)
    cell_id_col = cell_meta.columns[0]
    for candidate in ["updated_cellID", "CellId", "cell_id"]:
        if candidate in cell_meta.columns:
            cell_id_col = candidate
            break
    cell_meta[cell_id_col] = cell_meta[cell_id_col].astype(str)
    cell_meta = cell_meta.set_index(cell_id_col)
    # Reindex to match cells list (some may be missing if FastReseg discarded them)
    cell_meta = cell_meta.reindex(cells)

    # ── Build AnnData ──
    adata = sc.AnnData(X=X)
    adata.obs_names = cells
    adata.var_names = genes

    for col in ["updated_celltype", "reSeg_action"]:
        if col in cell_meta.columns:
            adata.obs[col] = cell_meta[col].values

    # Spatial coords from mean transcript positions
    coord_cols = [c for c in ["x", "y"] if c in cell_meta.columns]
    if len(coord_cols) == 2:
        adata.obsm["spatial"] = cell_meta[["x", "y"]].values
        adata.obs["cell_centroid_x"] = cell_meta["x"].values
        adata.obs["cell_centroid_y"] = cell_meta["y"].values

    # ── Reconstruct boundaries from updated transcript assignments ──
    boundary_path = output_dir / "fastreseg_boundaries.parquet"
    if trans_csv.exists():
        trans_df = pd.read_csv(trans_csv)
        # Identify updated cell ID column
        cell_col = "updated_cellID"
        if cell_col not in trans_df.columns:
            for candidate in ["CellId", "cell_id"]:
                if candidate in trans_df.columns:
                    cell_col = candidate
                    break
        trans_df[cell_col] = trans_df[cell_col].astype(str)

        # Convex hull per cell from transcript x/y
        x_col = "x" if "x" in trans_df.columns else "x_location"
        y_col = "y" if "y" in trans_df.columns else "y_location"
        grouped = trans_df[trans_df[cell_col] != "0"].groupby(cell_col)
        geoms, ids = [], []
        for cell_id, grp in grouped:
            pts = list(zip(grp[x_col], grp[y_col]))
            if len(pts) >= 3:
                geom = MultiPoint(pts).convex_hull
            elif len(pts) > 0:
                geom = MultiPoint(pts).buffer(3.0)  # ~3 µm buffer for tiny cells
            else:
                continue
            geoms.append(geom)
            ids.append(cell_id)

        gdf = gpd.GeoDataFrame({"cell_id": ids, "geometry": geoms},
                               crs=None)
        gdf.to_parquet(boundary_path)
        print(f"[INFO] Boundaries saved: {len(gdf)} cells → {boundary_path.name}")
    else:
        print("[WARN] updated_transcripts.csv not found — boundaries skipped")

    # ── Save h5ad ──
    h5ad_path = output_dir / f"{sample_id}.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"[INFO] Saved: {h5ad_path.name} ({adata.n_obs} cells × {adata.n_vars} genes)")

    return adata


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FastReseg (single sample)")
    parser.add_argument("--config",   required=True)
    parser.add_argument("--sample-dir", required=True,
                        help="Raw Xenium sample directory (for transcripts.parquet)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id",  required=True)
    parser.add_argument("--source-dir", required=True,
                        help="Source segmentation results directory (h5ad)")
    parser.add_argument("--reference-path", default=None)
    parser.add_argument("--reference-celltype-col", default="cell_type")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    method_cfg = get_method_config(cfg, "fastreseg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir)
    sample_dir = Path(args.sample_dir)
    inputs_dir = output_dir / "fastreseg_inputs"

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  FastReseg — {args.sample_id}")
    print(f"  Source:  {source_dir}")
    print(f"  Raw:     {sample_dir}")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    if not source_dir.exists():
        print(f"[ERROR] Source not found: {source_dir}")
        sys.exit(1)

    # ── Reference profiles ──
    ref_profiles_path = ""
    if args.reference_path:
        cache_dir = Path(args.reference_path).parent / "reference_cache"
        try:
            ref_profiles_path = str(prepare_fastreseg_reference(
                args.reference_path, args.reference_celltype_col, cache_dir
            ))
        except Exception as e:
            print(f"[WARN] Could not prepare ref profiles: {e}")

    # ── Stage 1: Export inputs ──
    try:
        export_inputs(source_dir, sample_dir, inputs_dir)
    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}")
        sys.exit(1)

    # ── Stage 2: R FastReseg ──
    print("\n── Stage 2: FastReseg (R) ──")
    result = subprocess.run(
        ["Rscript", str(_FASTRESEG_R_SCRIPT),
         str(inputs_dir), str(output_dir), ref_profiles_path],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] R failed:\n{result.stderr}")
        sys.exit(1)

    # ── Stage 3: Build output ──
    print("\n── Stage 3: Build output h5ad ──")
    try:
        build_output(output_dir, args.sample_id)
    except Exception as e:
        print(f"[ERROR] Stage 3 failed: {e}")
        sys.exit(1)

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "fastreseg", method_cfg, elapsed)
    print(f"\n[DONE] FastReseg — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
