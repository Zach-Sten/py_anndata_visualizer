"""
run_xenium_export.py — Export raw Xenium baseline segmentation as h5ad.

Reads the native Xenium output (cell_feature_matrix + cells.csv.gz), attaches
spatial coordinates, and writes {sample_id}.h5ad to xenium_reseg/{sample_id}/
so the classifier and QC pipeline treat it like any other segmentation method.

Called by the generated SLURM script:
    python run_xenium_export.py --config CONFIG --sample-dir /path/to/output-XETG... \
                                --output-dir /path/to/xenium_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config
from utils.data_io import configure_threads, save_run_metadata


def main():
    parser = argparse.ArgumentParser(description="Export Xenium baseline as h5ad")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--sample-dir", required=True,
                        help="Raw Xenium output-XETG... directory")
    parser.add_argument("--output-dir", required=True,
                        help="Destination: xenium_reseg/{sample_id}/")
    parser.add_argument("--sample-id",  required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  Xenium Export — {args.sample_id}")
    print(f"  Input:  {sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    import scanpy as sc
    import pandas as pd

    # Read only the count matrix directly — avoids loading images/transcripts via spatialdata_io
    mtx_dir = sample_dir / "cell_feature_matrix"
    print(f"[INFO] Reading count matrix from {mtx_dir} ...")
    adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", make_unique=True)
    print(f"[INFO] Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Attach full cells metadata and spatial coordinates from cells.csv.gz
    for cells_file in [sample_dir / "cells.csv.gz", sample_dir / "cells.csv"]:
        if cells_file.exists():
            df = pd.read_csv(cells_file)
            id_col = next((c for c in ["cell_id", "barcode"] if c in df.columns), None)
            if id_col:
                df = df.set_index(id_col)
            # Align to adata obs order
            df = df.reindex(adata.obs_names)
            adata.obs = df.copy()

            xy = [c for c in ["x_centroid", "y_centroid"] if c in df.columns]
            if len(xy) == 2:
                adata.obsm["spatial"] = df[xy].to_numpy()
                print(f"[INFO] Spatial coordinates attached from {cells_file.name}")
            break

    # Sanitize obs: object columns with mixed types (e.g. str + NaN) break h5ad writing.
    # Cast each object column to string, replacing NaN with empty string.
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].fillna("").astype(str)

    h5ad_path = output_dir / f"{args.sample_id}.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"[INFO] Saved: {h5ad_path}")

    elapsed = time.time() - t_start
    method_cfg = cfg.get("methods", {}).get("xenium_export", {})
    save_run_metadata(output_dir, "xenium_export", method_cfg, elapsed)
    print(f"[DONE] Xenium export — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
