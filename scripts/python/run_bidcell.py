"""
run_bidcell.py — BIDCell deep learning segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_bidcell.py --config CONFIG --sample-dir /path/to/output-XETG... \
                          --output-dir /path/to/bidcell_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import configure_threads, save_run_metadata, timed


def get_panel_genes(sample_dir: Path):
    """Read the spatial gene panel from Xenium features.tsv.gz."""
    features_gz = sample_dir / "cell_feature_matrix" / "features.tsv.gz"
    if features_gz.exists():
        import gzip
        with gzip.open(features_gz, "rt") as f:
            genes = [line.strip().split("\t")[1] for line in f if line.strip()]
        print(f"[INFO] Panel genes from features.tsv.gz: {len(genes)}")
        return set(genes)
    # Fallback: read unique genes from transcripts
    for fname in ["transcripts.parquet", "transcripts.csv.gz"]:
        p = sample_dir / fname
        if p.exists():
            if fname.endswith(".parquet"):
                import pyarrow.parquet as pq
                genes = set(pq.read_table(p, columns=["feature_name"])["feature_name"].to_pylist())
            else:
                genes = set(pd.read_csv(p, usecols=["feature_name"])["feature_name"].unique())
            print(f"[INFO] Panel genes from transcripts: {len(genes)}")
            return genes
    return None


def prepare_bidcell_reference(
    reference_path: str,
    celltype_col: str,
    panel_genes,
    cache_dir: Path,
):
    """Generate fp_ref, fp_pos_markers, fp_neg_markers CSVs for BIDCell.

    - fp_ref: genes × cell_types mean expression matrix (only panel genes)
    - fp_pos_markers: top-10th-percentile expressed genes per cell type (one column per type)
    - fp_neg_markers: bottom-10th-percentile expressed genes per cell type

    Files are cached so subsequent runs reuse them without recomputation.
    Returns (ref_csv, pos_csv, neg_csv) paths.
    """
    import scanpy as sc
    from scipy import sparse

    ref_csv = cache_dir / "sc_ref.csv"
    pos_csv = cache_dir / "sc_markers_pos.csv"
    neg_csv = cache_dir / "sc_markers_neg.csv"

    if ref_csv.exists() and pos_csv.exists() and neg_csv.exists():
        print(f"[INFO] BIDCell reference files found in cache: {cache_dir}")
        return ref_csv, pos_csv, neg_csv

    print(f"[INFO] Generating BIDCell reference files from: {reference_path}")
    ref = sc.read_h5ad(reference_path)

    if celltype_col not in ref.obs.columns:
        raise ValueError(
            f"Column '{celltype_col}' not found in reference .obs. "
            f"Available: {list(ref.obs.columns)}"
        )

    # Subset to spatial panel genes only
    if panel_genes:
        common = [g for g in ref.var_names if g in panel_genes]
        if not common:
            raise ValueError("No overlap between reference genes and spatial panel genes.")
        ref = ref[:, common].copy()
        print(f"[INFO] Reference subset to {len(common)} panel genes")

    X = ref.X.toarray() if sparse.issparse(ref.X) else np.array(ref.X)
    cell_types = ref.obs[celltype_col].astype(str)
    sorted_types = sorted(cell_types.unique())

    # Mean expression per cell type → fp_ref
    means = {}
    for ct in sorted_types:
        mask = (cell_types == ct).values
        means[ct] = X[mask].mean(axis=0)

    ref_df = pd.DataFrame(means, index=ref.var_names)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(ref_csv)
    print(f"[INFO] Saved fp_ref: {ref_csv}  ({ref_df.shape[0]} genes × {ref_df.shape[1]} cell types)")

    # Percentile-based pos/neg markers per cell type → fp_pos_markers, fp_neg_markers
    pos_markers = {}
    neg_markers = {}
    for ct, mean_expr in means.items():
        expressed = mean_expr[mean_expr > 0]
        if len(expressed) > 0:
            pos_cut = np.percentile(expressed, 90)
            pos_markers[ct] = ref.var_names[mean_expr >= pos_cut].tolist()
        else:
            pos_markers[ct] = []
        neg_cut = np.percentile(mean_expr, 10)
        neg_markers[ct] = ref.var_names[mean_expr <= neg_cut].tolist()

    # Pad to equal length with NaN so pandas writes a rectangular CSV
    pos_df = pd.DataFrame({ct: pd.Series(genes) for ct, genes in pos_markers.items()})
    neg_df = pd.DataFrame({ct: pd.Series(genes) for ct, genes in neg_markers.items()})
    pos_df.to_csv(pos_csv, index=False)
    neg_df.to_csv(neg_csv, index=False)
    print(f"[INFO] Saved fp_pos_markers: {pos_csv}")
    print(f"[INFO] Saved fp_neg_markers: {neg_csv}")

    return ref_csv, pos_csv, neg_csv


def main():
    parser = argparse.ArgumentParser(description="Run BIDCell segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--bidcell-config", default=None,
                        help="Pre-existing BIDCell YAML config (skips generation)")
    parser.add_argument("--reference-path", default=None,
                        help="Path to reference h5ad for generating BIDCell reference files")
    parser.add_argument("--reference-celltype-col", default="cell_type",
                        help="Column in reference .obs with cell type labels")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "bidcell")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  BIDCell — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] No GPU — BIDCell training will be very slow")
    except ImportError:
        print("[ERROR] PyTorch not available")
        sys.exit(1)

    # Get or generate BIDCell config
    import yaml as _yaml
    from bidcell import BIDCellModel

    sample_dir = Path(args.sample_dir)

    # ── Generate reference files if not yet cached ──
    ref_csv = pos_csv = neg_csv = None
    if args.reference_path:
        cache_dir = Path(args.reference_path).parent / "reference_cache"
        try:
            panel_genes = get_panel_genes(sample_dir)
            ref_csv, pos_csv, neg_csv = prepare_bidcell_reference(
                args.reference_path, args.reference_celltype_col, panel_genes, cache_dir
            )
        except Exception as e:
            print(f"[WARN] Could not prepare BIDCell reference files: {e}")

    if args.bidcell_config:
        bidcell_config = Path(args.bidcell_config)
    else:
        template = params.get("config_template", platform)
        bidcell_config = output_dir / f"bidcell_config_{template}.yaml"
        BIDCellModel.get_example_config(template)
        example_name = f"{template}_example_config.yaml"
        if os.path.exists(example_name):
            import shutil
            shutil.move(example_name, str(bidcell_config))

        # Patch auto-detectable paths for the sample
        PLATFORM_FILES = {
            "xenium": {
                "fp_dapi": "morphology_mip.ome.tif",
                "fp_transcripts": "transcripts.csv.gz",
            },
        }
        with open(bidcell_config) as f:
            bc_cfg = _yaml.safe_load(f)

        bc_cfg.setdefault("files", {})["data_dir"] = str(sample_dir)
        for key, fname in PLATFORM_FILES.get(platform, {}).items():
            bc_cfg["files"][key] = str(sample_dir / fname)

        # Patch in reference files if available
        if ref_csv and ref_csv.exists():
            bc_cfg["files"]["fp_ref"] = str(ref_csv)
        if pos_csv and pos_csv.exists():
            bc_cfg["files"]["fp_pos_markers"] = str(pos_csv)
        if neg_csv and neg_csv.exists():
            bc_cfg["files"]["fp_neg_markers"] = str(neg_csv)

        with open(bidcell_config, "w") as f:
            _yaml.dump(bc_cfg, f, default_flow_style=False)

        # Check if reference files are still unset — can't run without them
        ref_keys = ["fp_ref", "fp_pos_markers", "fp_neg_markers"]
        missing = [k for k in ref_keys if not Path(bc_cfg["files"].get(k, "")).exists()]
        if missing:
            print(f"[WARN] BIDCell requires reference files not yet provided: {missing}")
            print(f"[INFO] Edit the config and rerun with:")
            print(f"       --bidcell-config {bidcell_config}")
            print(f"[INFO] Config written to: {bidcell_config}")
            sys.exit(0)

    @timed("BIDCell full pipeline")
    def _run():
        os.chdir(str(output_dir))
        model = BIDCellModel(str(bidcell_config))
        model.run_pipeline()
    _run()

    print("[INFO] BIDCell outputs .tif label masks.")
    print("[INFO] Resize to DAPI dims with cv2.INTER_NEAREST before Explorer import.")

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "bidcell", method_cfg, elapsed)
    print(f"[DONE] BIDCell — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
