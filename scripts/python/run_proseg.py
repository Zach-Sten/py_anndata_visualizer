"""
run_proseg.py — ProSeg segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_proseg.py --config CONFIG --sample-dir /path/to/output-XETG... \
                         --output-dir /path/to/proseg_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, load_platform_data, prepare_patches,
    aggregate_and_save, replace_proseg_counts, save_run_metadata, timed,
)


def main():
    parser = argparse.ArgumentParser(description="Run ProSeg segmentation (single sample)")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--sample-dir", required=True, help="Path to the raw output-XETG... folder")
    parser.add_argument("--output-dir", required=True, help="Path to write results")
    parser.add_argument("--sample-id", required=True, help="Sample identifier")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "proseg")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpus = configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  ProSeg — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # ── Load ──
    sdata = load_platform_data(platform, args.sample_dir)

    # ── Patches ──
    prepare_patches(
        sdata,
        image_patch_width=params.get("patch_width", 1200),
        image_patch_overlap=params.get("patch_overlap", 10),
        transcript_patch_width=None,
        prior_shapes_key="cell_boundaries",
    )

    # ── Run ProSeg ──
    import sopa

    exp_path = str(output_dir / "expected_counts.mtx.gz")
    cmd_parts = []
    if params.get("xenium_mode", True):
        cmd_parts.append("--xenium")
    if params.get("no_diffusion", True):
        cmd_parts.append("--no-diffusion")
    cmd_parts.append(f"--nthreads {cpus}")
    if params.get("export_expected_counts", True):
        cmd_parts.append(f'--output-expected-counts "{exp_path}"')

    cmd_suffix = " ".join(cmd_parts)
    print(f"[INFO] ProSeg flags: {cmd_suffix}")

    @timed("ProSeg segmentation")
    def _run():
        sopa.segmentation.proseg(sdata, command_line_suffix=cmd_suffix)
    _run()

    # ── Aggregate + export ──
    adata = aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
    )

    # ── Replace with expected counts ──
    if params.get("export_expected_counts", True) and os.path.exists(exp_path):
        adata = replace_proseg_counts(adata, exp_path)
        adata.write_h5ad(output_dir / f"{args.sample_id}.h5ad")
        print(f"[INFO] Re-saved h5ad with integer counts")

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "proseg", method_cfg, elapsed)
    print(f"[DONE] ProSeg — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
