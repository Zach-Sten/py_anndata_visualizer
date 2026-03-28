"""
run_cellpose.py — Cellpose segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_cellpose.py --config CONFIG --sample-dir /path/to/output-XETG... \
                           --output-dir /path/to/cellpose_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, configure_dask, load_platform_data,
    aggregate_and_save, save_run_metadata, timed,
)


def main():
    parser = argparse.ArgumentParser(description="Run Cellpose segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "cellpose")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure dask workers find the pre-cached CPSAM model (cellpose 4.x)
    # rather than attempting an internet download on HPC nodes.
    if "CELLPOSE_LOCAL_MODELS_PATH" not in os.environ:
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/opt/cellpose_models"

    cpus = configure_threads()
    configure_dask(cpus)
    t_start = time.time()

    print("=" * 60)
    print(f"  Cellpose — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    sdata = load_platform_data(platform, args.sample_dir)

    import sopa
    sopa.segmentation.tissue(sdata)
    sopa.make_image_patches(
        sdata,
        patch_width=params.get("patch_width", 1200),
        patch_overlap=params.get("patch_overlap", 50),
    )
    sopa.settings.parallelization_backend = "dask"

    # GPU check
    use_gpu = params.get("gpu", True)
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] GPU not available, falling back to CPU")
                use_gpu = False
            else:
                print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            use_gpu = False

    @timed("Cellpose segmentation")
    def _run():
        sopa.segmentation.cellpose(
            sdata,
            channels=params.get("channels", ["DAPI"]),
            diameter=params.get("diameter", 35),
            gpu=use_gpu,
        )
    _run()

    # Transcript patches for aggregation
    sopa.make_transcript_patches(sdata, patch_width=500, prior_shapes_key="cellpose_boundaries")

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="cellpose",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "cellpose", method_cfg, elapsed)
    print(f"[DONE] Cellpose — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
