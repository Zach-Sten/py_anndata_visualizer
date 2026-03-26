"""
run_baysor.py — Baysor segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_baysor.py --config CONFIG --sample-dir /path/to/output-XETG... \
                         --output-dir /path/to/baysor_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import asyncio
import argparse
import warnings
from pathlib import Path

# Suppress noisy-but-harmless warnings from dependencies (main process)
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings("ignore", category=UserWarning, module="spatialdata_io")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, configure_dask, load_platform_data,
    prepare_patches, aggregate_and_save, save_run_metadata, timed,
)


def main():
    parser = argparse.ArgumentParser(description="Run Baysor segmentation (single sample)")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "baysor")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpus = configure_threads()
    if params.get("parallelization_backend") == "dask":
        configure_dask(cpus)

    t_start = time.time()

    print("=" * 60)
    print(f"  Baysor — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    sdata = load_platform_data(platform, args.sample_dir)

    # Rechunk large labels
    if hasattr(sdata, "labels") and "cell_labels" in sdata.labels:
        try:
            lbl = sdata.labels["cell_labels"]
            sdata.labels["cell_labels"] = lbl.copy(data=lbl.data.rechunk((2048, 2048)))
        except Exception:
            pass

    import sopa
    if params.get("parallelization_backend") == "dask":
        sopa.settings.parallelization_backend = "dask"

    prepare_patches(
        sdata,
        image_patch_width=params.get("image_patch_width", 1200),
        image_patch_overlap=params.get("patch_overlap", 10),
        transcript_patch_width=params.get("patch_width", 500),
        prior_shapes_key=params.get("prior_shapes_key", "cell_boundaries"),
    )

    n_tiles = len(sdata.shapes.get("transcripts_patches", []))
    print(f"[INFO] Running Baysor on {n_tiles} transcript patches...")

    def _has_baysor_results():
        """Check if sopa can find valid segmentation results in sdata."""
        try:
            from sopa.utils import get_boundaries
            key, _ = get_boundaries(sdata, return_key=True)
            print(f"[INFO] Valid segmentation found in sdata: '{key}'")
            return True
        except Exception:
            all_keys = list(sdata.shapes.keys())
            print(f"[INFO] No valid sopa segmentation in sdata. Shape keys: {all_keys}")
            return False

    from dask.diagnostics import ProgressBar

    @timed("Baysor segmentation")
    def _run():
        try:
            with ProgressBar():
                sopa.segmentation.baysor(sdata, min_area=params.get("min_area", 10))
        except (asyncio.TimeoutError, TimeoutError):
            # Dask worker cleanup times out because Julia/baysor subprocesses are
            # slow to exit. Check if results were committed before the timeout.
            if _has_baysor_results():
                print("[INFO] Results already in sdata — skipping retry")
                return
            print("[WARN] Dask worker cleanup timed out, results not in sdata — collecting...")
            try:
                with ProgressBar():
                    sopa.segmentation.baysor(sdata, min_area=params.get("min_area", 10))
            except (asyncio.TimeoutError, TimeoutError):
                print("[WARN] Dask cleanup timed out on collection pass — proceeding to aggregation...")
                if not _has_baysor_results():
                    raise RuntimeError(
                        "Baysor results not in sdata after two attempts. "
                        "Patch files are on disk but collection failed. "
                        "Check sopa version or try re-running without dask backend."
                    )
    _run()

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="baysor",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "baysor", method_cfg, elapsed)
    print(f"[DONE] Baysor — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
