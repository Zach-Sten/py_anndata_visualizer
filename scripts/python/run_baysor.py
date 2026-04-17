"""
run_baysor.py — Baysor segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_baysor.py --config CONFIG --sample-dir /path/to/output-XETG... \
                         --output-dir /path/to/baysor_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import stat
import shutil
import tempfile
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


def _install_baysor_wrapper():
    """
    Wrap the baysor binary so its stdout/stderr surface in the job's .err log.

    sopa captures Baysor subprocess output internally but discards it when raising
    CalledProcessError — so failures are silent.  We intercept by prepending a
    wrapper script to PATH before dask workers fork from this process.  Workers
    inherit the modified PATH, find our 'baysor' first, and the wrapper pipes all
    output to stderr (→ job .err) before forwarding the exit code.
    """
    real_baysor = shutil.which("baysor")
    if not real_baysor:
        print("[WARN] baysor not found in PATH — skipping output wrapper", file=sys.stderr)
        return

    wrapper_dir = tempfile.mkdtemp(prefix="baysor_wrapper_")
    wrapper = os.path.join(wrapper_dir, "baysor")
    with open(wrapper, "w") as f:
        f.write(f"""#!/bin/bash
# Auto-generated wrapper — pipes baysor output to job stderr for visibility
LOG=$(mktemp /tmp/baysor_XXXXXX.log)
"{real_baysor}" "$@" >"$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "[BAYSOR ERROR] exit $RC — patch log follows:" >&2
    cat "$LOG" >&2
fi
rm -f "$LOG"
exit $RC
""")
    os.chmod(wrapper, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    os.environ["PATH"] = wrapper_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[INFO] Baysor output wrapper installed (real binary: {real_baysor})")


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

    _install_baysor_wrapper()

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
        prior_shapes_key=params.get("prior_shapes_key", None),
    )

    n_tiles = len(sdata.shapes.get("transcripts_patches", []))
    print(f"[INFO] Running Baysor on {n_tiles} transcript patches...")

    @timed("Baysor segmentation")
    def _run():
        try:
            baysor_kwargs = {"min_area": params.get("min_area", 10)}
            # When no prior is set, sopa can't infer cell scale — must provide explicitly
            if not params.get("prior_shapes_key"):
                baysor_kwargs["scale"] = params.get("scale", 15)
            sopa.segmentation.baysor(sdata, **baysor_kwargs)
        except Exception as e:
            # CalledProcessError  — one patch's Baysor binary returned non-zero; dask
            #                       cancels remaining futures but completed patches already
            #                       wrote segmentation_counts.loom to disk.
            # TimeoutError        — dask worker cleanup hung after all patches finished.
            # Either way: resolve from whatever loom files landed on disk.
            print(f"[WARN] Baysor dask run failed ({type(e).__name__}: {e})")
            print("[WARN] Attempting to resolve completed patches from disk...")

            try:
                from sopa.segmentation.transcripts import resolve
            except ImportError:
                from sopa.segmentation.methods._baysor import resolve

            from sopa.utils import get_transcripts_patches_dirs
            all_dirs = get_transcripts_patches_dirs(sdata)

            # Only pass directories that fully finished (loom + polygon file both present)
            def _patch_complete(p):
                p = Path(p)
                if not (p / "segmentation_counts.loom").exists():
                    return False
                # sopa looks for segmentation_polygons.json (new) or segmentation.json (old)
                return (p / "segmentation_polygons.json").exists() or \
                       (p / "segmentation.json").exists()

            completed = [p for p in all_dirs if _patch_complete(p)]
            print(f"[INFO] Completed patches on disk: {len(completed)} / {len(all_dirs)}")

            if not completed:
                raise RuntimeError(
                    "No completed Baysor patches found on disk — cannot recover. "
                    "Check baysor logs for the root cause."
                ) from e

            min_area = params.get("min_area", 10)
            resolve(sdata, completed, min_area=min_area, key_added="baysor_boundaries")
            print(f"[INFO] Resolved from {len(completed)} patch(es) — baysor_boundaries added to sdata")
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
