"""
run_comseg.py — ComSeg transcript segmentation for a single sample.

ComSeg is transcript-based (like Baysor) but requires a prior image-based
segmentation to provide cell centroid seeds. By default this script uses
the xenium native cell_boundaries already loaded with the data. Set
prior_shapes_key in config to use a different method's boundaries.

Called by the generated SLURM script with sample-specific paths:
    python run_comseg.py --config CONFIG --sample-dir /path/to/output-XETG... \
                         --output-dir /path/to/comseg_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

# Suppress noisy-but-harmless warnings in the main process.
# For dask worker subprocesses, PYTHONWARNINGS env var is set below so workers
# inherit the same filters (filterwarnings() alone doesn't cross process boundaries).
warnings.filterwarnings("ignore", message="cudf: GPU unavailable", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning:cudf,ignore:An input array is constant",
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, configure_dask, load_platform_data,
    aggregate_and_save, save_run_metadata, timed,
)


def main():
    parser = argparse.ArgumentParser(description="Run ComSeg segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "comseg")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpus = configure_threads()
    configure_dask(cpus)
    t_start = time.time()

    print("=" * 60)
    print(f"  ComSeg — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    sdata = load_platform_data(platform, args.sample_dir)

    import sopa
    sopa.settings.parallelization_backend = "dask"

    # Prior shapes: ComSeg needs cell centroid seeds from an existing segmentation.
    # Default: xenium native boundaries (loaded with the data). Override via config.
    prior_shapes_key = params.get("prior_shapes_key", "cell_boundaries")
    if prior_shapes_key not in sdata.shapes:
        available = list(sdata.shapes.keys())
        raise RuntimeError(
            f"prior_shapes_key '{prior_shapes_key}' not found in sdata.shapes. "
            f"Available: {available}. Set prior_shapes_key in comseg config."
        )
    print(f"[INFO] Using prior segmentation: '{prior_shapes_key}' "
          f"({len(sdata.shapes[prior_shapes_key])} cells)")

    # Transcript patches — write_cells_centroids=True is required by ComSeg
    patch_width = params.get("patch_width", 200)
    sopa.make_transcript_patches(
        sdata,
        patch_width=patch_width,
        prior_shapes_key=prior_shapes_key,
        write_cells_centroids=True,
    )

    n_patches = len(sdata.shapes.get("transcripts_patches", []))
    print(f"[INFO] Transcript patches created: {n_patches} (patch_width={patch_width})")

    # Build ComSeg config dict from pipeline config params (all optional)
    # When a config dict is passed, SOPA uses it as-is and requires ALL keys
    # (skipping auto-computation from the prior). Provide complete defaults here.
    # mean_cell_diameter: ~10 µm is typical for Xenium; override in config if needed.
    # allow_disconnected_polygon=True is required for real tissue — the default False
    # causes crashes when cell polygons fragment at patch boundaries.
    comseg_config = {
        "dict_scale":                 {"x": 1, "y": 1, "z": 1},
        "mean_cell_diameter":         params.get("mean_cell_diameter", 10),
        "max_cell_radius":            params.get("max_cell_radius", 17.5),
        "norm_vector":                params.get("norm_vector", False),
        "alpha":                      params.get("alpha", 0.5),
        "allow_disconnected_polygon": params.get("allow_disconnected_polygon", True),
        "min_rna_per_cell":           params.get("min_rna_per_cell", 20),
    }
    print(f"[INFO] ComSeg config: {comseg_config}")

    @timed("ComSeg segmentation")
    def _run():
        sopa.segmentation.comseg(
            sdata,
            config=comseg_config,
            min_area=params.get("min_area", 0),
            recover=params.get("recover", False),
        )
    _run()

    n_cells = len(sdata.shapes.get("comseg_boundaries", []))
    print(f"[INFO] ComSeg complete — {n_cells} cells segmented")

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="comseg",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "comseg", method_cfg, elapsed)
    print(f"[DONE] ComSeg — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
