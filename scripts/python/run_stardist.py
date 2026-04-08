"""
run_stardist.py — StarDist segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_stardist.py --config CONFIG --sample-dir /path/to/output-XETG... \
                           --output-dir /path/to/stardist_reseg/XETG... --sample-id XETG...
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
    parser = argparse.ArgumentParser(description="Run StarDist segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "stardist")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Point csbdeep to the pre-cached models so workers never try to download.
    seg_models_path = cfg.get("data", {}).get("seg_models_path", "")
    if seg_models_path:
        os.environ["CSBDEEP_CACHE_DIR"] = seg_models_path
        print(f"[INFO] CSBDEEP_CACHE_DIR: {seg_models_path}")

    cpus = configure_threads()
    configure_dask(cpus)
    t_start = time.time()

    print("=" * 60)
    print(f"  StarDist — {args.sample_id}")
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

    # Select model: default to 2D_versatile_fluo for fluorescence/Xenium data;
    # set model_type: "2D_versatile_he" in config for H&E images.
    model_type = params.get("model_type", "2D_versatile_fluo")
    channels = params.get("channels", ["DAPI"])

    # Resolve local_model from seg_models_path when available — this bypasses
    # StarDist2D.from_pretrained() and keras's download logic entirely.
    # The extracted model lives at {seg_models_path}/models/StarDist2D/{model_type}/
    local_model = params.get("local_model", None)
    if not local_model and seg_models_path:
        candidate = Path(seg_models_path) / "models" / "StarDist2D" / model_type
        if candidate.is_dir():
            local_model = str(candidate)
            print(f"[INFO] Using local StarDist model: {local_model}")
        else:
            print(f"[WARN] Expected local model not found at {candidate}; falling back to from_pretrained")

    @timed("StarDist segmentation")
    def _run():
        kwargs = dict(
            channels=channels,
            min_area=params.get("min_area", 0),
            prob_thresh=params.get("prob_thresh", 0.2),
            nms_thresh=params.get("nms_thresh", 0.6),
        )
        if local_model:
            kwargs["local_model"] = local_model
        else:
            kwargs["model_type"] = model_type
        sopa.segmentation.stardist(sdata, **kwargs)
    _run()

    sopa.make_transcript_patches(
        sdata,
        patch_width=params.get("transcript_patch_width", 500),
        prior_shapes_key="stardist_boundaries",
    )

    aggregate_and_save(
        sdata, output_dir, args.sample_id,
        explorer_mode=params.get("explorer_mode", "+cbm"),
        method="stardist",
    )

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "stardist", method_cfg, elapsed)
    print(f"[DONE] StarDist — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
