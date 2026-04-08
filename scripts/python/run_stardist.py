"""
run_stardist.py — StarDist segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_stardist.py --config CONFIG --sample-dir /path/to/output-XETG... \
                           --output-dir /path/to/stardist_reseg/XETG... --sample-id XETG...

How model loading works (csbdeep / keras):
    csbdeep calls keras.get_file with NO cache_dir argument, so keras always
    uses ~/.keras as the cache root. With cache_subdir="models/StarDist2D/{key}"
    the zip lands at:
        ~/.keras/models/StarDist2D/{model_type}/{model_type}.zip
    and the extracted model at:
        ~/.keras/models/StarDist2D/{model_type}/{model_type}/
    (keras >= 3.6.0 may name the extraction {model_type}_extracted/ instead,
    and csbdeep then symlinks {model_type} → {model_type}_extracted).

    We pre-stage these paths from the wizard-downloaded copy before any dask
    workers spawn, so workers (fresh Python processes that don't inherit
    in-memory state) find the model locally without hitting the network.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import (
    configure_threads, configure_dask, load_platform_data,
    aggregate_and_save, save_run_metadata, timed,
)


def _stage_model_for_keras(model_type: str, src_model_dir: Path):
    """Populate ~/.keras/models/StarDist2D/{model_type}/ with the zip and
    extracted model so csbdeep/keras finds them without any network access.

    csbdeep calls keras.get_file(cache_subdir="models/StarDist2D/{key}"),
    so keras expects:
        ~/.keras/models/StarDist2D/{model_type}/{model_type}.zip   ← for hash check
        ~/.keras/models/StarDist2D/{model_type}/{model_type}/      ← extracted weights
        ~/.keras/models/StarDist2D/{model_type}/{model_type}_extracted/  ← keras >= 3.6.0

    src_model_dir: wizard-extracted dir, e.g.
        seg_models/models/StarDist2D/2D_versatile_fluo/
    src zip lives one level up:
        seg_models/models/StarDist2D/2D_versatile_fluo.zip
    """
    src_zip = src_model_dir.parent / f"{model_type}.zip"

    keras_subdir = Path.home() / ".keras" / "models" / "StarDist2D" / model_type
    keras_subdir.mkdir(parents=True, exist_ok=True)

    # ── zip (keras validates hash before using local copy) ────────────────────
    # Always overwrite — a stale/wrong zip from a previous attempt would fail
    # keras's hash check and trigger a re-download on compute nodes.
    dest_zip = keras_subdir / f"{model_type}.zip"
    if src_zip.exists():
        shutil.copy2(str(src_zip), str(dest_zip))
        print(f"[INFO] Staged zip → {dest_zip}")
    else:
        print(f"[WARN] Source zip not found at {src_zip} — workers may try to download")

    # ── extracted model dir (keras < 3.6.0 and as fallback) ──────────────────
    # Always recreate symlinks so a stale target from a previous attempt is fixed.
    dest_model = keras_subdir / model_type
    if dest_model.is_symlink():
        dest_model.unlink()
    if not dest_model.exists():
        try:
            dest_model.symlink_to(src_model_dir.resolve())
            print(f"[INFO] Staged model dir (symlink) → {dest_model}")
        except Exception as e:
            print(f"[WARN] Could not symlink model dir: {e}  (non-fatal if keras >= 3.6.0)")

    # ── _extracted variant (keras >= 3.6.0) ───────────────────────────────────
    dest_extracted = keras_subdir / f"{model_type}_extracted"
    if dest_extracted.is_symlink():
        dest_extracted.unlink()
    if not dest_extracted.exists():
        try:
            dest_extracted.symlink_to(src_model_dir.resolve())
            print(f"[INFO] Staged model dir (keras 3.6.0+ symlink) → {dest_extracted}")
        except Exception as e:
            print(f"[WARN] Could not create _extracted symlink: {e}  (non-fatal if keras < 3.6.0)")

    print(f"[INFO] Model staged at {keras_subdir}")


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

    model_type = params.get("model_type", "2D_versatile_fluo")
    channels   = params.get("channels", ["DAPI"])

    # ── Stage model into ~/.keras before any workers spawn ────────────────────
    # csbdeep passes no cache_dir to keras, so workers always look in ~/.keras.
    # Home dir is always mounted by Singularity and shared by every spawned
    # worker process — the only path that reliably survives dask-nanny spawn.
    seg_models_path = cfg.get("data", {}).get("seg_models_path", "")
    if seg_models_path:
        candidate = Path(seg_models_path) / "models" / "StarDist2D" / model_type
        print(f"[INFO] seg_models_path: {seg_models_path}")
        print(f"[INFO] Source model dir: {candidate}  exists={candidate.is_dir()}")
        if candidate.is_dir():
            _stage_model_for_keras(model_type, candidate)
        else:
            print(f"[WARN] Source model dir not found — workers will attempt download (will fail on compute nodes)")
    else:
        print(f"[WARN] seg_models_path not set in config — workers will attempt download (will fail on compute nodes)")

    cpus = configure_threads()
    configure_dask(cpus)
    t_start = time.time()

    print("=" * 60)
    print(f"  StarDist — {args.sample_id}")
    print(f"  Model:  {model_type}")
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

    @timed("StarDist segmentation")
    def _run():
        sopa.segmentation.stardist(
            sdata,
            model_type=model_type,
            channels=channels,
            min_area=params.get("min_area", 0),
            prob_thresh=params.get("prob_thresh", 0.2),
            nms_thresh=params.get("nms_thresh", 0.6),
        )
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
