"""
run_fastreseg.py — FastReseg refinement for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_fastreseg.py --config CONFIG --sample-dir /path/to/output-XETG... \
                            --output-dir /path/to/fastreseg_reseg/XETG... --sample-id XETG... \
                            --source-dir /path/to/proseg_reseg/XETG...
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import configure_threads, save_run_metadata


FASTRESEG_R_SCRIPT = """
# FastReseg post-hoc refinement — auto-generated
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_fastreseg.R <source_dir> <output_dir>")

source_dir <- args[1]
output_dir <- args[2]

cat("FastReseg refinement\\n")
cat("Source:", source_dir, "\\n")
cat("Output:", output_dir, "\\n")

library(FastReseg)

h5ad_files <- list.files(source_dir, pattern = "\\\\.h5ad$", full.names = TRUE)
if (length(h5ad_files) == 0) stop("No .h5ad files found in source directory")
cat("Found:", h5ad_files[1], "\\n")

# ── FastReseg template — adapt to your data ──
# result <- fastReseg_full_pipeline(
#   counts = counts, clust = clust, refProfiles = refProfiles,
#   score_baseline = score_baseline,
#   lowerCutoff = 0.5, higherCutoff = 0.9, ...
# )

cat("[NOTE] FastReseg R script is a template. Edit for your data.\\n")
"""


def main():
    parser = argparse.ArgumentParser(description="Run FastReseg (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-dir", required=True,
                        help="Path to the primary method's results to refine")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "fastreseg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  FastReseg — {args.sample_id}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    if not source_dir.exists():
        print(f"[ERROR] Source not found: {source_dir}")
        sys.exit(1)

    r_script_path = output_dir / "run_fastreseg.R"
    with open(r_script_path, "w") as f:
        f.write(FASTRESEG_R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_script_path), str(source_dir), str(output_dir)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] R failed:\n{result.stderr}")
        sys.exit(1)

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "fastreseg", method_cfg, elapsed)
    print(f"[DONE] FastReseg — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
