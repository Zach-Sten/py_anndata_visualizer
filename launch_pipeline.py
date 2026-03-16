#!/usr/bin/env python3
"""
launch_pipeline.py — Master launcher for the spatial segmentation pipeline.

Discovers samples (experiment / slide / single), generates per-sample SLURM
scripts, and optionally submits them with dependency chains:
  primary segmentation → post-hoc (fastreseg) → QC

Usage:
    # Dry run (generate scripts, don't submit):
    python launch_pipeline.py --config config/pipeline_config.yaml

    # Submit everything:
    python launch_pipeline.py --config config/pipeline_config.yaml --submit

    # Only specific methods:
    python launch_pipeline.py --config config/pipeline_config.yaml --submit --methods proseg baysor

    # List discovered samples without generating anything:
    python launch_pipeline.py --config config/pipeline_config.yaml --list
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from utils.config_loader import (
    load_config, discover_samples, list_enabled_methods,
    get_method_config, get_output_base_override, SampleInfo,
)
from slurm.generate_slurm import generate_slurm_script, METHOD_SCRIPTS


def submit_job(script_path: str, dependency_ids: list = None) -> str:
    """Submit via sbatch, optionally with afterok dependencies. Returns job ID."""
    cmd = ["sbatch"]
    if dependency_ids:
        dep_str = ":".join(str(d) for d in dependency_ids)
        cmd.extend(["--dependency", f"afterok:{dep_str}"])
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] sbatch: {result.stderr.strip()}")
        return None
    return result.stdout.strip().split()[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Launch spatial segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--submit", action="store_true", help="Submit to SLURM")
    parser.add_argument("--methods", nargs="+", help="Specific methods (default: all enabled)")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC jobs")
    parser.add_argument("--list", action="store_true", help="Just list discovered samples and exit")
    parser.add_argument("--outdir", default="scripts/slurm/generated")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = discover_samples(cfg)

    # ── List mode ──
    if args.list:
        print(f"\nDiscovered {len(samples)} sample(s):\n")
        by_slide = defaultdict(list)
        for s in samples:
            by_slide[s.slide_name].append(s)

        for slide, slide_samples in by_slide.items():
            print(f"  {slide}/")
            for s in slide_samples:
                print(f"    {s.sample_id}  ← {s.sample_dir.name}")
        print()
        return

    # ── Determine methods ──
    if args.methods:
        methods = args.methods
    else:
        methods = list_enabled_methods(cfg)

    primary = [m for m in methods if m not in ("cellspa_qc", "fastreseg")]
    post = [m for m in methods if m == "fastreseg"]
    run_qc = not args.skip_qc and "cellspa_qc" in cfg.get("methods", {}) and \
             cfg["methods"]["cellspa_qc"].get("enabled", True)

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Summary ──
    print("=" * 65)
    print("  Spatial Segmentation Pipeline")
    print("=" * 65)
    print(f"  Config:    {args.config}")
    print(f"  Samples:   {len(samples)}")
    print(f"  Methods:   {', '.join(primary)}")
    if post:
        print(f"  Post-hoc:  {', '.join(post)}")
    print(f"  QC:        {'yes' if run_qc else 'no'}")
    print(f"  Submit:    {'YES' if args.submit else 'DRY RUN'}")

    total_jobs = len(samples) * (len(primary) + len(post) + (1 if run_qc else 0))
    print(f"  Jobs:      {total_jobs}")
    print("=" * 65)

    # Group samples by slide for nice output
    by_slide = defaultdict(list)
    for s in samples:
        by_slide[s.slide_name].append(s)

    all_scripts = []

    for slide_name, slide_samples in by_slide.items():
        print(f"\n── {slide_name} ({len(slide_samples)} sample(s)) ──")

        for sample in slide_samples:
            # Track job IDs per sample for dependency chains
            seg_job_ids = []

            # ── Primary segmentation ──
            for method in primary:
                if method not in METHOD_SCRIPTS:
                    continue

                content = generate_slurm_script(cfg, method, sample, args.config)
                fname = f"submit_{method}_{sample.sample_id}.sh"
                script_path = out_path / fname

                with open(script_path, "w") as f:
                    f.write(content)
                os.chmod(script_path, 0o755)
                all_scripts.append(script_path)

                label = f"  {sample.sample_id} / {method}"
                if args.submit:
                    jid = submit_job(script_path)
                    if jid:
                        seg_job_ids.append(jid)
                        print(f"  [OK] {label} → job {jid}")
                    else:
                        print(f"  [FAIL] {label}")
                else:
                    print(f"  [OK] {fname}")

            # ── Post-hoc (depends on primary) ──
            post_job_ids = []
            for method in post:
                if method not in METHOD_SCRIPTS:
                    continue
                content = generate_slurm_script(cfg, method, sample, args.config)
                fname = f"submit_{method}_{sample.sample_id}.sh"
                script_path = out_path / fname
                with open(script_path, "w") as f:
                    f.write(content)
                os.chmod(script_path, 0o755)
                all_scripts.append(script_path)

                if args.submit:
                    jid = submit_job(script_path, dependency_ids=seg_job_ids)
                    if jid:
                        post_job_ids.append(jid)
                        print(f"  [OK] {sample.sample_id} / {method} → job {jid} (after primary)")
                else:
                    print(f"  [OK] {fname}")

            # ── QC (depends on all seg + post) ──
            if run_qc:
                content = generate_slurm_script(cfg, "cellspa_qc", sample, args.config)
                fname = f"submit_qc_{sample.sample_id}.sh"
                script_path = out_path / fname
                with open(script_path, "w") as f:
                    f.write(content)
                os.chmod(script_path, 0o755)
                all_scripts.append(script_path)

                if args.submit:
                    wait = seg_job_ids + post_job_ids
                    jid = submit_job(script_path, dependency_ids=wait if wait else None)
                    if jid:
                        print(f"  [OK] {sample.sample_id} / qc → job {jid} (after segmentation)")
                else:
                    print(f"  [OK] {fname}")

    # ── Final summary ──
    print("\n" + "=" * 65)
    print(f"  Generated {len(all_scripts)} SLURM script(s) → {out_path}/")
    if args.submit:
        print(f"  Monitor:   squeue -u $USER")
    else:
        print(f"\n  To submit: python launch_pipeline.py --config {args.config} --submit")
    print("=" * 65)


if __name__ == "__main__":
    main()
