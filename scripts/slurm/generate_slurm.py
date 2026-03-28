#!/usr/bin/env python3
"""
generate_slurm.py — Generate SLURM scripts for each sample × method combination.

One SLURM .sh file per sample per method. Each script passes the resolved
sample-dir, output-dir, and sample-id to the Python runner.

Usage:
    python scripts/slurm/generate_slurm.py --config config/pipeline_config.yaml --all
    python scripts/slurm/generate_slurm.py --config config/pipeline_config.yaml --method proseg
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import (
    load_config, discover_samples, get_method_config,
    get_output_base_override, get_container_path, list_enabled_methods,
    SampleInfo,
)

# Method → Python script path (relative to project root)
METHOD_SCRIPTS = {
    "proseg":     "scripts/python/run_proseg.py",
    "baysor":     "scripts/python/run_baysor.py",
    "cellpose":   "scripts/python/run_cellpose.py",
    "bidcell":    "scripts/python/run_bidcell.py",
    "fastreseg":  "scripts/python/run_fastreseg.py",
    "cellspa_qc": "scripts/python/run_qc.py",
    "classifier": "scripts/python/run_rough_annotation_classifer.py",
}


def generate_slurm_script(
    cfg: dict,
    method: str,
    sample: SampleInfo,
    config_path: str,
) -> str:
    """Generate SLURM script content for one sample × one method."""
    method_cfg = get_method_config(cfg, method)
    slurm = method_cfg["slurm"]
    container = get_container_path(cfg)
    output_base = get_output_base_override(cfg)

    output_dir = sample.output_dir(method, output_base)

    # Pipeline root — needed for cd, bind paths, and logs
    pipeline_root = str(Path(config_path).resolve().parent.parent)

    # Logs go inside the pipeline directory (always writable)
    log_dir = sample.log_dir_in_pipeline(pipeline_root)

    job_name = f"seg_{method}_{sample.sample_id}"
    python_script = METHOD_SCRIPTS.get(method)
    if not python_script:
        raise ValueError(f"No script registered for: {method}")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --time={slurm.get('time', '7-00:00:00')}",
        f"#SBATCH --nodes={slurm.get('nodes', 1)}",
        f"#SBATCH --ntasks={slurm.get('ntasks', 1)}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={slurm.get('mem', '400G')}",
    ]

    if slurm.get("gpu", False):
        lines.append("#SBATCH --gres=gpu:1g.10gb:1")
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("email"):
        lines.append(f"#SBATCH --mail-user={slurm['email']}")
        lines.append(f"#SBATCH --mail-type={slurm.get('mail_type', 'END,FAIL')}")

    lines += [
        "",
        f"# {method.upper()} — {sample.sample_id}",
        f"# Slide: {sample.slide_name}",
        f"# Input: {sample.sample_dir}",
        f"# Output: {output_dir}",
        "",
        f"mkdir -p {log_dir}",
        f"mkdir -p {output_dir}",
        "",
        f"# cd to pipeline root so Python imports resolve",
        f"cd {pipeline_root}",
        "",
        "echo '============================================'",
        f"echo '  {method.upper()} — {sample.sample_id}'",
        f"echo '  Slide: {sample.slide_name}'",
        "echo \"  Job: $SLURM_JOB_ID  Node: $(hostname)\"",
        "echo \"  Start: $(date)\"",
        "echo '============================================'",
        "",
    ]

    nv_flag = "--nv " if slurm.get("gpu", False) else ""

    # Build bind paths — collect all unique parent directories that need to be visible
    bind_paths = set()
    bind_paths.add(str(sample.sample_dir))           # raw data
    bind_paths.add(str(output_dir))                   # output
    bind_paths.add(str(log_dir))                      # logs
    bind_paths.add(str(Path(config_path).resolve().parent))  # config dir
    bind_paths.add(pipeline_root)                             # pipeline scripts

    bind_flag = " ".join(f"--bind {p}" for p in sorted(bind_paths))

    # Build the Python command with per-sample args
    python_bin = "/opt/miniforge3/envs/spatial_segmentation_env/bin/python"

    if method == "cellspa_qc":
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-id {sample.sample_id} "
            f"--slide-dir {sample.slide_dir} "
            f"--sample-dir {sample.sample_dir}"
        )
    elif method == "fastreseg":
        source_method = method_cfg["params"].get("source_method", "proseg")
        source_dir = sample.output_dir(source_method, output_base)
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-dir {sample.sample_dir} "
            f"--output-dir {output_dir} "
            f"--sample-id {sample.sample_id} "
            f"--source-dir {source_dir}"
        )
    else:
        py_args = (
            f"    {python_bin} {python_script} "
            f"--config {config_path} "
            f"--sample-dir {sample.sample_dir} "
            f"--output-dir {output_dir} "
            f"--sample-id {sample.sample_id}"
        )

    lines += [
        f"singularity exec {nv_flag}{bind_flag} \\",
        f"    {container} \\",
        py_args,
        "",
        "EXIT_CODE=$?",
        "",
    ]

    lines += [
        "echo \"Finished: $(date)  Exit code: $EXIT_CODE\"",
        "exit $EXIT_CODE",
    ]

    return "\n".join(lines)


def generate_classifier_script(
    cfg: dict,
    source_method: str,
    sample: SampleInfo,
    config_path: str,
) -> str:
    """Generate SLURM script for the classifier on one segmentation method's output.

    The classifier runs once per (seg_method, sample), annotating that method's h5ad.
    Output (annotated h5ad + CSV) is written into the same method output dir so
    the QC script can discover it alongside the original h5ad.
    """
    method_cfg = get_method_config(cfg, "classifier")
    slurm = method_cfg["slurm"]
    container = get_container_path(cfg)
    output_base = get_output_base_override(cfg)

    source_output_dir = sample.output_dir(source_method, output_base)
    h5ad_path = source_output_dir / f"{sample.sample_id}.h5ad"

    pipeline_root = str(Path(config_path).resolve().parent.parent)
    log_dir = sample.log_dir_in_pipeline(pipeline_root)

    job_name = f"seg_classify_{source_method}_{sample.sample_id}"
    python_script = METHOD_SCRIPTS["classifier"]

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
        f"#SBATCH --time={slurm.get('time', '0-06:00:00')}",
        f"#SBATCH --nodes={slurm.get('nodes', 1)}",
        f"#SBATCH --ntasks={slurm.get('ntasks', 1)}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={slurm.get('mem', '100G')}",
    ]

    if slurm.get("gpu", False):
        lines.append("#SBATCH --gres=gpu:1g.10gb:1")
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")

    bind_paths = set()
    bind_paths.add(str(source_output_dir))
    bind_paths.add(str(log_dir))
    bind_paths.add(str(Path(config_path).resolve().parent))
    bind_paths.add(pipeline_root)
    ref_path = method_cfg["params"].get("reference_path", "")
    if ref_path:
        bind_paths.add(str(Path(ref_path).parent))
    bind_flag = " ".join(f"--bind {p}" for p in sorted(bind_paths))

    nv_flag = "--nv " if slurm.get("gpu", False) else ""
    python_bin = "/opt/miniforge3/envs/spatial_segmentation_env/bin/python"

    celltype_col = method_cfg["params"].get("reference_celltype_col", "cell_type")
    gpu_flag = " --gpu" if slurm.get("gpu", False) else ""

    py_args = (
        f"    {python_bin} {python_script} "
        f"--reference {ref_path} "
        f"--celltype-col {celltype_col} "
        f"--query {h5ad_path} "
        f"--output-dir {source_output_dir} "
        f"--sample-id {sample.sample_id}"
        f"{gpu_flag}"
    )

    lines += [
        "",
        f"# CLASSIFIER ({source_method}) — {sample.sample_id}",
        f"# Slide: {sample.slide_name}",
        f"# Source: {source_output_dir}",
        "",
        f"mkdir -p {log_dir}",
        "",
        f"cd {pipeline_root}",
        "",
        "echo '============================================'",
        f"echo '  CLASSIFIER ({source_method}) — {sample.sample_id}'",
        f"echo '  Slide: {sample.slide_name}'",
        "echo \"  Job: $SLURM_JOB_ID  Node: $(hostname)\"",
        "echo \"  Start: $(date)\"",
        "echo '============================================'",
        "",
        f"singularity exec {nv_flag}{bind_flag} \\",
        f"    {container} \\",
        py_args,
        "",
        "EXIT_CODE=$?",
        "echo \"Finished: $(date)  Exit code: $EXIT_CODE\"",
        "exit $EXIT_CODE",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", help="Single method")
    parser.add_argument("--all", action="store_true", help="All enabled methods")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--outdir", default="scripts/slurm/generated")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = discover_samples(cfg)
    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.all:
        methods = list_enabled_methods(cfg)
    elif args.method:
        methods = [args.method]
    else:
        parser.error("Specify --method or --all")

    print(f"Discovered {len(samples)} sample(s), generating for {len(methods)} method(s)\n")

    generated = []
    for sample in samples:
        for method in methods:
            if method not in METHOD_SCRIPTS:
                continue

            content = generate_slurm_script(cfg, method, sample, args.config)
            fname = f"submit_{method}_{sample.sample_id}.sh"
            script_path = out_path / fname

            with open(script_path, "w") as f:
                f.write(content)
            os.chmod(script_path, 0o755)
            print(f"[OK] {script_path}")
            generated.append(script_path)

            if args.submit:
                result = subprocess.run(
                    ["sbatch", str(script_path)], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"     Submitted: {result.stdout.strip()}")
                else:
                    print(f"     [ERROR]: {result.stderr.strip()}")

    print(f"\n[DONE] {len(generated)} script(s) in {out_path}/")


if __name__ == "__main__":
    main()
