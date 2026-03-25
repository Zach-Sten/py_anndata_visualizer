#!/usr/bin/env python3
"""
segmentation_pipeline_master.py — Interactive terminal launcher.

Walk through pipeline configuration step by step, preview the config,
discover samples, review the job plan, and submit — all from one command.

Usage:
    python segmentation_pipeline_master.py

    # Or skip the wizard and use an existing config:
    python segmentation_pipeline_master.py --config config/my_config.yaml
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from utils.config_loader import (
    load_config, discover_samples, list_enabled_methods,
    get_method_config, get_output_base_override, SampleInfo,
)
from slurm.generate_slurm import generate_slurm_script, METHOD_SCRIPTS


# ============================================================================
# Terminal formatting
# ============================================================================

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
ARROW = f"{CYAN}→{RESET}"


def width():
    return min(shutil.get_terminal_size().columns, 80)


def banner():
    w = width()
    PURPLE = "\033[35m"
    print()
    print(f"{'━' * w}")
    print(f"{BOLD}{PURPLE}")
    print(f"    ·  ✦      ·    ✧        ✦   ·      ✧    ·   ✦     ·")
    print(f"  ✧       ·       ✦    ·  ✧       ·  ✦      ✧       ·")
    print(f"")
    print(f"  ███████╗██████╗  █████╗ ████████╗██╗ █████╗ ██╗")
    print(f"  ██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║")
    print(f"  ███████╗██████╔╝███████║   ██║   ██║███████║██║")
    print(f"  ╚════██║██╔═══╝ ██╔══██║   ██║   ██║██╔══██║██║")
    print(f"  ███████║██║     ██║  ██║   ██║   ██║██║  ██║███████╗")
    print(f"  ╚══════╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝")
    print(f"  ███████╗███████╗ ██████╗    ██╗    ██╗██╗███████╗")
    print(f"  ██╔════╝██╔════╝██╔════╝    ██║    ██║██║╚══███╔╝")
    print(f"  ███████╗█████╗  ██║  ███╗   ██║ █╗ ██║██║  ███╔╝")
    print(f"  ╚════██║██╔══╝  ██║   ██║   ██║███╗██║██║ ███╔╝")
    print(f"  ███████║███████╗╚██████╔╝   ╚███╔███╔╝██║███████╗")
    print(f"  ╚══════╝╚══════╝ ╚═════╝     ╚══╝╚══╝ ╚═╝╚══════╝")
    print(f"")
    print(f"  ·    ✧      ✦  ·       ✧   ·    ✦      ·  ✧     ✦  ·")
    print(f"    ✦     ·  ✧      ·  ✦    ✧     ·   ✦    ·    ✧")
    print(f"{RESET}")
    print(f"  {PURPLE}✦ SOPA · ProSeg · Baysor · Cellpose · BIDCell · FastReseg · CellSPA ✦{RESET}")
    print(f"{'━' * w}")
    print()


def section(title):
    w = width()
    print(f"\n{BOLD}── {title} {'─' * max(0, w - len(title) - 4)}{RESET}\n")


def prompt(label, default=None, required=True):
    """Prompt for text input with optional default."""
    if default:
        display = f"  {label} {DIM}[{default}]{RESET}: "
    else:
        display = f"  {label}: "

    while True:
        val = input(display).strip()
        if not val and default is not None:
            return default
        if val:
            return val
        if not required:
            return ""
        print(f"  {RED}Required field.{RESET}")


def prompt_choice(label, options, default=None):
    """Prompt to choose from numbered options."""
    print(f"  {label}")
    for i, opt in enumerate(options, 1):
        marker = f"{GREEN}*{RESET}" if opt == default else " "
        print(f"   {marker} {i}) {opt}")

    while True:
        hint = f" [{options.index(default) + 1}]" if default else ""
        val = input(f"  Choice{hint}: ").strip()
        if not val and default:
            return default
        try:
            idx = int(val)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            if val in options:
                return val
        print(f"  {RED}Enter 1-{len(options)}{RESET}")


def prompt_yn(label, default=True):
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    val = input(f"  {label} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def prompt_multi(label, options, defaults=None):
    """Select multiple from a list. Returns list of selected."""
    if defaults is None:
        defaults = options[:]

    print(f"  {label}")
    for i, opt in enumerate(options, 1):
        marker = CHECK if opt in defaults else " "
        print(f"   {marker} {i}) {opt}")

    print(f"  {DIM}Enter numbers separated by spaces, or 'all' / 'none'{RESET}")
    val = input(f"  Select [{','.join(str(options.index(d)+1) for d in defaults)}]: ").strip()

    if not val:
        return defaults
    if val.lower() == "all":
        return options[:]
    if val.lower() == "none":
        return []

    selected = []
    for tok in val.replace(",", " ").split():
        try:
            idx = int(tok)
            if 1 <= idx <= len(options):
                selected.append(options[idx - 1])
        except ValueError:
            if tok in options:
                selected.append(tok)
    return selected if selected else defaults


def path_prompt(label, default=None, must_exist=False):
    """Prompt for a file/directory path with tab completion."""
    try:
        import readline
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
        # Enable filesystem tab completion
        import glob
        def _completer(text, state):
            # Expand ~ and glob for matches
            expanded = os.path.expanduser(text)
            matches = glob.glob(expanded + "*")
            # Add trailing / for directories
            matches = [m + "/" if os.path.isdir(m) else m for m in matches]
            return matches[state] if state < len(matches) else None
        readline.set_completer(_completer)
    except (ImportError, Exception):
        pass  # readline not available on all platforms

    while True:
        val = prompt(label, default=default, required=True)
        val = os.path.expanduser(val)
        # Resolve to absolute path — critical for SLURM jobs that run on compute nodes
        val = os.path.realpath(val)
        if must_exist and not os.path.exists(val):
            print(f"  {RED}Path not found: {val}{RESET}")
            if not prompt_yn("Continue anyway?", default=False):
                continue
        # Reset completer
        try:
            import readline
            readline.set_completer(None)
        except ImportError:
            pass
        return val


# ============================================================================
# Config wizard
# ============================================================================

def wizard():
    """Interactive configuration wizard. Returns a config dict."""
    banner()

    cfg = {"data": {}, "paths": {}, "slurm": {"default": {}}, "methods": {}}

    # ── Data ──
    section("Data")
    cfg["data"]["platform"] = prompt_choice(
        "Platform:", ["xenium", "cosmx", "merscope", "stereoseq"], default="xenium"
    )

    data_mode = prompt_choice(
        "Data mode:", ["experiment (all slides)", "single sample"], default="experiment (all slides)"
    )

    if data_mode.startswith("experiment"):
        cfg["data"]["experiment_dir"] = path_prompt(
            "Experiment directory path", must_exist=True
        )
    else:
        cfg["data"]["sample_dir"] = path_prompt(
            "Sample directory path (output-XETG... folder)", must_exist=True
        )

    cfg["data"]["sample_glob"] = "output-*"
    cfg["data"]["include"] = []
    cfg["data"]["exclude"] = []

    # Try to discover samples immediately for feedback
    print()
    try:
        test_cfg = {
            "data": cfg["data"],
            "paths": {"container_sif": "/placeholder"},
            "methods": {},
        }
        samples = discover_samples(test_cfg)
        by_slide = defaultdict(list)
        for s in samples:
            by_slide[s.slide_name].append(s)

        print(f"  {CHECK} Discovered {BOLD}{len(samples)}{RESET} sample(s):")
        for slide, ss in by_slide.items():
            print(f"     {slide}/")
            for s in ss:
                print(f"       {ARROW} {s.sample_id}")
    except Exception as e:
        print(f"  {YELLOW}⚠ Could not discover samples: {e}{RESET}")
        print(f"  {DIM}(This is OK if the path isn't accessible from this machine){RESET}")
        samples = []

    # ── Exclude/include ──
    if samples and len(samples) > 1:
        print()
        if prompt_yn("Filter samples?", default=False):
            excl = prompt("Exclude (comma-separated substrings)", default="", required=False)
            if excl:
                cfg["data"]["exclude"] = [s.strip() for s in excl.split(",")]

    # ── Container ──
    section("Container")
    cfg["paths"]["container_sif"] = path_prompt("Path to .sif container", must_exist=True)

    # ── Output ──
    section("Output")
    print(f"  {DIM}Leave blank to place results in experiment directory{RESET}")
    override = prompt("Output path", default="", required=False)
    cfg["paths"]["output_base_override"] = override

    # ── Methods ──
    section("Segmentation Methods")
    all_methods = ["proseg", "baysor", "cellpose", "bidcell", "fastreseg"]
    selected = prompt_multi(
        f"Which methods to run? {DIM}(✓ = SOPA-supported){RESET}", all_methods, defaults=["proseg", "baysor", "cellpose"]
    )

    METHOD_DEFAULTS = {
        "proseg": {
            "slurm": {"mem": "400G", "cpus_per_task": 8},
            "params": {"xenium_mode": True, "no_diffusion": True, "patch_width": 1200,
                       "patch_overlap": 10, "export_expected_counts": True, "explorer_mode": "+cbm"},
        },
        "baysor": {
            "slurm": {"mem": "600G", "cpus_per_task": 8},
            "params": {"min_area": 10, "patch_width": 500, "patch_overlap": 10, "image_patch_width": 1200,
                       "prior_shapes_key": "cell_boundaries", "parallelization_backend": "dask", "explorer_mode": "+cbm"},
        },
        "cellpose": {
            "slurm": {"mem": "200G", "cpus_per_task": 8, "gpu": True, "time": "2-00:00:00"},
            "params": {"channels": ["DAPI"], "diameter": 35, "gpu": True,
                       "patch_width": 1200, "patch_overlap": 50, "explorer_mode": "+cbm"},
        },
        "bidcell": {
            "slurm": {"mem": "300G", "cpus_per_task": 8, "gpu": True, "time": "3-00:00:00"},
            "params": {"config_template": "xenium", "explorer_mode": "+cbm"},
        },
    }

    FASTRESEG_DEFAULT = {
        "slurm": {"mem": "200G", "cpus_per_task": 4, "time": "1-00:00:00"},
        "params": {"source_method": "proseg"},
    }

    # Ask GPU/CPU for any selected GPU-capable methods
    GPU_METHODS = {"cellpose", "bidcell"}
    print()
    for method in [m for m in selected if m in GPU_METHODS]:
        use_gpu = prompt_yn(f"Use GPU for {method}?", default=True)
        if not use_gpu:
            METHOD_DEFAULTS[method]["slurm"]["gpu"] = False
            if method == "cellpose":
                METHOD_DEFAULTS[method]["params"]["gpu"] = False

    for method in all_methods:
        enabled = method in selected
        cfg["methods"][method] = {"enabled": enabled}
        if enabled and method in METHOD_DEFAULTS:
            cfg["methods"][method].update(METHOD_DEFAULTS[method])
        elif method == "fastreseg" and enabled:
            cfg["methods"][method].update(FASTRESEG_DEFAULT)

    print()
    run_qc = prompt_yn("Run CellSpa QC (runs after segmentation)?", default=True)
    cfg["methods"]["cellspa_qc"] = {
        "enabled": run_qc,
        "slurm": {"mem": "100G", "cpus_per_task": 4, "time": "0-12:00:00"},
        "params": {"methods_to_qc": selected},
    }

    # ── SLURM defaults (edit the saved YAML for custom partition/account) ──
    cfg["slurm"] = {
        "partition": "", "account": "", "email": "", "mail_type": "END,FAIL",
        "default": {"nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "400G", "time": "7-00:00:00", "gpu": False},
    }

    # ── Notifications ──
    section("Notifications")
    cfg["notifications"] = {}
    cfg["notifications"]["email"] = prompt("Email for job alerts (blank for none)", default="", required=False)

    return cfg


# ============================================================================
# Config review + pretty print
# ============================================================================

def print_config_review(cfg):
    """Pretty-print the config for review."""
    w = width()
    section("Configuration Review")

    data = cfg.get("data", {})
    print(f"  {BOLD}Platform:{RESET}    {data.get('platform', '')}")
    if data.get("experiment_dir"):
        print(f"  {BOLD}Data mode:{RESET}   experiment")
        print(f"  {BOLD}Path:{RESET}        {data['experiment_dir']}")
    elif data.get("sample_dir"):
        print(f"  {BOLD}Data mode:{RESET}   single sample")
        print(f"  {BOLD}Path:{RESET}        {data['sample_dir']}")

    if data.get("exclude"):
        print(f"  {BOLD}Exclude:{RESET}     {', '.join(data['exclude'])}")

    paths = cfg.get("paths", {})
    print(f"  {BOLD}Container:{RESET}   {paths.get('container_sif', '')}")
    override = paths.get("output_base_override", "")
    print(f"  {BOLD}Output:{RESET}      {override if override else '(next to raw data)'}")

    # Methods
    methods = cfg.get("methods", {})
    enabled = [m for m, mc in methods.items() if mc.get("enabled", False)]
    disabled = [m for m, mc in methods.items() if not mc.get("enabled", False)]

    print(f"\n  {BOLD}Methods enabled:{RESET}")
    for m in enabled:
        mc = methods[m]
        slurm_info = mc.get("slurm", {})
        mem = slurm_info.get("mem", "")
        gpu = " +GPU" if slurm_info.get("gpu", False) else ""
        print(f"    {CHECK} {m:12s}  {DIM}{mem}{gpu}{RESET}")

    if disabled:
        for m in disabled:
            print(f"    {DIM}  {m:12s}  disabled{RESET}")

    # Notifications
    notif = cfg.get("notifications", {})
    if notif.get("email"):
        print(f"\n  {BOLD}Notify:{RESET}      email → {notif['email']}")

    print()


# ============================================================================
# Submit logic (reused from launch_pipeline.py)
# ============================================================================

def submit_job(script_path, dependency_ids=None):
    """Submit via sbatch. Returns job ID or None."""
    import subprocess
    cmd = ["sbatch"]
    if dependency_ids:
        dep_str = ":".join(str(d) for d in dependency_ids)
        cmd.extend(["--dependency", f"afterok:{dep_str}"])
    cmd.append(str(script_path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    {RED}sbatch error: {result.stderr.strip()}{RESET}")
        return None
    return result.stdout.strip().split()[-1]


def generate_and_submit(cfg, config_path, do_submit=False):
    """Generate SLURM scripts and optionally submit them."""
    samples = discover_samples(cfg)
    methods = list_enabled_methods(cfg)

    primary = [m for m in methods if m not in ("cellspa_qc", "fastreseg")]
    post = [m for m in methods if m == "fastreseg"]
    run_qc = "cellspa_qc" in methods

    out_path = Path("scripts/slurm/generated")
    out_path.mkdir(parents=True, exist_ok=True)

    by_slide = defaultdict(list)
    for s in samples:
        by_slide[s.slide_name].append(s)

    # Pre-create log directories — SLURM needs these to exist BEFORE the job starts
    pipeline_root = str(Path(config_path).resolve().parent.parent)
    for sample in samples:
        sample.log_dir_in_pipeline(pipeline_root).mkdir(parents=True, exist_ok=True)

    total = len(samples) * (len(primary) + len(post) + (1 if run_qc else 0))

    section("Job Generation")
    print(f"  {BOLD}{len(samples)}{RESET} sample(s) × {BOLD}{len(primary)}{RESET} method(s)" +
          (f" + {len(post)} post-hoc" if post else "") +
          (f" + QC" if run_qc else "") +
          f" = {BOLD}{total}{RESET} jobs\n")

    all_scripts = []
    all_job_ids = []

    for slide_name, slide_samples in by_slide.items():
        print(f"  {BOLD}{slide_name}{RESET} ({len(slide_samples)} sample(s))")

        for sample in slide_samples:
            seg_ids = []

            for method in primary:
                if method not in METHOD_SCRIPTS:
                    continue
                content = generate_slurm_script(cfg, method, sample, config_path)
                fname = f"submit_{method}_{sample.sample_id}.sh"
                spath = out_path / fname
                with open(spath, "w") as f:
                    f.write(content)
                os.chmod(spath, 0o755)
                all_scripts.append(spath)

                if do_submit:
                    jid = submit_job(spath)
                    if jid:
                        seg_ids.append(jid)
                        all_job_ids.append(jid)
                        print(f"    {CHECK} {sample.sample_id}/{method} {ARROW} job {jid}")
                    else:
                        print(f"    {CROSS} {sample.sample_id}/{method}")
                else:
                    print(f"    {CHECK} {fname}")

            post_ids = []
            for method in post:
                if method not in METHOD_SCRIPTS:
                    continue
                content = generate_slurm_script(cfg, method, sample, config_path)
                spath = out_path / f"submit_{method}_{sample.sample_id}.sh"
                with open(spath, "w") as f:
                    f.write(content)
                os.chmod(spath, 0o755)
                all_scripts.append(spath)

                if do_submit:
                    jid = submit_job(spath, dependency_ids=seg_ids)
                    if jid:
                        post_ids.append(jid)
                        all_job_ids.append(jid)
                        print(f"    {CHECK} {sample.sample_id}/{method} {ARROW} job {jid} {DIM}(after primary){RESET}")
                else:
                    print(f"    {CHECK} submit_{method}_{sample.sample_id}.sh")

            if run_qc:
                content = generate_slurm_script(cfg, "cellspa_qc", sample, config_path)
                spath = out_path / f"submit_qc_{sample.sample_id}.sh"
                with open(spath, "w") as f:
                    f.write(content)
                os.chmod(spath, 0o755)
                all_scripts.append(spath)

                if do_submit:
                    wait = seg_ids + post_ids
                    jid = submit_job(spath, dependency_ids=wait if wait else None)
                    if jid:
                        all_job_ids.append(jid)
                        print(f"    {CHECK} {sample.sample_id}/qc {ARROW} job {jid} {DIM}(after seg){RESET}")
                else:
                    print(f"    {CHECK} submit_qc_{sample.sample_id}.sh")

        print()

    # Summary
    w = width()
    print(f"  {BOLD}{len(all_scripts)}{RESET} script(s) generated {ARROW} {out_path}/")
    if do_submit:
        print(f"  {BOLD}{len(all_job_ids)}{RESET} job(s) submitted")
        print(f"  Monitor: {BOLD}squeue -u $USER{RESET}")
    print(f"  Config:  {BOLD}{config_path}{RESET}")
    print(f"{'━' * w}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spatial Segmentation Pipeline — Interactive Launcher")
    parser.add_argument("--config", help="Skip wizard, use existing config YAML")
    args = parser.parse_args()

    if args.config:
        # Direct mode — load existing config
        config_path = os.path.realpath(args.config)
        cfg = load_config(config_path)
    else:
        # Interactive wizard
        cfg = wizard()

        # Save config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.realpath(f"config/pipeline_{timestamp}.yaml")
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ── Review ──
    print_config_review(cfg)

    # ── Discover + preview ──
    try:
        samples = discover_samples(cfg)
        by_slide = defaultdict(list)
        for s in samples:
            by_slide[s.slide_name].append(s)

        section("Discovered Samples")
        for slide, ss in by_slide.items():
            print(f"  {slide}/")
            for s in ss:
                print(f"    {ARROW} {s.sample_id}")
        print(f"\n  {BOLD}Total: {len(samples)} sample(s){RESET}\n")
    except Exception as e:
        print(f"\n  {YELLOW}⚠ Cannot discover samples: {e}{RESET}")
        print(f"  {DIM}Generating scripts anyway (paths will be baked in from config){RESET}\n")

    # ── Generate + Submit ──
    print(f"  Generate SLURM scripts?")
    print(f"   {GREEN}*{RESET} y) generate and run")
    print(f"     g) generate only")
    print(f"     n) cancel")
    action = input(f"  Choice [y]: ").strip().lower()
    if not action:
        action = "y"

    if action == "n":
        print(f"\n  {DIM}Exiting. Config saved to: {config_path}{RESET}\n")
        return
    elif action == "g":
        print()
        generate_and_submit(cfg, config_path, do_submit=False)
        print(f"  {DIM}To submit later: python segmentation_pipeline_master.py --config {config_path}{RESET}\n")
    else:
        print()
        generate_and_submit(cfg, config_path, do_submit=True)


if __name__ == "__main__":
    main()
