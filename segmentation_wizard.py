#!/usr/bin/env python3
"""
segmentation_wizard.py — Interactive terminal launcher.

Walk through pipeline configuration step by step, preview the config,
discover samples, review the job plan, and submit — all from one command.

Usage:
    python segmentation_wizard.py

    # Or skip the wizard and use an existing config:
    python segmentation_wizard.py --config config/my_config.yaml
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
    get_method_config, get_output_base_override, is_slide_mode, SampleInfo,
)
from slurm.generate_slurm import (
    generate_slurm_script, generate_classifier_script,
    generate_multi_qc_script, METHOD_SCRIPTS,
)


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
    hint = "y/n [y]" if default else "y/n [n]"
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
    val = input(f"  Select [{','.join(str(options.index(d)+1) for d in defaults)}]: ").strip().strip("'\"")

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


def path_prompt(label, default=None, must_exist=False, required=True):
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
        val = prompt(label, default=default, required=required)
        if not val:
            # Reset completer and return empty (optional path skipped)
            try:
                import readline
                readline.set_completer(None)
            except ImportError:
                pass
            return ""
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

    # ── Reference data ──
    section("Reference Data")
    print(f"  {DIM}Reference h5ad used for cell type classification.{RESET}")
    print(f"  {DIM}Leave blank to skip classification.{RESET}\n")
    ref_path = path_prompt("Reference h5ad path (blank to skip)", default="", must_exist=False, required=False)
    cfg["data"]["reference_path"] = ref_path
    if ref_path:
        cfg["data"]["reference_celltype_col"] = prompt(
            "Cell type column in reference .obs", default="cell_type"
        ).strip("'\"")  # strip accidental quotes
    else:
        cfg["data"]["reference_celltype_col"] = ""

    # ── Data ──
    section("Data")
    cfg["data"]["platform"] = prompt_choice(
        "Platform:", ["xenium", "cosmx", "merscope", "stereoseq"], default="xenium"
    )

    data_mode = prompt_choice(
        "Data mode:",
        ["slide (multiple samples)", "single sample"],
        default="slide (multiple samples)",
    )

    if data_mode.startswith("slide"):
        cfg["data"]["slide_dir"] = path_prompt(
            "Slide directory path (folder containing multiple sample sub-folders)", must_exist=True
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

    # Ask GPU/CPU for any selected GPU-capable methods
    GPU_METHODS = {"cellpose", "bidcell"}
    gpu_partition = None  # ask once if any GPU method is enabled
    print()
    for method in [m for m in selected if m in GPU_METHODS]:
        use_gpu = prompt_yn(f"Use GPU for {method}?", default=True)
        if not use_gpu:
            METHOD_DEFAULTS[method]["slurm"]["gpu"] = False
            if method == "cellpose":
                METHOD_DEFAULTS[method]["params"]["gpu"] = False
        else:
            # Ask for GPU partition once (GPU nodes are often in a specific partition)
            if gpu_partition is None:
                gpu_partition = prompt("  GPU partition", default="common").strip() or "common"
            METHOD_DEFAULTS[method]["slurm"]["partition"] = gpu_partition

    # Ask patch sizes for SOPA-based methods (baysor, proseg, cellpose)
    SOPA_METHODS = {"baysor", "proseg", "cellpose"}
    sopa_selected = [m for m in selected if m in SOPA_METHODS]
    if sopa_selected:
        print()
        print(f"  {BOLD}SOPA patch settings{RESET} {DIM}(baysor / proseg / cellpose){RESET}")
        print(f"  {DIM}Larger patches = more transcripts per patch = more stable segmentation{RESET}")
        transcript_patch_width = int(prompt("  Transcript patch width (µm)", default="500").strip() or "500")
        image_patch_width      = int(prompt("  Image patch width (px)",      default="1200").strip() or "1200")
        for m in sopa_selected:
            if "patch_width" in METHOD_DEFAULTS[m]["params"]:
                METHOD_DEFAULTS[m]["params"]["patch_width"] = transcript_patch_width
            if "image_patch_width" in METHOD_DEFAULTS[m]["params"]:
                METHOD_DEFAULTS[m]["params"]["image_patch_width"] = image_patch_width

    # Ask Baysor prior if selected
    if "baysor" in selected:
        print()
        print(f"  {BOLD}Baysor prior segmentation{RESET}")
        prior_options = [
            ("cell_boundaries",    "Xenium native (default) — use native Xenium cell boundaries"),
            ("cellpose_boundaries","Cellpose — use cellpose output as prior (requires cellpose)"),
            ("none",               "No prior — transcript-only, not recommended unless all else fails"),
        ]
        for i, (key, desc) in enumerate(prior_options, 1):
            default_marker = f"  {DIM}← default{RESET}" if i == 1 else ""
            print(f"    {i}. {desc}{default_marker}")
        raw = prompt("  Choice [1-3]", default="1").strip() or "1"
        try:
            idx = int(raw) - 1
            prior_key = prior_options[idx][0] if 0 <= idx < len(prior_options) else "cell_boundaries"
        except (ValueError, IndexError):
            prior_key = "cell_boundaries"
        if prior_key == "none":
            METHOD_DEFAULTS["baysor"]["params"].pop("prior_shapes_key", None)
            print(f"  {CHECK} Baysor prior: {BOLD}none (transcript-only){RESET}")
        else:
            METHOD_DEFAULTS["baysor"]["params"]["prior_shapes_key"] = prior_key
            print(f"  {CHECK} Baysor prior: {BOLD}{prior_key}{RESET}")
        print(f"  {DIM}Tip: if Baysor fails, try decreasing patch sizes first; as a fallback, rerun with no prior.{RESET}")

    # Ask FastReseg source method if selected
    fastreseg_source = "xenium"
    if "fastreseg" in selected:
        available_sources = ["xenium", "proseg", "baysor", "cellpose", "bidcell"]
        print()
        print(f"  {BOLD}FastReseg{RESET} — which segmentation to refine?")
        for i, src in enumerate(available_sources, 1):
            label = "10x Xenium (original)" if src == "xenium" else src
            default_marker = f"  {DIM}← default{RESET}" if i == 1 else ""
            print(f"    {i}. {label}{default_marker}")
        raw = prompt(f"  Choice [1-{len(available_sources)}]", default="1")
        try:
            idx = int(raw.strip()) - 1
            fastreseg_source = available_sources[idx] if 0 <= idx < len(available_sources) else "xenium"
        except (ValueError, IndexError):
            fastreseg_source = "xenium"
        label = "10x Xenium (original)" if fastreseg_source == "xenium" else fastreseg_source
        print(f"  {CHECK} FastReseg will refine: {BOLD}{label}{RESET}")

    # Ask memory per selected method
    MEM_DEFAULTS = {
        "proseg": "400G", "baysor": "600G", "cellpose": "200G",
        "bidcell": "300G", "fastreseg": "400G",
    }
    fastreseg_mem = "400G"
    if selected:
        print()
        print(f"  {BOLD}Memory per method{RESET} {DIM}(adjust if OOM errors occur){RESET}")
        for m in selected:
            default_mem = MEM_DEFAULTS.get(m, "400G")
            mem = prompt(f"  {m} memory (e.g. 400G)", default=default_mem).strip() or default_mem
            if mem[-1].isdigit():
                mem += "G"
            if m in METHOD_DEFAULTS:
                METHOD_DEFAULTS[m]["slurm"]["mem"] = mem
            elif m == "fastreseg":
                fastreseg_mem = mem

    for method in all_methods:
        enabled = method in selected
        cfg["methods"][method] = {"enabled": enabled}
        if enabled and method in METHOD_DEFAULTS:
            cfg["methods"][method].update(METHOD_DEFAULTS[method])
        elif method == "fastreseg" and enabled:
            cfg["methods"][method].update({
                "slurm": {"mem": fastreseg_mem, "cpus_per_task": 4, "time": "1-00:00:00"},
                "params": {"source_method": fastreseg_source, "explorer_mode": "+cb"},
            })

    # Classifier — requires a reference dataset to be set
    print()
    run_classifier = False
    classifier_gpu = False
    classifier_retrain = False
    if cfg["data"].get("reference_path"):
        run_classifier = prompt_yn("Classify cell types?", default=False)
        if run_classifier:
            # Ask use_rank FIRST so the cache check can include it
            classifier_use_rank = prompt_yn(
                "Use rank-based feature matrix? (recommended; disable to use raw counts directly)",
                default=True,
            )

            # Check if a cached model already exists for this data_dir
            _output_base = cfg.get("paths", {}).get("output_base_override", "")
            _slide_dir   = cfg["data"].get("slide_dir", "")
            _smp_dir     = cfg["data"].get("sample_dir", "")
            if _output_base:
                _data_dir = Path(_output_base)
            elif _slide_dir:
                _data_dir = Path(_slide_dir)
            else:
                _data_dir = Path(_smp_dir).parent
            _ref_path   = cfg["data"].get("reference_path", "")
            _ref_col    = cfg["data"].get("reference_celltype_col", "cell_type")
            _ref_stem   = Path(_ref_path).stem if _ref_path else "classifier"
            _cache      = _data_dir / f"classifier_cache_{_ref_stem}" / "model.json"
            _cache_info = _data_dir / f"classifier_cache_{_ref_stem}" / "cache_info.json"
            _cache_matches = False
            if _cache.exists():
                if _cache_info.exists() and _ref_path:
                    import json as _json
                    _meta = _json.loads(_cache_info.read_text())
                    _cached_ref  = _meta.get("reference_path", "")
                    _cached_col  = _meta.get("celltype_col", "").strip("'\"")
                    _cached_rank = _meta.get("use_rank", True)
                    _cache_matches = (
                        _cached_ref  == str(Path(_ref_path).resolve()) and
                        _cached_col  == _ref_col.strip("'\"") and
                        _cached_rank == classifier_use_rank
                    )
                else:
                    # No metadata — treat as stale when a reference is specified
                    _cache_matches = not bool(_ref_path)
            if _cache_matches:
                print(f"  {CHECK} Cached model found: {_cache.parent.name}  (ref: {Path(_ref_path).name})")
                classifier_retrain = not prompt_yn("Use cached model?", default=True)
                if classifier_retrain:
                    print(f"  {DIM}Will retrain from scratch{RESET}")
            elif _cache.exists():
                print(f"  {DIM}Cache exists but does not match current settings — will retrain{RESET}")
            classifier_gpu = prompt_yn("Use GPU for classifier (XGBoost)?", default=False)
    else:
        print(f"  {DIM}Cell type classification skipped — no reference path provided{RESET}")
        classifier_use_rank = True
    cfg["methods"]["classifier"] = {
        "enabled": run_classifier,
        "slurm": {"mem": "100G", "cpus_per_task": 8, "gpu": classifier_gpu, "time": "1-00:00:00"},
        "params": {
            "retrain":   classifier_retrain,
            "use_rank":  classifier_use_rank,
        },
    }
    cfg["methods"]["xenium_export"] = {
        "enabled": run_classifier,
        "slurm": {"mem": "50G", "cpus_per_task": 4, "time": "0-02:00:00"},
        "params": {},
    }

    print()
    if not selected:
        print(f"  {DIM}No segmentation methods selected — QC will run on existing results{RESET}")
    if run_classifier:
        run_qc = True
        print(f"  {CHECK} CellSPA QC enabled automatically (classifier requires it)")
    else:
        run_qc = prompt_yn("Run CellSPA QC?", default=True)
    cfg["methods"]["cellspa_qc"] = {
        "enabled": run_qc,
        "slurm": {"mem": "100G", "cpus_per_task": 4, "time": "0-12:00:00"},
        "params": {"methods_to_qc": selected},
    }

    cfg["methods"]["celltype_qc"] = {
        "enabled": False,
        "slurm": {"mem": "100G", "cpus_per_task": 4, "time": "0-06:00:00"},
        "params": {},
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
    elif data.get("slide_dir"):
        print(f"  {BOLD}Data mode:{RESET}   slide (multiple samples)")
        print(f"  {BOLD}Path:{RESET}        {data['slide_dir']}")
    elif data.get("sample_dir"):
        print(f"  {BOLD}Data mode:{RESET}   single sample")
        print(f"  {BOLD}Path:{RESET}        {data['sample_dir']}")

    if data.get("reference_path"):
        print(f"  {BOLD}Reference:{RESET}   {data['reference_path']}")
        print(f"  {BOLD}Celltype col:{RESET} {data.get('reference_celltype_col', 'cell_type')}")
    else:
        print(f"  {BOLD}Reference:{RESET}   {DIM}(none — classification disabled){RESET}")

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

def submit_job(script_path, dependency_ids=None, afterany=False):
    """Submit via sbatch. Returns job ID or None."""
    import subprocess
    cmd = ["sbatch"]
    if dependency_ids:
        dep_str = ":".join(str(d) for d in dependency_ids)
        dep_type = "afterany" if afterany else "afterok"
        cmd.extend(["--dependency", f"{dep_type}:{dep_str}"])
    cmd.append(str(script_path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    {RED}sbatch error: {result.stderr.strip()}{RESET}")
        return None
    return result.stdout.strip().split()[-1]


def _generate_chain_notify_script(
    chain_id: str, all_job_ids: list, manifest_path: Path,
    qc_pdf_paths: list, cfg: dict, pipeline_root: str, log_dir: Path,
) -> str:
    """Generate a lightweight SLURM script that sends the chain summary email."""
    notif      = cfg.get("notifications", {})
    email      = notif.get("email", "")
    phone      = notif.get("phone", "")
    phone_arg  = f"--phone {phone}" if phone else ""
    attach_args = " ".join(f'"{p}"' for p in qc_pdf_paths) if qc_pdf_paths else ""

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=seg_chain_notify_{chain_id}",
        f"#SBATCH --output={log_dir}/chain_notify_{chain_id}_%j.out",
        f"#SBATCH --error={log_dir}/chain_notify_{chain_id}_%j.err",
        "#SBATCH --time=0-00:15:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=4G",
        "",
        f"cd {pipeline_root}",
        "",
        f"python scripts/utils/notify_chain.py \\",
        f"    --email {email} \\",
    ]
    if phone_arg:
        lines.append(f"    {phone_arg} \\")
    lines += [
        f"    --manifest {manifest_path} \\",
        f"    --event finish \\",
    ]
    if attach_args:
        lines.append(f"    --attachments {attach_args}")
    else:
        lines[-1] = lines[-1].rstrip(" \\")  # clean trailing backslash on --event line

    return "\n".join(lines) + "\n"


def generate_and_submit(cfg, config_path, do_submit=False):
    """Generate SLURM scripts and optionally submit them."""
    import json, subprocess as _sp
    samples = discover_samples(cfg)
    methods = list_enabled_methods(cfg)

    _skip = {"cellspa_qc", "fastreseg", "classifier", "celltype_qc", "xenium_export"}
    primary = [m for m in methods if m not in _skip]
    post = [m for m in methods if m == "fastreseg"]
    run_qc = "cellspa_qc" in methods
    run_classifier = cfg.get("methods", {}).get("classifier", {}).get("enabled", False)

    out_path = Path("scripts/slurm/generated")
    out_path.mkdir(parents=True, exist_ok=True)

    multi_sample = is_slide_mode(cfg)

    by_slide = defaultdict(list)
    for s in samples:
        by_slide[s.slide_name].append(s)

    # Pre-create log directories — SLURM needs these to exist BEFORE the job starts
    pipeline_root = str(Path(config_path).resolve().parent.parent)
    for sample in samples:
        sample.log_dir_in_pipeline(pipeline_root).mkdir(parents=True, exist_ok=True)

    # Chain ID + manifest for consolidated notifications
    chain_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    chain_log_dir = Path(pipeline_root) / "logs" / f"chain_{chain_id}"
    chain_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chain_log_dir / "manifest.json"
    chain_manifest: dict = {"chain_id": chain_id, "jobs": []}

    notif       = cfg.get("notifications", {})
    notify_email = notif.get("email", "")
    notify_phone = notif.get("phone", "")
    has_notify   = bool(notify_email)

    # QC PDF paths (one per sample in single mode; one combined in multi-sample mode)
    output_base = get_output_base_override(cfg)
    qc_pdf_paths = []
    if multi_sample:
        # All samples share one slide_dir → one combined PDF
        slide_dir = samples[0].slide_dir
        qc_base = (Path(output_base) / slide_dir.name if output_base else slide_dir)
        qc_pdf_paths.append(str(qc_base / "qc" / "combined" / "qc_report.pdf"))
    else:
        for sample in samples:
            qc_dir = (
                (Path(output_base) / sample.slide_name / "qc" / sample.sample_id)
                if output_base else
                (sample.slide_dir / "qc" / sample.sample_id)
            )
            qc_pdf_paths.append(str(qc_dir / "qc_report.pdf"))

    classify_count = 1 if run_classifier else 0
    xenium_export_count = len(samples) if run_classifier else 0
    qc_count = 1 if run_qc else 0  # always 1 job (combined or per-sample loop is N but 1 in slide mode)
    total = (len(samples) * (len(primary) + len(post))
             + xenium_export_count + classify_count
             + (len(samples) * qc_count if not multi_sample else qc_count))

    section("Job Generation")
    qc_desc = " + 1 combined QC" if (run_qc and multi_sample) else (f" + QC" if run_qc else "")
    print(f"  {BOLD}{len(samples)}{RESET} sample(s) × {BOLD}{len(primary)}{RESET} method(s)" +
          (f" + {len(post)} post-hoc" if post else "") +
          (f" + xenium_export + 1 classifier (all samples)" if run_classifier else "") +
          qc_desc +
          f" = {BOLD}{total}{RESET} jobs\n")

    all_scripts = []
    all_job_ids = []
    all_seg_post_ids = []    # every seg+post ID across all samples (classifier dep)
    xenium_export_ids = []   # xenium_export job IDs (also fed into classifier dep)
    qc_scripts = []          # (spath, sample) — submitted after classifier

    # ── Pass 1: seg + post jobs per sample ──
    for slide_name, slide_samples in by_slide.items():
        print(f"  {BOLD}{slide_name}{RESET} ({len(slide_samples)} sample(s))")

        for sample in slide_samples:
            seg_ids = []

            # ── Xenium export (no dependencies — runs alongside seg jobs) ──
            if run_classifier:
                content = generate_slurm_script(cfg, "xenium_export", sample, config_path)
                fname = f"submit_xenium_export_{sample.sample_id}.sh"
                spath = out_path / fname
                with open(spath, "w") as f:
                    f.write(content)
                os.chmod(spath, 0o755)
                all_scripts.append(spath)

                if do_submit:
                    jid = submit_job(spath)
                    if jid:
                        xenium_export_ids.append(jid)
                        all_job_ids.append(jid)
                        chain_manifest["jobs"].append(
                            {"method": "xenium_export", "sample_id": sample.sample_id, "job_id": jid}
                        )
                        print(f"    {CHECK} {sample.sample_id}/xenium_export {ARROW} job {jid}")
                    else:
                        print(f"    {CROSS} {sample.sample_id}/xenium_export")
                else:
                    print(f"    {CHECK} {fname}")

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
                        all_seg_post_ids.append(jid)
                        chain_manifest["jobs"].append(
                            {"method": method, "sample_id": sample.sample_id, "job_id": jid}
                        )
                        print(f"    {CHECK} {sample.sample_id}/{method} {ARROW} job {jid}")
                    else:
                        print(f"    {CROSS} {sample.sample_id}/{method}")
                else:
                    print(f"    {CHECK} {fname}")

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
                    post_deps = seg_ids + xenium_export_ids
                    jid = submit_job(spath, dependency_ids=post_deps if post_deps else None)
                    if jid:
                        all_job_ids.append(jid)
                        all_seg_post_ids.append(jid)
                        chain_manifest["jobs"].append(
                            {"method": method, "sample_id": sample.sample_id, "job_id": jid}
                        )
                        dep_note = "after xenium_export" if not seg_ids else "after primary + xenium_export"
                        print(f"    {CHECK} {sample.sample_id}/{method} {ARROW} job {jid} {DIM}({dep_note}){RESET}")
                else:
                    print(f"    {CHECK} submit_{method}_{sample.sample_id}.sh")

            # Per-sample QC only in single/experiment mode (multi_sample uses one combined job)
            if run_qc and not multi_sample:
                content = generate_slurm_script(cfg, "cellspa_qc", sample, config_path)
                spath = out_path / f"submit_qc_{sample.sample_id}.sh"
                with open(spath, "w") as f:
                    f.write(content)
                os.chmod(spath, 0o755)
                all_scripts.append(spath)
                qc_scripts.append((spath, sample))
                if not do_submit:
                    print(f"    {CHECK} submit_qc_{sample.sample_id}.sh")

        print()

    # ── ONE classifier job — trains once, predicts on every *_reseg h5ad found ──
    classify_id = None
    if run_classifier:
        content = generate_classifier_script(cfg, config_path, sample_ids=[s.sample_id for s in samples])
        fname = "submit_classify_all.sh"
        spath = out_path / fname
        with open(spath, "w") as f:
            f.write(content)
        os.chmod(spath, 0o755)
        all_scripts.append(spath)

        if do_submit:
            dep = all_seg_post_ids + xenium_export_ids or None
            jid = submit_job(spath, dependency_ids=dep)
            if jid:
                classify_id = jid
                all_job_ids.append(jid)
                chain_manifest["jobs"].append(
                    {"method": "classifier", "sample_id": "all", "job_id": jid}
                )
                dep_note = f" {DIM}(after all seg jobs){RESET}" if dep else ""
                print(f"  {CHECK} classifier/all {ARROW} job {jid}{dep_note}")
            else:
                print(f"  {CROSS} classifier/all")
        else:
            print(f"  {CHECK} {fname}")
        print()

    # ── Pass 2: QC (depends on seg + classifier) ──
    if run_qc:
        wait = all_seg_post_ids + ([classify_id] if classify_id else [])

        if multi_sample:
            # ONE combined QC job across all samples
            content = generate_multi_qc_script(cfg, config_path, samples)
            spath = out_path / "submit_qc_combined.sh"
            with open(spath, "w") as f:
                f.write(content)
            os.chmod(spath, 0o755)
            all_scripts.append(spath)
            if do_submit:
                jid = submit_job(spath, dependency_ids=wait if wait else None)
                if jid:
                    all_job_ids.append(jid)
                    chain_manifest["jobs"].append(
                        {"method": "cellspa_qc", "sample_id": "combined", "job_id": jid}
                    )
                    print(f"  {CHECK} qc/combined {ARROW} job {jid} {DIM}(after classifier){RESET}")
            else:
                print(f"  {CHECK} submit_qc_combined.sh")
            print()
        elif do_submit:
            # Per-sample QC (single/experiment mode)
            for spath, sample in qc_scripts:
                jid = submit_job(spath, dependency_ids=wait if wait else None)
                if jid:
                    all_job_ids.append(jid)
                    chain_manifest["jobs"].append(
                        {"method": "cellspa_qc", "sample_id": sample.sample_id, "job_id": jid}
                    )
                    print(f"    {CHECK} {sample.sample_id}/qc {ARROW} job {jid} {DIM}(after classifier){RESET}")
            print()

    # ── Chain notifications ──
    if do_submit and has_notify and all_job_ids:
        # Write manifest so chain_notify job can read it later
        manifest_path.write_text(json.dumps(chain_manifest, indent=2))

        # Send start email immediately from here (login node)
        _sp.run(
            ["python", f"{pipeline_root}/scripts/utils/notify_chain.py",
             "--email", notify_email,
             "--manifest", str(manifest_path),
             "--event", "start"],
            check=False,
        )

        # Generate chain_notify SLURM job (afterany — runs regardless of job outcomes)
        notify_script_content = _generate_chain_notify_script(
            chain_id, all_job_ids, manifest_path,
            qc_pdf_paths, cfg, pipeline_root, chain_log_dir,
        )
        notify_spath = out_path / f"chain_notify_{chain_id}.sh"
        notify_spath.write_text(notify_script_content)
        os.chmod(notify_spath, 0o755)
        notify_jid = submit_job(notify_spath, dependency_ids=all_job_ids, afterany=True)
        if notify_jid:
            print(f"  {CHECK} chain_notify {ARROW} job {notify_jid} {DIM}(after all jobs){RESET}")

    # Summary
    w = width()
    print(f"  {BOLD}{len(all_scripts)}{RESET} script(s) generated {ARROW} {out_path}/")
    if do_submit:
        print(f"  {BOLD}{len(all_job_ids)}{RESET} job(s) submitted")
        print(f"  Monitor: {BOLD}squeue -u $USER{RESET}")
    print(f"  Config:  {BOLD}{config_path}{RESET}")
    print(f"{'━' * w}")
    print()

    PURPLE = "\033[35m"
    print(PURPLE + BOLD + f"    ·  ✦      ·    ✧        ✦   ·      ✧    ·   ✦     ·" + RESET)
    print(PURPLE + BOLD + f"·  ✦      ·   ✧    ·   Off I go!   ✧   ·   ✦   ·   ✧ · " + RESET)
    print(PURPLE + BOLD + f"  ✧       ·       ✦    ·  ✧       ·  ✦      ✧       ·" + RESET)
    rocket = r"""
                                  ....
                                .'' .'''
.                             .'   :
\                          .:    :
 \                        _:    :       ..----.._
  \                    .:::.....:::.. .'         ''.
   \                 .'  #-. .-######'     #        '.
    \                 '.##'/ ' ################       :
     \                  #####################         :
      \               ..##.-.#### .''''###'.._        :
       \             :--:########:            '.    .' :
        \..__...--.. :--:#######.'   '.         '.     :
        :     :  : : '':'-:'':'::        .         '.  .'
        '---'''..: :    ':    '..'''.      '.        :'
           \  :: : :     '      ''''''.     '.      .:
            \ ::  : :     '            '.      '      :
             \::   : :           ....' ..:       '     '.
              \::  : :    .....####\ .~~.:.             :
               \':.:.:.:'#########.===. ~ |.'-.   . '''.. :
                \    .'  ########## \ \ _.' '. '-.       '''.
                :\  :     ########   \ \      '.  '-.        :
               :  \'    '   #### :    \ \      :.    '-.      :
              :  .'\ :'  :     :     \ \       :      '-.    :
             : .'  .\ '  :      :     :\ \       :        '.   :
             ::   :  \'  :.      :     : \ \      :          '. :
             ::. :    \  : :      :    ;  \ \     :           '.:
              : ':    '\ :  :     :     :  \:\     :        ..'
                 :    ' \ :        :     ;  \|      :   .'''
                 '.   '  \:                         :.''
                  .:..... \:       :            ..''
                 '._____|'.\ ......'''''''.:..'''
                          \
"""
    print(PURPLE + BOLD + rocket + RESET)
    print(PURPLE + BOLD + f"    ·  ✦      ·    ✧        ✦   ·      ✧    ·   ✦     ·" + RESET)
    print(PURPLE + BOLD + f"  ✧       ·       ✦    ·  ✧       ·  ✦      ✧       ·" + RESET)


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

        skip = {"cellspa_qc", "classifier", "celltype_qc"}
        enabled = [m for m, mc in cfg.get("methods", {}).items() if mc.get("enabled") and m not in skip]
        qc_enabled         = cfg.get("methods", {}).get("cellspa_qc",   {}).get("enabled", False)
        classifier_enabled = cfg.get("methods", {}).get("classifier",   {}).get("enabled", False)
        celltype_qc_enabled = cfg.get("methods", {}).get("celltype_qc", {}).get("enabled", False)
        method_str = ", ".join(enabled) if enabled else f"{DIM}none{RESET}"
        extra = []
        if qc_enabled:          extra.append("CellSPA QC")
        if classifier_enabled:  extra.append("classifier")
        if celltype_qc_enabled: extra.append("cell type QC")
        extra_str = (" + " + ", ".join(extra)) if extra else ""
        print(f"\n  {BOLD}Total: {len(samples)} sample(s){RESET}")
        print(f"  {BOLD}Methods:{RESET} {method_str}{extra_str}\n")
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
        print(f"  {DIM}To submit later: python segmentation_wizard.py --config {config_path}{RESET}\n")
    else:
        print()
        generate_and_submit(cfg, config_path, do_submit=True)


if __name__ == "__main__":
    main()
