"""
config_loader.py — Load config, discover samples, resolve output paths.

Supports three data modes:
  1. experiment_dir → crawl all slides, find all output-* samples
  2. slide_dir      → find all output-* samples in one slide
  3. sample_dir     → single output-* folder

Each discovered sample becomes a SampleInfo with resolved input/output paths.
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# SampleInfo — one discovered sample with all paths resolved
# ============================================================================

@dataclass
class SampleInfo:
    """Represents a single discovered sample ready for processing."""
    sample_id: str              # e.g. "XETG00143__0032645"
    sample_dir: Path            # full path to the output-XETG... folder (raw input)
    slide_dir: Path             # parent slide folder
    slide_name: str             # e.g. "20241114__203842__11142024_SPITZER_HN_DYSPLASIA1"
    platform: str               # e.g. "xenium"

    def output_dir(self, method: str, output_base_override: str = "") -> Path:
        """
        Where results go for this sample + method.

        Default: {slide_dir}/{method}_reseg/{sample_id}/
        Override: {output_base_override}/{slide_name}/{method}_reseg/{sample_id}/
        """
        if output_base_override:
            return Path(output_base_override) / self.slide_name / f"{method}_reseg" / self.sample_id
        return self.slide_dir / f"{method}_reseg" / self.sample_id

    def log_dir(self, output_base_override: str = "") -> Path:
        """Where SLURM logs go for this sample's slide."""
        if output_base_override:
            return Path(output_base_override) / self.slide_name / "logs"
        return self.slide_dir / "logs"


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate the pipeline YAML config."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Validate required sections
    if "data" not in cfg:
        raise ValueError("Config missing 'data' section")
    if "paths" not in cfg:
        raise ValueError("Config missing 'paths' section")
    if "container_sif" not in cfg["paths"]:
        raise ValueError("Config missing paths.container_sif")

    data = cfg["data"]
    has_experiment = bool(data.get("experiment_dir"))
    has_sample = bool(data.get("sample_dir"))

    active_modes = sum([has_experiment, has_sample])
    if active_modes == 0:
        raise ValueError(
            "Config data section must set one of: experiment_dir or sample_dir"
        )
    if active_modes > 1:
        raise ValueError(
            "Config data section has multiple modes set. "
            "Use exactly ONE of: experiment_dir or sample_dir"
        )

    return cfg


# ============================================================================
# Sample discovery
# ============================================================================

def _extract_sample_id(folder_name: str) -> str:
    """
    Extract a clean sample ID from an output-* folder name.

    'output-XETG00143__0032645__Region_1__20241114__203854'
    →  'XETG00143__0032645'

    Falls back to the full folder name if the pattern doesn't match.
    """
    # Strip the leading "output-"
    name = folder_name
    if name.startswith("output-"):
        name = name[len("output-"):]

    # Try to extract the cartridge + slide portion: XETG00143__0032645
    # Pattern: letters+digits + __ + digits (the two primary identifiers)
    match = re.match(r"^([A-Za-z0-9]+__\d+)", name)
    if match:
        return match.group(1)

    # Fallback: return everything before __Region or just the full name
    if "__Region" in name:
        return name.split("__Region")[0]

    return name


def _matches_filters(folder_name: str, include: list, exclude: list) -> bool:
    """Check if a sample folder passes include/exclude filters."""
    if exclude:
        for pattern in exclude:
            if pattern in folder_name:
                return False
    if include:
        return any(pattern in folder_name for pattern in include)
    return True  # no include filter = accept all


def discover_samples(cfg: dict) -> List[SampleInfo]:
    """
    Discover all samples based on the config's data mode.

    Returns a sorted list of SampleInfo objects.
    """
    data = cfg["data"]
    platform = data.get("platform", "xenium")
    sample_glob = data.get("sample_glob", "output-*")
    include = data.get("include", []) or []
    exclude = data.get("exclude", []) or []

    samples = []

    if data.get("experiment_dir"):
        # Mode 1: Experiment — find all slides, then all samples in each
        exp_dir = Path(data["experiment_dir"]).resolve()
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

        # Slides are direct subdirectories of the experiment dir
        slide_dirs = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not slide_dirs:
            raise ValueError(f"No slide folders found in: {exp_dir}")

        for slide_dir in slide_dirs:
            sample_dirs = sorted(slide_dir.glob(sample_glob))
            for sd in sample_dirs:
                if sd.is_dir() and _matches_filters(sd.name, include, exclude):
                    samples.append(SampleInfo(
                        sample_id=_extract_sample_id(sd.name),
                        sample_dir=sd.resolve(),
                        slide_dir=slide_dir.resolve(),
                        slide_name=slide_dir.name,
                        platform=platform,
                    ))

    elif data.get("sample_dir"):
        # Single sample
        sample_dir = Path(data["sample_dir"]).resolve()
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

        slide_dir = sample_dir.parent
        samples.append(SampleInfo(
            sample_id=_extract_sample_id(sample_dir.name),
            sample_dir=sample_dir,
            slide_dir=slide_dir,
            slide_name=slide_dir.name,
            platform=platform,
        ))

    if not samples:
        raise ValueError(
            "No samples discovered. Check your data paths and include/exclude filters."
        )

    return samples


# ============================================================================
# Config accessors (method config, container path, etc.)
# ============================================================================

def get_method_config(cfg: dict, method: str) -> dict:
    """Get merged SLURM + params config for a specific method."""
    methods = cfg.get("methods", {})
    if method not in methods:
        raise ValueError(f"Method '{method}' not found in config. Available: {list(methods.keys())}")

    method_cfg = methods[method]

    # Merge SLURM: method overrides → defaults
    slurm_defaults = cfg.get("slurm", {}).get("default", {})
    slurm_method = method_cfg.get("slurm", {})
    merged_slurm = {**slurm_defaults, **slurm_method}

    # Add global SLURM settings
    for key in ["partition", "account", "email", "mail_type"]:
        if key in cfg.get("slurm", {}):
            merged_slurm[key] = cfg["slurm"][key]

    return {
        "enabled": method_cfg.get("enabled", True),
        "slurm": merged_slurm,
        "params": method_cfg.get("params", {}),
    }


def get_output_base_override(cfg: dict) -> str:
    """Get the output base override path (empty string = use default layout)."""
    return cfg.get("paths", {}).get("output_base_override", "") or ""


def get_container_path(cfg: dict) -> str:
    """Return the .sif container path, resolved to absolute."""
    return str(Path(cfg["paths"]["container_sif"]).resolve())


def list_enabled_methods(cfg: dict) -> list:
    """Return list of method names that are enabled."""
    return [
        name for name, mcfg in cfg.get("methods", {}).items()
        if mcfg.get("enabled", True)
    ]


def ensure_sample_dirs(sample: SampleInfo, method: str, output_base: str = "") -> Path:
    """Create output + log directories for a sample/method combo. Returns output dir."""
    out_dir = sample.output_dir(method, output_base)
    log_dir = sample.log_dir(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
