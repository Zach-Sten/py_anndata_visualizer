# Spatial Segmentation Wizard

<img src="img/segmentation_wizard_icon.png" width="200" align="left" hspace="20">

An interactive pipeline for running and benchmarking multiple spatial transcriptomics segmentation methods on HPC clusters via SLURM. All methods execute inside a single Singularity container.

We're working on adding in a docker container as well and are excited to implement a classifier that builds off of reference data to give quick annotations for downstream QC metrics. Stay tuned for more! ✨🪄💫✨

Suggestions for QC elements or methods to add are always welcome! 

<br clear="left">
## Overview

The pipeline wraps multiple segmentation methods behind a single interactive wizard. Point it at your data, pick your methods, and it generates and submits all SLURM jobs — one per sample per method — with automatic dependency chaining so QC runs after segmentation finishes.

## Comparisons

<p align="center">
  <img src="img/comparison_1.png" width="500">
</p>

## Current Status

**Working:**
- **ProSeg** — probabilistic segmentation, full Explorer export
- **Baysor** — Bayesian transcript-based segmentation, dask-parallelized
- **Xenium baseline** — native Xenium segmentation loaded as reference
- **QC report** — automated 4-page PDF comparing all methods, emailed on completion
  - Basic metrics: cell count, genes/cell, counts/cell, % transcripts captured
  - Morphological metrics: cell area, elongation, circularity, compactness, eccentricity, solidity, convexity, density, nuclear ratio
  - All metrics computed from segmentation boundary geometry in Python (shapely)
  - Guide pages interleaved with data pages for interpretation context
- **Notifications** — email + SMS on job start, finish, and error; PDF attached on finish
- **Interactive wizard** — full config generation, sample discovery, job preview, and SLURM submission

**In progress:**
- Cellpose, BIDCell, FastReseg integration (scripts present, validation ongoing)

## Supported Methods

| Method | Type | Status |
|--------|------|--------|
| **ProSeg** | Probabilistic (Rust, via SOPA) | ✓ Working |
| **Baysor** | Bayesian transcript-based (Julia, via SOPA) | ✓ Working |
| **Cellpose** | Neural network (Python, via SOPA) | In progress |
| **BIDCell** | Deep learning (PyTorch) | In progress |
| **FastReseg** | Post-hoc refinement (R) | In progress |

## Quick Start

```bash
pip install pyyaml  # only dependency outside the container

# Interactive wizard — walks you through everything:
python segmentation_wizard.py

# Or use an existing config:
python segmentation_wizard.py --config config/my_config.yaml
```

### Test locally (no HPC needed)

```bash
python setup_test_data.py
python segmentation_wizard.py --config config/test_config.yaml
ls scripts/slurm/generated/
```

## Output

Results are written into `{method}_reseg/` folders alongside your raw data. Raw data is never modified. Each completed sample produces:

| File | Description |
|------|-------------|
| `{sample_id}.h5ad` | AnnData count matrix |
| `cells.zarr.zip` | Cell boundaries for Xenium Explorer |
| `cell_feature_matrix.zarr.zip` | Count matrix for Explorer |
| `run_metadata_{method}.json` | Timing, parameters, SLURM job ID |

QC output per slide:
- `qc_report.pdf` — 4-page comparative report (emailed automatically)
- `morpho_{method}.csv` — per-cell morphological metrics for all methods
- `cellspa_{method}.csv` — basic count-based QC summary

## Architecture

```
segmentation_wizard.py      ← interactive wizard
        │
        ▼
config/*.yaml                        ← saved configurations
        │
        ▼
scripts/slurm/generated/             ← one .sh per sample × method
        │
        ▼  (singularity exec container.sif python ...)
scripts/python/
  run_proseg.py / run_baysor.py / run_cellpose.py / run_bidcell.py / run_qc.py
        │
        ▼
scripts/utils/
  data_io.py        ← shared loading, patching, aggregation, export
  config_loader.py  ← config parsing + sample discovery
  notify.py         ← email/SMS notifications
```

## Container

All segmentation methods run inside a single Singularity container (`container/Singularity_spatial_segmentation_v3`). Contains Python 3.10, SOPA, ProSeg, Baysor, Cellpose, BIDCell, FastReseg, CellSPA, scanpy, spatialdata, PyTorch + CUDA, R + spatial packages.

```bash
sudo -E singularity build seg_sin_v3.sif Singularity_spatial_segmentation_v3
```

## Requirements

- **Local:** Python 3.7+ with `pyyaml`
- **HPC:** Singularity, SLURM, the built `.sif`
- **Notifications:** `sendmail` available on the cluster

## Future Plans

- **Reference-based annotation tool** — lightweight interactive tool for annotating segmented cells using reference transcriptomic data, enabling rapid cell type assignment and cross-method comparison in tissue context
- Additional segmentation method integrations
- Expanded QC metrics and spatial analysis
