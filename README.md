# Spatial Segmentation Wizard

An interactive pipeline for running and benchmarking multiple spatial transcriptomics segmentation methods on HPC clusters via SLURM. All methods execute inside a single Singularity container.

## Supported Methods

| Method | Type | Language | GPU | Orchestrated by |
|--------|------|----------|-----|-----------------|
| **ProSeg** | Probabilistic | Rust | No | SOPA |
| **Baysor** | Bayesian | Julia | No | SOPA |
| **Cellpose** | Neural network | Python | Yes | SOPA |
| **BIDCell** | Deep learning | Python/PyTorch | Yes | Standalone |
| **FastReseg** | Post-hoc refinement | R | No | Standalone |
| **CellSPA** | QC assessment | R | No | Standalone |

## Quick Start

```bash
pip install pyyaml  # only dependency outside the container

# Interactive wizard — walks you through everything:
python segmentation_pipeline_master.py

# Or use an existing config:
python segmentation_pipeline_master.py --config config/my_config.yaml
```

The wizard prompts for your data path, container location, which methods to run, and optional email/text notifications — then generates and optionally submits all SLURM jobs.

### Test locally (no HPC needed)

```bash
python setup_test_data.py                                            # creates fake directory tree
python segmentation_pipeline_master.py --config config/test_config.yaml  # generates scripts, skip submit
ls scripts/slurm/generated/                                          # inspect what would be submitted
```

## Data Modes

The pipeline auto-discovers sample folders under whichever path you provide. Set one of these in the config (or let the wizard handle it):

**Experiment mode** — process every sample across all slides:
```yaml
data:
  experiment_dir: "/path/to/experiment"
```

**Single sample mode** — process one specific sample folder:
```yaml
data:
  sample_dir: "/path/to/experiment/slide_folder/output-SAMPLE_001"
```

### Filtering

Restrict which samples to process with substring matching:
```yaml
data:
  include: ["SAMPLE_001", "SAMPLE_002"]   # only these (empty = all)
  exclude: ["SAMPLE_003"]                 # skip these
```

## Output Layout

Results are written next to your raw data inside `{method}_reseg/` folders. Raw data is never modified.

```
experiment/
├── slide_folder_1/
│   ├── output-SAMPLE_001/              ← raw data (untouched)
│   ├── output-SAMPLE_002/              ← raw data (untouched)
│   ├── proseg_reseg/
│   │   ├── SAMPLE_001/
│   │   │   ├── SAMPLE_001.h5ad
│   │   │   ├── cells.zarr.zip
│   │   │   ├── cell_feature_matrix.zarr.zip
│   │   │   ├── experiment.xenium
│   │   │   └── run_metadata_proseg.json
│   │   └── SAMPLE_002/
│   ├── baysor_reseg/
│   │   └── ...
│   ├── qc/
│   │   └── ...
│   └── logs/
└── slide_folder_2/
    └── ...
```

To redirect output to a central location:
```yaml
paths:
  output_base_override: "/scratch/user/segmentation_results"
```

## What Gets Saved Per Method

| File | Description |
|------|-------------|
| `{sample_id}.h5ad` | AnnData: cells × genes count matrix |
| `cells.zarr.zip` | Cell boundaries for Xenium Explorer |
| `cell_feature_matrix.zarr.zip` | Count matrix for Explorer |
| `experiment.xenium` | Explorer metadata |
| `run_metadata_{method}.json` | Timing, parameters, SLURM job ID |
| `expected_counts.mtx.gz` | ProSeg only: raw expected counts |

## Notifications

Get a text or email when jobs complete or fail. Configure in the wizard or in the YAML:

```yaml
notifications:
  email: "you@institute.edu"
  phone: "5551234567"        # just your number, no carrier needed
```

Text notifications are sent via email-to-SMS gateways using the cluster's `sendmail`. Verify it's available with `which sendmail`.

## Architecture

```
segmentation_pipeline_master.py      ← interactive wizard (start here)
launch_pipeline.py                   ← non-interactive launcher (for scripting)
        │
        ▼
config/*.yaml                        ← saved configurations
        │
        ▼
scripts/slurm/generated/             ← one .sh per sample × method
  submit_{method}_{sample_id}.sh
        │
        ▼ (each .sh runs: singularity exec container.sif python ...)
scripts/python/
  run_proseg.py
  run_baysor.py
  run_cellpose.py
  run_bidcell.py
  run_fastreseg.py
  run_qc.py
        │
        ▼
scripts/utils/                       ← shared libraries
  config_loader.py                   ← config parsing + sample discovery
  data_io.py                         ← data loading, patching, export
  notify.py                          ← email/SMS notifications

scripts/segger_functions/            ← reference-based comparison metrics
  metrics.py                         ← MECR, contamination, sensitivity, etc.
```

## Notebooks

Launch JupyterLab inside the container:
```bash
singularity exec --nv container.sif jupyter lab --no-browser --port=8888
```

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Load raw data, preview tissue and patches |
| `02_method_results.ipynb` | Per-method deep dive after jobs complete |
| `03_qc_comparison.ipynb` | Cross-method QC benchmarking |
| `04_bidcell_workflow.ipynb` | BIDCell config, mask resize for Explorer |
| `05_r_postprocessing.ipynb` | FastReseg refinement, CellSPA QC (R) |
| `06_segger_comparison.ipynb` | Reference-based comparison (MECR, contamination, sensitivity) |

## Adding a New Method

1. Create `scripts/python/run_newmethod.py` (takes `--config`, `--sample-dir`, `--output-dir`, `--sample-id`)
2. Register it in `scripts/slurm/generate_slurm.py` → `METHOD_SCRIPTS` dict
3. Add a config section under `methods:` in the YAML
4. The launcher picks it up automatically

## Container

Build files in `container/`:
- `Singularity_spatial_segmentation_v1` — build recipe
- `spatial_segmentation_env_v1.yml` — conda environment

```bash
sudo -E singularity build container.sif Singularity_spatial_segmentation_v1
```

Contains: Python 3.10, SOPA, ProSeg, Baysor, BIDCell, Cellpose, FastReseg, CellSPA, scanpy, squidpy, spatialdata, PyTorch + CUDA, R + spatial packages, JupyterLab.

## Requirements

- **Local (wizard only):** Python 3.7+ with `pyyaml`
- **HPC:** Singularity, SLURM, the built `.sif` container
- **Notifications:** `sendmail` on the cluster (check with `which sendmail`)
