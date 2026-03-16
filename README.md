# Spatial Segmentation Pipeline

A unified pipeline for benchmarking and running multiple spatial transcriptomics segmentation methods on HPC via SLURM. All methods run inside a single Singularity container (`seg_sin_V1.sif`).

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
# 1. Edit config — set your data path and container location
cp config/pipeline_config.yaml config/my_experiment.yaml
vim config/my_experiment.yaml

# 2. Check what samples were discovered
python launch_pipeline.py --config config/my_experiment.yaml --list

# 3. Dry run — generate SLURM scripts without submitting
python launch_pipeline.py --config config/my_experiment.yaml

# 4. Submit everything
python launch_pipeline.py --config config/my_experiment.yaml --submit

# 5. Monitor
squeue -u $USER
```

## Three Data Modes

The pipeline auto-discovers Xenium sample folders (`output-*`) under whichever path you provide. Set **one** of these in the config:

### Mode 1 — Full Experiment
Process every sample across all slides:
```yaml
data:
  experiment_dir: "/data/raw/Xenium_dysplasia"
  # Contains: DYSPLASIA1/, DYSPLASIA2/, ... each with output-XETG... folders
```

### Mode 2 — Single Slide
Process all samples within one slide folder:
```yaml
data:
  slide_dir: "/data/raw/Xenium_dysplasia/20241114__203842__SPITZER_HN_DYSPLASIA1"
```

### Mode 3 — Single Sample
Process one specific sample:
```yaml
data:
  sample_dir: "/data/raw/.../output-XETG00143__0032645__Region_1__20241114__203854"
```

### Filtering

Restrict which samples to process with substring matching:
```yaml
data:
  include: ["0032645", "0034280"]   # only these (empty = all)
  exclude: ["0036000"]              # skip these
```

## Output Layout

Results are written **next to your raw data** inside `{method}_reseg/` folders. Raw data is never touched.

```
20241114__203842__11142024_SPITZER_HN_DYSPLASIA1/
├── output-XETG00143__0032645__Region_1.../   ← raw (untouched)
├── output-XETG00143__0034280__Region_1.../   ← raw (untouched)
│
├── proseg_reseg/
│   ├── XETG00143__0032645/
│   │   ├── XETG00143__0032645.h5ad
│   │   ├── expected_counts.mtx.gz
│   │   ├── cells.zarr.zip
│   │   ├── cell_feature_matrix.zarr.zip
│   │   ├── experiment.xenium
│   │   └── run_metadata_proseg.json
│   └── XETG00143__0034280/
│
├── baysor_reseg/
│   ├── XETG00143__0032645/
│   └── XETG00143__0034280/
│
├── cellpose_reseg/
│   └── ...
│
├── qc/
│   ├── XETG00143__0032645/
│   │   ├── qc_violin_proseg.png
│   │   ├── qc_scatter_baysor.png
│   │   └── method_comparison.csv
│   └── XETG00143__0034280/
│
└── logs/
    ├── seg_proseg_XETG00143__0032645_12345.out
    └── ...
```

To redirect all output to a central location instead:
```yaml
paths:
  output_base_override: "/scratch/user/segmentation_results"
```

## What Gets Saved Per Method

Every method produces a consistent set of outputs:

| File | Description |
|------|-------------|
| `{sample_id}.h5ad` | AnnData: cells × genes count matrix |
| `cells.zarr.zip` | Cell boundaries for Xenium Explorer |
| `cell_feature_matrix.zarr.zip` | Count matrix for Explorer |
| `experiment.xenium` | Explorer metadata |
| `run_metadata_{method}.json` | Timing, parameters, SLURM job ID |
| `expected_counts.mtx.gz` | ProSeg only: raw expected counts |

## Pipeline Architecture

```
config/pipeline_config.yaml          ← edit this
        │
        ▼
launch_pipeline.py --list            ← discover samples
launch_pipeline.py --submit          ← generate + submit SLURM jobs
        │
        ▼ (generates one .sh per sample × method)
scripts/slurm/generated/
  submit_proseg_XETG00143__0032645.sh
  submit_baysor_XETG00143__0032645.sh
  submit_qc_XETG00143__0032645.sh
  ...
        │
        ▼ (each .sh calls singularity exec → python runner)
scripts/python/
  run_proseg.py   --sample-dir ... --output-dir ... --sample-id ...
  run_baysor.py
  run_cellpose.py
  run_bidcell.py
  run_fastreseg.py
  run_qc.py
        │
        ▼ (shared utilities)
scripts/utils/
  config_loader.py   ← config parsing + sample discovery
  data_io.py         ← data loading, patching, export, timing
```

## Running Individual Methods

```bash
# Generate + submit just proseg for all samples:
python launch_pipeline.py --config config/my_experiment.yaml --submit --methods proseg

# Or generate SLURM scripts without the launcher:
python scripts/slurm/generate_slurm.py --config config/my_experiment.yaml --method proseg

# Or run locally (no SLURM) for a single sample:
singularity exec --nv seg_sin_V1.sif \
    python scripts/python/run_proseg.py \
    --config config/my_experiment.yaml \
    --sample-dir /path/to/output-XETG... \
    --output-dir /path/to/proseg_reseg/XETG... \
    --sample-id XETG00143__0032645
```

## Notebooks

Launch JupyterLab inside the container:
```bash
singularity exec --nv seg_sin_V1.sif jupyter lab --no-browser --port=8888
```

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Load raw data, preview tissue/patches |
| `02_method_results.ipynb` | Per-method deep dive after jobs complete |
| `03_qc_comparison.ipynb` | Cross-method QC benchmarking |
| `04_bidcell_workflow.ipynb` | BIDCell config, mask resize for Explorer |
| `05_r_postprocessing.ipynb` | FastReseg refinement, CellSPA QC (R) |

## Adding a New Method

1. Create `scripts/python/run_newmethod.py` (takes `--config`, `--sample-dir`, `--output-dir`, `--sample-id`)
2. Register it in `scripts/slurm/generate_slurm.py` → `METHOD_SCRIPTS` dict
3. Add config section under `methods:` in the YAML
4. Done — the launcher picks it up automatically

## Container

Build files are in `container/`:
- `Singularity_spatial_segmentation_v1` — build recipe
- `spatial_segmentation_env_v1.yml` — conda environment

```bash
sudo -E singularity build seg_sin_V1.sif Singularity_spatial_segmentation_v1
```

## Authors

Zachary Stensland, Madison Lotstein, Rebecca Jaszczak
