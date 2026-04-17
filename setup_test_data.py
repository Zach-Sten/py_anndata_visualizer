#!/usr/bin/env python3
"""
setup_test_data.py — Create a fake experiment directory tree for testing.

Generates a mock spatial transcriptomics experiment with multiple slides
and samples so you can test the pipeline wizard and SLURM script generation
without real data.

Usage:
    python setup_test_data.py
    python segmentation_wizard.py --config config/test_config.yaml
"""

from pathlib import Path

TEST_ROOT = Path("test_data")

# Mock experiment: 3 slides, varying numbers of samples per slide
slides = {
    "slide_01_experiment_A": [
        "output-SAM001__000001__Region_1__20240101__120000",
        "output-SAM001__000002__Region_1__20240101__120000",
    ],
    "slide_02_experiment_A": [
        "output-SAM002__000003__Region_1__20240201__140000",
    ],
    "slide_03_experiment_B": [
        "output-SAM003__000004__Region_1__20240301__100000",
        "output-SAM003__000005__Region_1__20240301__100000",
        "output-SAM003__000006__Region_1__20240301__100000",
    ],
}

# Placeholder files that mimic typical spatial transcriptomics output
PLACEHOLDER_FILES = [
    "experiment.xenium",
    "transcripts.parquet",
    "cells.parquet",
    "cells.csv.gz",
    "cell_feature_matrix.h5",
    "morphology.ome.tif",
]

print("Creating test directory tree...\n")

total_samples = 0
for slide_name, sample_dirs in slides.items():
    for sample_dir in sample_dirs:
        full_path = TEST_ROOT / slide_name / sample_dir
        full_path.mkdir(parents=True, exist_ok=True)

        for fname in PLACEHOLDER_FILES:
            (full_path / fname).touch()

        total_samples += 1
        print(f"  {full_path}/")

print(f"\n  {total_samples} sample(s) across {len(slides)} slide(s)")
print(f"\nDone! Tree created at: {TEST_ROOT.resolve()}")
print(f"\nNext steps:")
print(f"  python segmentation_wizard.py --config config/test_config.yaml")
print(f"  # or run the interactive wizard:")
print(f"  python segmentation_wizard.py")
