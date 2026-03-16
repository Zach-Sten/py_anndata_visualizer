#!/usr/bin/env python3
"""
setup_test_data.py — Create a fake experiment directory tree for testing.

Run this once, then use the test config:
    python setup_test_data.py
    python segmentation_pipeline_master.py --config config/test_config.yaml
"""

from pathlib import Path

TEST_ROOT = Path("test_data")

# Two slides, two samples each
slides = {
    "20241114__203842__11142024_SPITZER_HN_DYSPLASIA1": [
        "output-XETG00143__0032645__Region_1__20241114__203854",
        "output-XETG00143__0034280__Region_1__20241114__203854",
    ],
    "20241121__001251__11202024_SPITZER_HN_DYSPLASIA2": [
        "output-XETG00143__0035100__Region_1__20241121__001300",
    ],
}

print("Creating test directory tree...\n")

for slide_name, sample_dirs in slides.items():
    for sample_dir in sample_dirs:
        full_path = TEST_ROOT / slide_name / sample_dir
        full_path.mkdir(parents=True, exist_ok=True)

        # Create placeholder files that mimic Xenium output
        (full_path / "experiment.xenium").touch()
        (full_path / "transcripts.parquet").touch()
        (full_path / "cells.parquet").touch()
        (full_path / "morphology.ome.tif").touch()

        print(f"  {full_path}/")

print(f"\nDone! Tree created at: {TEST_ROOT.resolve()}")
print(f"\nNow run:")
print(f"  python segmentation_pipeline_master.py --config config/test_config.yaml")
