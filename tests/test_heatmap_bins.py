"""Smoke test for compute_heatmap_bins (the full ribbon-binning function).

The point-in-quad primitive is covered in test_heatmap_geometry.py; this checks
the end-to-end binning: a straight ribbon with cells at known positions should
place one cell per bin and average expression correctly.

Run with `pytest` or `python tests/test_heatmap_bins.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
from py_anndata_visualizer.tools.heatmap_functions import compute_heatmap_bins


def _make_adata():
    # 5 cells along y=0 at x = 1,3,5,7,9; one gene with expression 1..5.
    spatial = np.array([[x, 0.0] for x in (1, 3, 5, 7, 9)], dtype=float)
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    adata.var_names = ["G"]
    adata.obsm["spatial"] = spatial
    return adata


def _straight_ribbon():
    # Straight line (0,0)->(10,0) with control points on the line => t maps linearly to x.
    return {
        "start": {"x": 0, "y": 0},
        "end": {"x": 10, "y": 0},
        "controlPoints": [{"x": 10 / 3, "y": 0}, {"x": 20 / 3, "y": 0}],
        "widthStart": 4, "widthMid": 4, "widthEnd": 4,
    }


def test_heatmap_bins_one_cell_per_bin():
    adata = _make_adata()
    payload = {"ribbon": _straight_ribbon(), "genes": ["G"], "numBins": 5, "embedding": "spatial"}
    resp = compute_heatmap_bins({"payload": json.dumps(payload)}, adata=adata)
    assert resp["type"] == "heatmap_result"
    assert resp["numBins"] == 5
    assert resp["totalCells"] == 5
    # Even bins along x -> one cell each -> mean expression is just that cell's value.
    assert np.allclose(resp["heatmap"]["G"], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert [b["cell_count"] for b in resp["bins"]] == [1, 1, 1, 1, 1]


def test_heatmap_missing_ribbon_errors():
    adata = _make_adata()
    resp = compute_heatmap_bins({"payload": json.dumps({"genes": ["G"]})}, adata=adata)
    assert resp["type"] == "error"


def test_heatmap_missing_genes_errors():
    adata = _make_adata()
    resp = compute_heatmap_bins({"payload": json.dumps({"ribbon": _straight_ribbon()})}, adata=adata)
    assert resp["type"] == "error"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
