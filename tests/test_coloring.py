"""Tests for the coloring hot path: obs column codes and gene expression.

Every recolor flows through get_obs_column / get_gene_expression, which quantize
to uint8, zlib-compress, and base64-encode. These tests decode the payload and
verify the quantization and category mapping are correct.

Run with `pytest` or `python tests/test_coloring.py`.
"""

import base64
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
from py_anndata_visualizer.tools.callback_functions import (
    get_obs_column,
    get_gene_expression,
)


def _decode_codes(b64):
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=np.uint8)


def _make_adata():
    X = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)  # one gene "G"
    obs = pd.DataFrame(
        {
            "ct": pd.Categorical(["A", "B", "A", "C"]),
            "score": [0.0, 5.0, 10.0, 5.0],
        },
        index=[f"c{i}" for i in range(4)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = ["G"]
    return adata


# --------------------------------------------------------------------------
# get_gene_expression
# --------------------------------------------------------------------------

def test_gene_expression_quantization():
    adata = _make_adata()
    resp = get_gene_expression({"gene": "G"}, adata=adata)
    assert resp["type"] == "gex_values"
    assert resp["vmax"] == 4.0
    codes = _decode_codes(resp["values_b64"])
    # v/vmax*254+1 for v>0; 0 stays 0. Max value -> 255.
    assert codes.tolist() == [0, 64, 128, 255]
    assert resp["count"] == 4


def test_gene_expression_unknown_gene_errors():
    adata = _make_adata()
    resp = get_gene_expression({"gene": "NOPE"}, adata=adata)
    assert resp["type"] == "error"


def test_gene_expression_empty_name_errors():
    adata = _make_adata()
    assert get_gene_expression({"gene": "  "}, adata=adata)["type"] == "error"


# --------------------------------------------------------------------------
# get_obs_column
# --------------------------------------------------------------------------

def test_obs_categorical_codes_and_categories():
    adata = _make_adata()
    resp = get_obs_column({"column": "ct"}, adata=adata)
    assert resp["mode"] == "categorical"
    assert resp["categories"] == ["A", "B", "C"]
    codes = _decode_codes(resp["codes_b64"])
    # 1-indexed category codes: A=1, B=2, C=3 -> [1,2,1,3]
    assert codes.tolist() == [1, 2, 1, 3]


def test_obs_numeric_is_continuous():
    adata = _make_adata()
    resp = get_obs_column({"column": "score"}, adata=adata)
    assert resp["mode"] == "continuous"
    codes = _decode_codes(resp["codes_b64"])
    # (v-vmin)/(vmax-vmin)*254+1: 0->1, 5->128, 10->255, 5->128
    assert codes.tolist() == [1, 128, 255, 128]


def test_obs_palette_from_uns():
    adata = _make_adata()
    adata.uns["ct_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    resp = get_obs_column({"column": "ct"}, adata=adata)
    assert resp["colors"] == ["#ff0000", "#00ff00", "#0000ff"]


def test_obs_missing_column_errors():
    adata = _make_adata()
    resp = get_obs_column({"column": "nope"}, adata=adata)
    assert resp["type"] == "error"
    assert "available_columns" in resp


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
