"""Tests for the chunk/viewport selection and embedding-key resolution.

These are the correctness core of the data loader: given an AnnData, they decide
*which* cells' coordinates get sent and in *what order*. The critical invariant
is alignment — the k-th coordinate in a returned binary blob must belong to the
k-th index in the returned `indices` list. A mismatch there silently plots cells
at the wrong positions.

Imported through the package (needs the pyav3 env deps). Run with `pytest` or
`python tests/test_chunk_and_embedding.py`.
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
    get_chunk_cells,
    get_viewport_cells,
)
from py_anndata_visualizer.tools.region_functions import _get_embedding_coords
from py_anndata_visualizer.tools.heatmap_functions import _get_coordinates


# Known 10-cell layout; chunk assignments split 0/1/2.
_CHUNKS = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
_SPATIAL = np.array([
    [0.0, 0.0],    # 0  chunk0
    [1.0, 1.0],    # 1  chunk0
    [0.5, 0.5],    # 2  chunk1   dist 0.71  -> in circle
    [2.0, 2.0],    # 3  chunk1   dist 2.83  -> in circle
    [10.0, 10.0],  # 4  chunk1   dist 14.1  -> out
    [-1.0, -1.0],  # 5  chunk2   dist 1.41  -> in circle
    [3.0, 4.0],    # 6  chunk2   dist 5.00  -> on radius (<=) -> in
    [4.0, 4.0],    # 7  chunk2   dist 5.66  -> out
    [0.0, 3.0],    # 8  chunk2   dist 3.00  -> in circle
    [-4.0, -4.0],  # 9  chunk2   dist 5.66  -> out
], dtype=float)


def _make_adata():
    n = 10
    rng = np.random.default_rng(1)
    X = rng.random((n, 3)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "__chunk__": _CHUNKS,
            "cell_type": pd.Categorical(list("ABAABB ABAB".replace(" ", ""))),
            "score": np.arange(n, dtype=float),
        },
        index=[f"cell{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = ["GeneA", "GeneB", "GeneC"]
    adata.obsm["spatial"] = _SPATIAL.copy()
    adata.obsm["X_umap"] = rng.random((n, 2))
    adata.obsm["X_pca"] = rng.random((n, 2))
    return adata


def _decode(b64_str, count, compressed):
    raw = base64.b64decode(b64_str)
    if compressed:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype="<f4").reshape(count, 2)


# --------------------------------------------------------------------------
# get_chunk_cells
# --------------------------------------------------------------------------

def test_chunk_selects_correct_indices():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 1}, adata=adata)
    assert resp["type"] == "chunk_data"
    assert resp["indices"] == [2, 3, 4]
    assert resp["count"] == 3


def test_chunk_binary_is_aligned_with_indices():
    # The core invariant: decoded coord k belongs to cell indices[k].
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 2}, adata=adata)
    idx = resp["indices"]
    coords = _decode(resp["spatial_binary"], count=resp["count"], compressed=True)
    expected = _SPATIAL[idx].astype(np.float32)
    assert np.array_equal(coords, expected)


def test_chunk_all_embeddings_present_and_aligned():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 1}, adata=adata)
    idx = resp["indices"]
    for key, obsm_key in [("umap_binary", "X_umap"), ("pca_binary", "X_pca")]:
        coords = _decode(resp[key], count=resp["count"], compressed=True)
        expected = np.asarray(adata.obsm[obsm_key])[idx, :2].astype(np.float32)
        assert np.array_equal(coords, expected), f"{key} misaligned"


def test_chunk_echoes_request_id():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 1, "requestId": 42}, adata=adata)
    assert resp["requestId"] == 42


def test_chunk_empty_chunk_is_error():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 99}, adata=adata)
    assert resp["type"] == "error"


def test_chunk_missing_assignment_is_error():
    adata = _make_adata()
    del adata.obs["__chunk__"]
    resp = get_chunk_cells({"chunk": 1}, adata=adata)
    assert resp["type"] == "error"


def test_chunk_gene_expression_matches():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 1, "activeGene": "GeneB"}, adata=adata)
    idx = resp["indices"]
    expected = adata.X[idx, 1].tolist()
    assert np.allclose(resp["gex_values"], expected)
    assert resp["gex_gene"] == "GeneB"


def test_chunk_categorical_obs_matches():
    adata = _make_adata()
    resp = get_chunk_cells({"chunk": 1, "activeColumn": "cell_type"}, adata=adata)
    assert resp["obs_mode"] == "categorical"
    assert resp["obs_categories"] == ["A", "B"]
    # cells 2,3,4 are A,A,B -> codes 0,0,1
    assert resp["obs_values"] == [0, 0, 1]


# --------------------------------------------------------------------------
# get_viewport_cells
# --------------------------------------------------------------------------

def test_viewport_selects_circle_and_excludes_chunk0():
    adata = _make_adata()
    resp = get_viewport_cells(
        {"embedding": "spatial", "viewMinX": -5, "viewMaxX": 5,
         "viewMinY": -5, "viewMaxY": 5},
        adata=adata,
    )
    assert resp["type"] == "viewport_cells"
    # In circle (dist<=5) and not chunk 0: indices 2,3,5,6,8.
    assert resp["indices"] == [2, 3, 5, 6, 8]
    assert resp["chunks"] == [1, 1, 2, 2, 2]


def test_viewport_binary_aligned():
    adata = _make_adata()
    resp = get_viewport_cells(
        {"embedding": "spatial", "viewMinX": -5, "viewMaxX": 5,
         "viewMinY": -5, "viewMaxY": 5},
        adata=adata,
    )
    idx = resp["indices"]
    coords = _decode(resp["coords_binary"], count=resp["coords_count"], compressed=False)
    assert np.array_equal(coords, _SPATIAL[idx].astype(np.float32))


def test_viewport_invalid_bounds_is_error():
    adata = _make_adata()
    resp = get_viewport_cells({"embedding": "spatial"}, adata=adata)
    assert resp["type"] == "error"


# --------------------------------------------------------------------------
# embedding-key resolution
# --------------------------------------------------------------------------

def test_region_embedding_key_as_is():
    adata = _make_adata()
    coords = _get_embedding_coords(adata, "spatial")
    assert np.array_equal(coords, _SPATIAL[:, :2])


def test_region_embedding_x_prefix_fallback():
    adata = _make_adata()
    adata.obsm["X_layout"] = np.arange(20, dtype=float).reshape(10, 2)
    coords = _get_embedding_coords(adata, "layout")  # resolves to X_layout
    assert np.array_equal(coords, adata.obsm["X_layout"][:, :2])


def test_region_embedding_unknown_falls_back_to_spatial():
    adata = _make_adata()
    coords = _get_embedding_coords(adata, "does_not_exist")
    assert np.array_equal(coords, _SPATIAL[:, :2])


def test_region_embedding_returns_only_two_columns():
    adata = _make_adata()
    adata.obsm["threeD"] = np.arange(30, dtype=float).reshape(10, 3)
    coords = _get_embedding_coords(adata, "threeD")
    assert coords.shape == (10, 2)


def test_region_embedding_none_when_absent():
    adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    assert _get_embedding_coords(adata, "spatial") is None


def test_heatmap_coordinates_priority_fallback():
    adata = _make_adata()
    del adata.obsm["spatial"]  # next priority is X_umap
    coords = _get_coordinates(adata)
    assert np.array_equal(coords, np.asarray(adata.obsm["X_umap"])[:, :2])


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
