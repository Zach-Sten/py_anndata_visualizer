"""Tests for the region and layout pipelines.

These are the trickiest, most stateful parts of the tool and the foundation the
future mudata/image-alignment work will build on: regions must capture the right
cells and survive a save/load round-trip, and layout must move each sample as a
rigid block so region geometry stays aligned when the layout changes.

Imported through the package (needs the pyav3 env deps). Run with `pytest` or
`python tests/test_region_and_layout.py`.
"""

import base64
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
from py_anndata_visualizer.tools.callback_functions import (
    compute_layout,
    get_sample_meta,
    _reorder_coords_to_js_order,
)
from py_anndata_visualizer.tools.region_functions import (
    save_region_masks,
    load_region_masks,
)

_CHUNKS = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
_SPATIAL = np.array([
    [0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [2.0, 2.0], [10.0, 10.0],
    [-1.0, -1.0], [3.0, 4.0], [4.0, 4.0], [0.0, 3.0], [-4.0, -4.0],
], dtype=float)
_SAMPLE = np.array(["s1"] * 5 + ["s2"] * 5)


def _make_adata():
    n = 10
    X = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "__chunk__": _CHUNKS,
            "sample": _SAMPLE,
            "condition": np.where(_SAMPLE == "s1", "ctrl", "treated"),  # consistent per sample
        },
        index=[f"cell{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = ["GeneA", "GeneB", "GeneC"]
    adata.obsm["spatial"] = _SPATIAL.copy()
    adata.uns["__sample_names__"] = ["s1", "s2"]
    return adata


def _loading_order(chunk_vals):
    n_chunks = int(chunk_vals.max()) + 1
    return np.concatenate([np.sort(np.where(chunk_vals == c)[0]) for c in range(n_chunks)])


def _decode(b64_str, count, compressed):
    raw = base64.b64decode(b64_str)
    if compressed:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype="<f4").reshape(count, 2)


# --------------------------------------------------------------------------
# _reorder_coords_to_js_order
# --------------------------------------------------------------------------

def test_reorder_interleaved_chunks():
    adata = ad.AnnData(X=np.zeros((5, 1), dtype=np.float32))
    adata.obs["__chunk__"] = np.array([1, 0, 1, 0, 2])
    coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
    out = _reorder_coords_to_js_order(coords, adata)
    # chunk 0 cells: idx 1,3 ; chunk 1: idx 0,2 ; chunk 2: idx 4
    assert out.tolist() == [[1, 1], [3, 3], [0, 0], [2, 2], [4, 4]]


def test_reorder_noop_without_chunk_column():
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    coords = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    assert np.array_equal(_reorder_coords_to_js_order(coords, adata), coords)


# --------------------------------------------------------------------------
# compute_layout
# --------------------------------------------------------------------------

def test_layout_is_rigid_per_sample_translation():
    adata = _make_adata()
    resp = compute_layout({"sample_id": "sample"}, adata=adata)
    assert resp["type"] == "layout_coords"

    decoded = _decode(resp["coords_binary"], resp["count"], compressed=True)
    # Undo the JS loading-order reordering to get back to adata order.
    order = _loading_order(_CHUNKS)
    new_coords = np.empty_like(decoded)
    new_coords[order] = decoded

    # Each sample must be a rigid translation: relative positions preserved.
    for s in ("s1", "s2"):
        mask = _SAMPLE == s
        orig_rel = _SPATIAL[mask] - _SPATIAL[mask][0]
        new_rel = new_coords[mask] - new_coords[mask][0]
        assert np.allclose(new_rel, orig_rel, atol=1e-3), f"sample {s} not rigidly translated"


def test_layout_separates_samples():
    adata = _make_adata()
    resp = compute_layout({"sample_id": "sample"}, adata=adata)
    decoded = _decode(resp["coords_binary"], resp["count"], compressed=True)
    order = _loading_order(_CHUNKS)
    new_coords = np.empty_like(decoded)
    new_coords[order] = decoded
    c1 = new_coords[_SAMPLE == "s1"].mean(axis=0)
    c2 = new_coords[_SAMPLE == "s2"].mean(axis=0)
    assert not np.allclose(c1, c2), "samples should be placed at distinct grid positions"


def test_layout_preserves_count_and_labels():
    adata = _make_adata()
    resp = compute_layout({"sample_id": "sample"}, adata=adata)
    assert resp["count"] == 10
    assert sorted(resp["sample_labels"]) == ["s1", "s2"]
    assert len(resp["sample_label_positions"]) == 2


def test_layout_missing_sample_col_is_error():
    adata = _make_adata()
    resp = compute_layout({"sample_id": "not_a_col"}, adata=adata)
    assert resp["type"] == "error"


# --------------------------------------------------------------------------
# get_sample_meta
# --------------------------------------------------------------------------

def test_sample_meta_consistent_column():
    adata = _make_adata()
    resp = get_sample_meta({"column": "condition", "sample_id": "sample"}, adata=adata)
    assert resp["valid"] is True
    assert resp["cats"] == ["ctrl", "treated"]
    assert resp["codes"] == [0, 1]  # s1 -> ctrl(0), s2 -> treated(1)


def test_sample_meta_inconsistent_column_is_invalid():
    adata = _make_adata()
    # "sample" itself varies within... no; make a column that differs within a sample.
    adata.obs["mixed"] = np.array(["a", "b", "a", "a", "a", "x", "x", "x", "x", "x"])
    resp = get_sample_meta({"column": "mixed", "sample_id": "sample"}, adata=adata)
    assert resp["valid"] is False


# --------------------------------------------------------------------------
# region masks: save -> load round-trip + point-in-polygon capture
# --------------------------------------------------------------------------

def _region_payload():
    # Square covering [-2.5, 2.5] in both axes -> encloses cells 0,1,2,3,5.
    square = [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]
    return {
        "groups": {"G1": {"selections": ["R1"], "expanded": True, "visible": True}},
        "regions": {"R1": {"indices": [0], "visible": True, "tool": "region"}},
        "polygons": [{"name": "R1", "polygons": [square], "centroid_x": 0.0, "centroid_y": 0.0}],
        "metadata": {"column": "region", "fill_opacity": 0.1},
    }


def test_region_save_captures_cells_in_polygon():
    adata = _make_adata()
    save_region_masks({"payload": json.dumps(_region_payload()), "embedding": "spatial"}, adata=adata)
    loaded = load_region_masks({"source": "region_masks", "embedding": "spatial"}, adata=adata)
    # Point-in-polygon should capture exactly cells 0,1,2,3,5 (not the seed [0] from payload).
    assert sorted(loaded["regions"]["R1"]["indices"]) == [0, 1, 2, 3, 5]
    assert loaded["regions"]["R1"]["count"] == 5


def test_region_roundtrip_preserves_structure():
    adata = _make_adata()
    save_region_masks({"payload": json.dumps(_region_payload()), "embedding": "spatial"}, adata=adata)
    loaded = load_region_masks({"source": "region_masks", "embedding": "spatial"}, adata=adata)
    assert loaded["type"] == "region_masks_loaded"
    assert "G1" in loaded["groups"]
    assert loaded["groups"]["G1"]["selections"] == ["R1"]
    poly = next(p for p in loaded["polygons"] if p["name"] == "R1")
    assert poly["polygons"] == [[[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]]


def test_region_store_is_json_safe():
    adata = _make_adata()
    save_region_masks({"payload": json.dumps(_region_payload()), "embedding": "spatial"}, adata=adata)
    raw = adata.uns["region_masks"]
    assert isinstance(raw, str)
    json.loads(raw)  # must be valid JSON (h5ad-safe, no numpy/ragged arrays)


def test_region_records_stored_embedding():
    adata = _make_adata()
    save_region_masks({"payload": json.dumps(_region_payload()), "embedding": "spatial"}, adata=adata)
    store = json.loads(adata.uns["region_masks"])
    assert store["metadata"]["embedding"] == "spatial"


def test_region_load_missing_source_is_error():
    adata = _make_adata()
    resp = load_region_masks({"source": "region_masks", "embedding": "spatial"}, adata=adata)
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
