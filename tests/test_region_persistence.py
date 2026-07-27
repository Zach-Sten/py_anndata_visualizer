"""Tests for region/manual-mask persistence and region->obs assignment.

These exercise the functions that write into adata.uns / adata.obs — including
the two where the numpy-int64 -> json.dumps crash was fixed
(recapture_region_cells, save_region_group_to_obs). A save/load round-trip here
is the guard that keeps masks from silently failing again.

Run with `pytest` or `python tests/test_region_persistence.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
from py_anndata_visualizer.tools.region_functions import (
    recapture_region_cells,
    save_manual_masks,
    load_manual_masks,
    save_region_group_to_obs,
)

_SPATIAL = np.array([
    [0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [2.0, 2.0], [10.0, 10.0],
    [-1.0, -1.0], [3.0, 4.0], [4.0, 4.0], [0.0, 3.0], [-4.0, -4.0],
], dtype=float)
# Square covering [-2.5, 2.5]^2 -> encloses cells 0,1,2,3,5.
_SQUARE = [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]
_INSIDE = [0, 1, 2, 3, 5]


def _make_adata():
    n = 10
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = _SPATIAL.copy()
    return adata


# --------------------------------------------------------------------------
# recapture_region_cells  (fixed int64 -> json bug lives here)
# --------------------------------------------------------------------------

def test_recapture_finds_cells_in_polygon():
    adata = _make_adata()
    payload = {"polygons": [{"name": "R", "polygons": [_SQUARE]}], "embedding": "spatial"}
    resp = recapture_region_cells({"payload": json.dumps(payload)}, adata=adata)
    assert resp["type"] == "region_cells_recaptured"
    assert sorted(resp["results"]["R"]["indices"]) == _INSIDE
    assert resp["results"]["R"]["count"] == 5


def test_recapture_updates_json_safe_cache():
    adata = _make_adata()
    payload = {"polygons": [{"name": "R", "polygons": [_SQUARE]}], "embedding": "spatial"}
    recapture_region_cells({"payload": json.dumps(payload)}, adata=adata)
    cache = json.loads(adata.uns["_sel_idx_cache_"])  # must be valid JSON (the int64 bug)
    assert sorted(cache["R"]) == _INSIDE
    assert all(isinstance(i, int) for i in cache["R"])


def test_recapture_empty_polygon_reports_failure():
    adata = _make_adata()
    far = [[100, 100], [101, 100], [101, 101], [100, 101]]
    payload = {"polygons": [{"name": "R", "polygons": [far]}], "embedding": "spatial"}
    resp = recapture_region_cells({"payload": json.dumps(payload)}, adata=adata)
    assert resp["total_recaptured"] == 0
    assert any(f["name"] == "R" for f in resp["failed"])


# --------------------------------------------------------------------------
# manual masks: save -> load round-trip + delta decoding
# --------------------------------------------------------------------------

def test_manual_masks_roundtrip_uncompressed():
    adata = _make_adata()
    payload = {
        "groups": {"G": {"selections": ["S1"], "expanded": True}},
        "selections": {"S1": {"indices": [1, 3, 5], "tool": "lasso"}},
        "embedding": "spatial",
    }
    save_manual_masks({"payload": json.dumps(payload), "embedding": "spatial"}, adata=adata)
    loaded = load_manual_masks({"source": "manual_masks", "embedding": "spatial"}, adata=adata)
    assert loaded["selections"]["S1"]["indices"] == [1, 3, 5]
    assert "G" in loaded["groups"]


def test_manual_masks_delta_decoding():
    adata = _make_adata()
    # deltas: first value is the first index, rest are diffs -> [5, 8, 12, 20]
    payload = {
        "selections": {"S1": {"deltas": "5,3,4,8"}},
        "compressed": True,
        "embedding": "spatial",
    }
    save_manual_masks({"payload": json.dumps(payload), "embedding": "spatial"}, adata=adata)
    store = json.loads(adata.uns["manual_masks"])
    assert store["selections"]["S1"]["indices"] == [5, 8, 12, 20]


def test_manual_masks_store_is_json_safe():
    adata = _make_adata()
    payload = {"selections": {"S1": {"indices": [0, 2, 4]}}, "embedding": "spatial"}
    save_manual_masks({"payload": json.dumps(payload), "embedding": "spatial"}, adata=adata)
    assert isinstance(adata.uns["manual_masks"], str)
    json.loads(adata.uns["manual_masks"])


# --------------------------------------------------------------------------
# save_region_group_to_obs  (fixed int64 -> json bug lives here)
# --------------------------------------------------------------------------

def test_region_group_writes_categorical_obs():
    adata = _make_adata()
    resp = save_region_group_to_obs(
        {
            "group_name": "myregions",
            "region_names": ["R"],
            "polygons": json.dumps([{"name": "R", "polygons": [_SQUARE]}]),
            "embedding": "spatial",
        },
        adata=adata,
    )
    assert resp["type"] == "region_group_obs_saved"
    assert resp["labeled_cells"] == 5

    col = adata.obs["myregions"]
    assert isinstance(col.dtype, pd.CategoricalDtype)
    labeled = [i for i, v in enumerate(col) if v == "R"]
    assert labeled == _INSIDE
    # the rest are unlabeled
    assert col.notna().sum() == 5


def test_region_group_missing_args_errors():
    adata = _make_adata()
    assert save_region_group_to_obs({"region_names": ["R"]}, adata=adata)["type"] == "error"
    assert save_region_group_to_obs({"group_name": "g"}, adata=adata)["type"] == "error"


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
