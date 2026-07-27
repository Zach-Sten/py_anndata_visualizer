"""Tests for layout persistence: save_layout / load_layout / delete_layout.

Layouts are saved into adata.obsm['X_<name>'] and reloaded as binary coords.
The new save format reconstructs full cell positions from per-sample centroids
by a rigid per-sample offset — the same alignment property regions rely on.

Run with `pytest` or `python tests/test_layout_persistence.py`.
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
    save_layout,
    load_layout,
    delete_layout,
)

_CHUNKS = np.array([0, 0, 1, 1, 2, 2])
_SAMPLE = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
_SPATIAL = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # s1
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # s2 (same shape, will be moved apart)
], dtype=np.float32)


def _make_adata():
    obs = pd.DataFrame(
        {"sample": _SAMPLE, "__chunk__": _CHUNKS},
        index=[f"c{i}" for i in range(6)],
    )
    adata = ad.AnnData(X=np.zeros((6, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = _SPATIAL.copy()
    return adata


def _encode(arr):
    return base64.b64encode(zlib.compress(arr.astype(np.float32).tobytes())).decode("ascii")


def _decode(b64, count):
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype="<f4").reshape(count, 2)


def _loading_order(chunk_vals):
    n = int(chunk_vals.max()) + 1
    return np.concatenate([np.sort(np.where(chunk_vals == c)[0]) for c in range(n)])


# --------------------------------------------------------------------------
# legacy format (full coords) + load round-trip
# --------------------------------------------------------------------------

def test_save_legacy_coords_and_load_roundtrip():
    adata = _make_adata()
    coords = np.arange(12, dtype=np.float32).reshape(6, 2)
    save_layout({"name": "test", "coords_b64": _encode(coords)}, adata=adata)
    assert "X_test" in adata.obsm
    assert np.array_equal(adata.obsm["X_test"], coords)

    loaded = load_layout({"name": "test", "sample_id": "sample"}, adata=adata)
    decoded = _decode(loaded["coords_binary"], loaded["count"])
    # load reorders to JS loading order; invert it to compare with saved coords.
    order = _loading_order(_CHUNKS)
    back = np.empty_like(decoded)
    back[order] = decoded
    assert np.allclose(back, coords)


def test_save_legacy_count_mismatch_errors():
    adata = _make_adata()
    coords = np.zeros((5, 2), dtype=np.float32)  # 5 != 6 cells
    resp = save_layout({"name": "bad", "coords_b64": _encode(coords)}, adata=adata)
    assert resp["type"] == "error"


# --------------------------------------------------------------------------
# new format: per-sample centroid reconstruction (rigid offset)
# --------------------------------------------------------------------------

def test_save_new_format_rigid_centroid_reconstruction():
    adata = _make_adata()
    # Target grid centroids for the two samples.
    targets = {"s1": np.array([100.0, 0.0]), "s2": np.array([0.0, 100.0])}
    centroids = np.array([targets["s1"], targets["s2"]], dtype=np.float32)
    save_layout(
        {"name": "grid", "centroids_b64": _encode(centroids),
         "sample_labels": ["s1", "s2"], "sample_id": "sample"},
        adata=adata,
    )
    layout = np.asarray(adata.obsm["X_grid"])
    for s in ("s1", "s2"):
        mask = _SAMPLE == s
        # each sample's cells recentered on its target centroid...
        assert np.allclose(layout[mask].mean(axis=0), targets[s], atol=1e-4)
        # ...as a rigid translation (relative geometry preserved).
        orig_rel = _SPATIAL[mask] - _SPATIAL[mask][0]
        new_rel = layout[mask] - layout[mask][0]
        assert np.allclose(new_rel, orig_rel, atol=1e-4)


# --------------------------------------------------------------------------
# delete
# --------------------------------------------------------------------------

def test_delete_layout_removes_key():
    adata = _make_adata()
    coords = np.zeros((6, 2), dtype=np.float32)
    save_layout({"name": "temp", "coords_b64": _encode(coords)}, adata=adata)
    assert "X_temp" in adata.obsm
    resp = delete_layout({"name": "temp"}, adata=adata)
    assert resp["type"] == "layout_deleted"
    assert "X_temp" not in adata.obsm


def test_load_missing_layout_errors():
    adata = _make_adata()
    assert load_layout({"name": "ghost"}, adata=adata)["type"] == "error"
    assert delete_layout({"name": "ghost"}, adata=adata)["type"] == "error"


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
