"""Tests for the DBSCAN streaming protocol, alpha-shape polygons, and the
moving-layout polygon transform.

These are the most complex region pieces:
  * run_dbscan streams one sample per call, caching cluster indices in
    adata.uns['_dbscan_tmp'] across calls;
  * compute_alpha_shapes turns those cached clusters into polygons;
  * _transform_polygons_to_embedding translates a region's polygon to follow its
    cells when the layout/embedding changes — the alignment primitive the future
    mudata/image work depends on.

Imported through the package (needs pyav3 deps). Run with `pytest` or
`python tests/test_dbscan_alphashape.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
from py_anndata_visualizer.tools.region_functions import (
    run_dbscan,
    compute_alpha_shapes,
    _transform_polygons_to_embedding,
    _detect_stored_embedding,
)


def _blob(cx, cy):
    """9 points on a 3x3 grid (spacing 1) centered at (cx, cy)."""
    return [[cx + dx, cy + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


def _make_adata():
    # Two samples, each with two well-separated 9-point blobs.
    s1 = _blob(0, 0) + _blob(100, 100)
    s2 = _blob(0, 0) + _blob(100, 100)
    spatial = np.array(s1 + s2, dtype=float)
    n = len(spatial)  # 36
    sample = np.array(["s1"] * 18 + ["s2"] * 18)
    obs = pd.DataFrame({"sample": sample}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = spatial
    return adata


# --------------------------------------------------------------------------
# run_dbscan — streaming continuation protocol
# --------------------------------------------------------------------------

def test_dbscan_streams_one_sample_per_call():
    adata = _make_adata()
    params = {"column": "all_cells", "eps": 2, "min_samples": 3, "sample_id": "sample"}

    first = run_dbscan(dict(params), adata=adata)
    assert first["type"] == "dbscan_progress"
    assert first["total_samples"] == 2
    assert first["sample_idx"] == 1
    assert len(first["clusters_this_sample"]) == 2  # two blobs in sample 1

    second = run_dbscan({"continue": True}, adata=adata)
    assert second["type"] == "dbscan_result"
    assert second["total_clusters"] == 4  # 2 samples x 2 blobs


def test_dbscan_caches_json_safe_indices():
    adata = _make_adata()
    run_dbscan({"column": "all_cells", "eps": 2, "min_samples": 3, "sample_id": "sample"}, adata=adata)
    run_dbscan({"continue": True}, adata=adata)

    raw = adata.uns["_dbscan_tmp"]
    cached = json.loads(raw)  # must be valid JSON (no numpy scalars)
    assert len(cached) == 4
    for name, idx in cached.items():
        assert all(isinstance(i, int) for i in idx), f"{name} has non-int indices"
        assert len(idx) == 9  # each blob has 9 cells


def test_dbscan_cluster_centroids_match_blobs():
    adata = _make_adata()
    run_dbscan({"column": "all_cells", "eps": 2, "min_samples": 3, "sample_id": "sample"}, adata=adata)
    result = run_dbscan({"continue": True}, adata=adata)
    centroids = {(round(c["centroid_x"]), round(c["centroid_y"])) for c in result["clusters"]}
    assert centroids == {(0, 0), (100, 100)}


def test_dbscan_missing_column_is_error():
    adata = _make_adata()
    resp = run_dbscan({"column": "nope", "category": "x"}, adata=adata)
    assert resp["type"] == "error"


# --------------------------------------------------------------------------
# compute_alpha_shapes
# --------------------------------------------------------------------------

def test_alpha_shapes_produce_polygons_for_clusters():
    adata = _make_adata()
    run_dbscan({"column": "all_cells", "eps": 2, "min_samples": 3, "sample_id": "sample"}, adata=adata)
    result = run_dbscan({"continue": True}, adata=adata)
    cluster_names = [{"name": c["name"]} for c in result["clusters"]]

    resp = compute_alpha_shapes({"clusters": cluster_names, "alpha": 0.0}, adata=adata)  # alpha 0 = convex hull
    assert resp["type"] == "alpha_shapes"
    assert resp["total_regions"] == 4
    for region in resp["regions"]:
        assert len(region["polygons"]) >= 1
        assert len(region["polygons"][0]) >= 4  # hull of a 3x3 grid
        # response must be JSON-safe
        json.dumps(region)


def test_alpha_shapes_missing_cache_is_error():
    adata = _make_adata()  # no DBSCAN run -> no _dbscan_tmp
    resp = compute_alpha_shapes({"clusters": [{"name": "x"}], "alpha": 0.0}, adata=adata)
    assert resp["type"] == "error"


# --------------------------------------------------------------------------
# moving-layout polygon transform
# --------------------------------------------------------------------------

def _transform_fixture():
    adata = _make_adata()
    # A second embedding shifted by a known offset; polygon should follow.
    adata.obsm["X_layout"] = adata.obsm["spatial"] + np.array([50.0, -20.0])
    # Region = the first blob (cells 0..8), polygon a triangle in spatial space.
    full_indices = {"R": list(range(9))}
    polys = [{
        "name": "R",
        "polygons": [[[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]],
        "centroid_x": 0.0, "centroid_y": 0.0,
    }]
    return adata, polys, full_indices


def test_transform_translates_polygon_by_cell_offset():
    adata, polys, full_indices = _transform_fixture()
    out = _transform_polygons_to_embedding(
        polys, full_indices, adata, source_embedding="spatial", target_embedding="X_layout"
    )
    ring = out[0]["polygons"][0]
    expected = [[-1 + 50, -1 - 20], [1 + 50, -1 - 20], [0 + 50, 1 - 20]]
    assert np.allclose(ring, expected)
    # centroid updated to the target-space cell centroid (blob0 center + offset).
    assert np.allclose([out[0]["centroid_x"], out[0]["centroid_y"]], [50.0, -20.0])


def test_transform_preserves_shape_rigidly():
    adata, polys, full_indices = _transform_fixture()
    out = _transform_polygons_to_embedding(
        polys, full_indices, adata, source_embedding="spatial", target_embedding="X_layout"
    )
    orig = np.array(polys[0]["polygons"][0])
    moved = np.array(out[0]["polygons"][0])
    # edge vectors identical => pure translation, no scale/rotation.
    assert np.allclose(np.diff(orig, axis=0), np.diff(moved, axis=0))


def test_transform_noop_when_source_equals_target():
    adata, polys, full_indices = _transform_fixture()
    out = _transform_polygons_to_embedding(
        polys, full_indices, adata, source_embedding="spatial", target_embedding="spatial"
    )
    assert out is polys  # returned unchanged


def test_detect_stored_embedding_picks_matching_space():
    adata, polys, full_indices = _transform_fixture()
    # polys are in spatial space; detector should identify "spatial" over X_layout.
    assert _detect_stored_embedding(polys, full_indices, adata) == "spatial"


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
