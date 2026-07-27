"""Tests for the point-in-quad primitive used by the spatial heatmap binning.

`_points_in_quad` was reimplemented on top of matplotlib's vectorized
point-in-polygon test (replacing a hand-rolled cross-product winding test that
also carried dead code). These tests pin its behavior:

  * known geometric cases (square + sheared quad), and
  * equivalence to the previous cross-product implementation on random points,
    excluding points near the boundary where fill conventions legitimately differ.

The module is loaded directly from its file so the test only needs numpy +
matplotlib, not the full ipywidgets/IPython stack the package __init__ pulls in.
Run with either `pytest` or plain `python tests/test_heatmap_geometry.py`.
"""

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "py_anndata_visualizer" / "tools" / "heatmap_functions.py"
)
_spec = importlib.util.spec_from_file_location("heatmap_functions", _MODULE_PATH)
heatmap_functions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heatmap_functions)
_points_in_quad = heatmap_functions._points_in_quad


def _reference_point_in_quad(points, quad):
    """The previous cross-product implementation, kept here as the oracle."""
    n = len(quad)
    crosses = []
    for i in range(n):
        v1 = quad[(i + 1) % n] - quad[i]
        v2 = points - quad[i]
        crosses.append(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
    crosses = np.array(crosses)
    return np.all(crosses >= 0, axis=0) | np.all(crosses <= 0, axis=0)


def _min_dist_to_edges(points, quad):
    """Min distance from each point to the quad's edges (to skip boundary cases)."""
    n = len(quad)
    dists = np.full(len(points), np.inf)
    for i in range(n):
        a = quad[i]
        b = quad[(i + 1) % n]
        ab = b - a
        length2 = float(ab @ ab)
        if length2 == 0.0:
            seg = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip((points - a) @ ab / length2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            seg = np.linalg.norm(points - proj, axis=1)
        dists = np.minimum(dists, seg)
    return dists


def test_unit_square_known_points():
    quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.array([
        [0.5, 0.5],   # center — inside
        [0.1, 0.9],   # inside
        [2.0, 2.0],   # outside
        [-0.5, 0.5],  # outside
        [0.5, 1.5],   # outside (above)
    ])
    result = _points_in_quad(pts, quad)
    assert result.tolist() == [True, True, False, False, False]


def test_sheared_quad_known_points():
    # A non-axis-aligned convex quad.
    quad = np.array([[0.0, 0.0], [2.0, 1.0], [3.0, 3.0], [1.0, 2.0]])
    pts = np.array([
        [1.5, 1.5],   # interior
        [0.05, 0.05],  # near a vertex but just inside
        [5.0, 5.0],   # far outside
        [0.0, 3.0],   # outside
    ])
    result = _points_in_quad(pts, quad)
    assert result[0]        # clearly inside
    assert not result[2]    # clearly outside
    assert not result[3]


def test_matches_reference_away_from_boundary():
    rng = np.random.default_rng(20260727)
    for _ in range(50):
        # Convex quad by construction: 4 vertices on an ellipse in angular order.
        # (The old cross-product oracle is only valid for convex polygons, which is
        # also the documented contract of _points_in_quad.)
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=4))
        r = rng.uniform(1.0, 5.0)
        sx, sy = rng.uniform(0.5, 2.0, size=2)
        center = rng.uniform(-10, 10, size=2)
        quad = center + np.stack([np.cos(angles) * sx, np.sin(angles) * sy], axis=1) * r

        pts = center + rng.uniform(-8, 8, size=(400, 2))
        # Skip points hugging an edge, where boundary fill conventions differ.
        keep = _min_dist_to_edges(pts, quad) > 1e-6
        pts = pts[keep]

        new = _points_in_quad(pts, quad)
        ref = _reference_point_in_quad(pts, quad)
        assert np.array_equal(new, ref), "point-in-quad diverged from reference"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
