"""Tests for the data-movement primitives every callback response flows through.

`tools/utils.py` is the bridge between Python (numpy/pandas/scipy) and the
browser: `_serialize_result` turns arbitrary callback results into a
JSON-compatible structure, `_pack_coords_binary` is the binary coordinate
channel, and `_b64` encodes HTML payloads. A bug here silently corrupts every
plot, so these tests pin the contracts hard.

The module is loaded directly from its file so the test only needs
numpy/pandas/scipy, not the ipywidgets/IPython stack the package __init__ pulls
in. Run with `pytest` or plain `python tests/test_serialization.py`.
"""

import base64
import importlib.util
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "py_anndata_visualizer" / "tools" / "utils.py"
)
_spec = importlib.util.spec_from_file_location("pav_utils", _MODULE_PATH)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)

_pack_coords_binary = utils._pack_coords_binary
_b64 = utils._b64
_serialize_result = utils._serialize_result


def _decode_coords(b64_str, count, compressed=False):
    """Mirror the JS-side decode: base64 -> (unzip) -> float32 -> (N, 2)."""
    raw = base64.b64decode(b64_str)
    if compressed:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype="<f4").reshape(count, 2)


def _is_json_ready(obj):
    """True if obj serializes to strict JSON the browser's JSON.parse accepts.

    allow_nan=False makes json.dumps raise on NaN/Inf, which JSON.parse rejects.
    """
    json.dumps(obj, allow_nan=False)
    return True


# --------------------------------------------------------------------------
# _b64
# --------------------------------------------------------------------------

def test_b64_roundtrip_ascii():
    s = "<html>hello</html>"
    assert base64.b64decode(_b64(s)).decode("utf-8") == s


def test_b64_roundtrip_unicode_and_empty():
    for s in ["café — ünïcode ✓", ""]:
        assert base64.b64decode(_b64(s)).decode("utf-8") == s


# --------------------------------------------------------------------------
# _pack_coords_binary
# --------------------------------------------------------------------------

def test_pack_coords_roundtrip_uncompressed():
    coords = np.array([[1.5, 2.5], [-3.25, 4.0], [0.0, 100.0]], dtype=np.float32)
    packed = _pack_coords_binary(coords, compress=False)
    back = _decode_coords(packed, count=3, compressed=False)
    assert np.array_equal(back, coords)


def test_pack_coords_roundtrip_compressed():
    coords = np.random.default_rng(0).uniform(-1000, 1000, size=(500, 2)).astype(np.float32)
    packed = _pack_coords_binary(coords, compress=True)
    back = _decode_coords(packed, count=500, compressed=True)
    assert np.array_equal(back, coords)


def test_pack_coords_downcasts_float64_to_float32():
    coords = np.array([[1.123456789, 2.0]], dtype=np.float64)
    back = _decode_coords(_pack_coords_binary(coords), count=1)
    assert back.dtype == np.float32
    assert np.allclose(back, coords.astype(np.float32))


def test_pack_coords_empty():
    coords = np.zeros((0, 2), dtype=np.float32)
    back = _decode_coords(_pack_coords_binary(coords), count=0)
    assert back.shape == (0, 2)


# --------------------------------------------------------------------------
# _serialize_result — type coverage
# --------------------------------------------------------------------------

def test_serialize_primitives_pass_through():
    for v in ["x", 1, 2.5, True, False, None]:
        assert _serialize_result(v) == v


def test_serialize_numpy_int_and_float():
    assert _serialize_result(np.int64(7)) == 7
    assert isinstance(_serialize_result(np.int64(7)), int)
    assert _serialize_result(np.float32(2.5)) == 2.5
    assert isinstance(_serialize_result(np.float64(2.5)), float)


def test_serialize_numpy_bool_becomes_python_bool():
    out = _serialize_result(np.bool_(True))
    assert out is True, f"numpy bool should serialize to a real bool, got {out!r}"


def test_serialize_ndarray_1d_and_2d():
    assert _serialize_result(np.array([1, 2, 3])) == [1, 2, 3]
    assert _serialize_result(np.array([[1.0, 2.0], [3.0, 4.0]])) == [[1.0, 2.0], [3.0, 4.0]]


def test_serialize_nan_and_inf_become_none():
    # NaN/Inf are invalid JSON and crash browser JSON.parse; expect null.
    assert _serialize_result(float("nan")) is None
    assert _serialize_result(np.float64("inf")) is None
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    assert _serialize_result(arr) == [1.0, None, None, None, 2.0]


def test_serialize_series():
    assert _serialize_result(pd.Series([1, 2, 3])) == [1, 2, 3]


def test_serialize_dataframe_is_json_safe():
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    out = _serialize_result(df)
    assert out == [{"a": 1, "b": 0.5}, {"a": 2, "b": 1.5}]
    assert _is_json_ready(out)  # must not contain numpy scalars


def test_serialize_sparse_matrix():
    m = sp.csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]]))
    assert _serialize_result(m) == [[0.0, 1.0], [2.0, 0.0]]


def test_serialize_nested_structure():
    data = {"a": np.array([1, 2]), "b": [np.int64(3), {"c": np.float64(4.5)}]}
    assert _serialize_result(data) == {"a": [1, 2], "b": [3, {"c": 4.5}]}


def test_serialize_numpy_dict_key_is_json_safe():
    out = _serialize_result({np.int64(5): "v"})
    assert _is_json_ready(out)


def test_serialize_unknown_object_falls_back_to_str():
    class Widget:
        def __repr__(self):
            return "<Widget>"
    assert _serialize_result(Widget()) == "<Widget>"


def test_serialize_output_is_always_json_ready():
    payload = {
        "type": "chunk_data",
        "count": np.int64(3),
        "values": np.array([1.0, np.nan, 3.0]),
        "flag": np.bool_(False),
        "rows": pd.DataFrame({"x": [1, 2]}),
        "nested": [np.float64("inf"), {"k": np.int32(9)}],
    }
    assert _is_json_ready(_serialize_result(payload))


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
