"""
Utility functions for data serialization and binary encoding.
"""

import base64
import math
import zlib
from typing import Any

import numpy as np
import pandas as pd


def _pack_coords_binary(coords_array: np.ndarray, compress: bool = False) -> str:
    """Pack Nx2 float32 coordinates as base64, optionally with zlib compression.
    
    Args:
        coords_array: Nx2 array of coordinates
        compress: Whether to apply zlib compression (level 6)
        
    Returns:
        Base64-encoded string of the binary data
    """
    raw_bytes = coords_array.astype(np.float32).tobytes()
    if compress:
        raw_bytes = zlib.compress(raw_bytes, level=6)  # level 6 is good balance of speed/size
    return base64.b64encode(raw_bytes).decode('ascii')


def _b64(s: str) -> str:
    """Base64 encode a string.
    
    Args:
        s: String to encode
        
    Returns:
        Base64-encoded string
    """
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _json_safe_float(x: float) -> Any:
    """Return x, or None if it is NaN/Inf.

    NaN and Infinity are not valid JSON and crash the browser's JSON.parse,
    so non-finite floats are represented as null (the JSON convention for
    "missing/undefined numeric value").
    """
    return x if math.isfinite(x) else None


def _coerce_key(k: Any) -> Any:
    """Coerce a dict key to a JSON-serializable scalar.

    json.dumps requires keys to be str/int/float/bool/None — a numpy scalar
    key (e.g. an np.int64 category id) raises TypeError otherwise.
    """
    if isinstance(k, np.bool_):
        return bool(k)
    if isinstance(k, np.integer):
        return int(k)
    if isinstance(k, np.floating):
        return _json_safe_float(float(k))
    if isinstance(k, (str, int, float, bool, type(None))):
        return k
    return str(k)


def _serialize_result(result: Any) -> Any:
    """Safely serialize callback results to a JSON-compatible structure.

    Converts numpy arrays, numpy scalars, pandas objects, and sparse matrices
    to native Python equivalents, and guarantees the output can be json.dumps'd
    and parsed by the browser: non-finite floats (NaN/Inf) become None, numpy
    booleans become real bools, and numpy scalar dict keys are coerced.

    Args:
        result: Any Python object to serialize

    Returns:
        JSON-compatible dictionary/list/primitive
    """
    if isinstance(result, dict):
        return {_coerce_key(k): _serialize_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_serialize_result(item) for item in result]
    if isinstance(result, np.ndarray):
        if result.dtype.kind == "f":
            # Fast path when clean; only pay the object conversion if NaN/Inf present.
            if np.isfinite(result).all():
                return result.tolist()
            out = result.astype(object)
            out[~np.isfinite(result)] = None
            return out.tolist()
        return result.tolist()
    if isinstance(result, np.bool_):
        return bool(result)
    if isinstance(result, np.integer):
        return int(result)
    if isinstance(result, np.floating):
        return _json_safe_float(float(result))
    if isinstance(result, pd.Series):
        return _serialize_result(result.to_numpy())
    if isinstance(result, pd.DataFrame):
        return _serialize_result(result.to_dict("records"))
    if hasattr(result, "toarray"):
        # Sparse matrix
        return _serialize_result(result.toarray())
    if isinstance(result, bool):
        return result
    if isinstance(result, float):
        return _json_safe_float(result)
    if isinstance(result, (str, int, type(None))):
        return result
    return str(result)
