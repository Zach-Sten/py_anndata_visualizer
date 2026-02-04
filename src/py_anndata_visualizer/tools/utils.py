"""
Utility functions for data serialization and binary encoding.
"""

import base64
import zlib
from typing import Any, Dict

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


def _serialize_result(result: Any) -> Dict:
    """Safely serialize callback results to JSON-compatible format.
    
    Handles numpy arrays, pandas objects, sparse matrices, and other
    common scientific Python types.
    
    Args:
        result: Any Python object to serialize
        
    Returns:
        JSON-compatible dictionary/list/primitive
    """
    if isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    elif isinstance(result, (list, tuple)):
        return [_serialize_result(item) for item in result]
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, (np.integer, np.floating)):
        return result.item()
    elif isinstance(result, pd.Series):
        return result.tolist()
    elif isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    elif hasattr(result, "toarray"):
        # Sparse matrix
        return result.toarray().tolist()
    elif isinstance(result, (str, int, float, bool, type(None))):
        return result
    else:
        return str(result)
