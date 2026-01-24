"""
Bridge module - connects HTML UI to Python callbacks.
"""

from .link_buttons import link_buttons_to_python, _b64, _serialize_result

__all__ = ["link_buttons_to_python", "_b64", "_serialize_result"]