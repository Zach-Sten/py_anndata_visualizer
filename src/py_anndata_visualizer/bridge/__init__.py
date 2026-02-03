"""
Bridge module for py_anndata_visualizer.

Provides the core functionality for connecting HTML UI to Python callbacks
via ipywidgets-based communication.
"""

from .link_buttons import link_buttons_to_python

__all__ = [
    "link_buttons_to_python",
]
