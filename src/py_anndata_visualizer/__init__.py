"""
AnnData Visualizer - Interactive spatial plotting for single-cell data.
"""

# Clean up any orphaned widgets on import
try:
    import ipywidgets as widgets
    widgets.Widget.close_all()
except:
    pass

from .plotting.adata_interface import create_adata_interface

__version__ = "0.1.0-beta"
__all__ = ["create_adata_interface"]