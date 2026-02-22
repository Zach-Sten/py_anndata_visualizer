"""
py_anndata_visualizer - Interactive visualization for AnnData objects.

A high-performance, WebGL-based scatter plot visualization tool for
single-cell and spatial transcriptomics data stored in AnnData format.

Features:
- Interactive pan/zoom/rotate with smooth animations
- Support for spatial, UMAP, and PCA embeddings
- Progressive chunked loading for millions of cells
- Gene expression coloring with customizable colormaps
- Categorical obs column coloring with interactive legend
- Lasso, rectangle, circle, and polygon selection tools
- Grid-based sample layout computation
- Region detection via DBSCAN + alpha shape boundaries
- Export to PNG

Basic Usage:
    >>> import anndata as ad
    >>> from py_anndata_visualizer import create_adata_interface
    >>> adata = ad.read_h5ad("my_data.h5ad")
    >>> create_adata_interface(adata, figsize=(1000, 700))

With sample layout support:
    >>> create_adata_interface(adata, figsize=(1000, 700), sample_id="sample")
"""

__version__ = "0.1.0"

# Main interface
from .plotting import create_adata_interface

# Bridge functionality (for advanced users)
from .bridge import link_buttons_to_python

# Callback functions (for extending/customizing)
from .tools import (
    # Utils
    _pack_coords_binary,
    _b64,
    _serialize_result,
    
    # Callbacks
    get_obs_column,
    get_gene_expression,
    get_gene_group_expression,
    set_embedding_spatial,
    set_embedding_umap,
    set_embedding_pca,
    clear_obs,
    clear_gex,
    get_viewport_cells,
    get_chunk_cells,
    compute_layout,
    save_layout,
    load_layout,
    delete_layout,
    get_sample_meta,
    save_obs_column,
    save_color_scheme,
    
    # Region functions
    run_dbscan,
    compute_alpha_shapes,
)

# Communication helpers (for advanced users)
from .helpers import (
    create_data_bridges,
    create_poll_button,
    send_to_javascript,
    make_callback_handler,
    get_dispatcher_script,
    get_communication_script,
)

__all__ = [
    # Version
    "__version__",
    
    # Main API
    "create_adata_interface",
    "link_buttons_to_python",
    
    # Utils
    "_pack_coords_binary",
    "_b64",
    "_serialize_result",
    
    # Callbacks
    "get_obs_column",
    "get_gene_expression",
    "get_gene_group_expression",
    "set_embedding_spatial",
    "set_embedding_umap",
    "set_embedding_pca",
    "clear_obs",
    "clear_gex",
    "get_viewport_cells",
    "get_chunk_cells",
    "compute_layout",
    "save_layout",
    "load_layout",
    "delete_layout",
    "get_sample_meta",
    "save_obs_column",
    "save_color_scheme",
    
    # Region functions
    "run_dbscan",
    "compute_alpha_shapes",
    
    # Helpers
    "create_data_bridges",
    "create_poll_button",
    "send_to_javascript",
    "make_callback_handler",
    "get_dispatcher_script",
    "get_communication_script",
]