"""
Tools module for py_anndata_visualizer.

Contains utility functions and callback handlers for the visualization interface.
"""

from .utils import (
    _pack_coords_binary,
    _b64,
    _serialize_result,
)

from .callback_functions import (
    # Embedding functions
    load_full_embedding,
    set_embedding_spatial,
    set_embedding_umap,
    set_embedding_pca,
    
    # Clear functions
    clear_plot,
    clear_obs,
    clear_gex,
    
    # Obs column functions
    get_obs_column,
    save_obs_column,
    save_color_scheme,
    
    # Gene expression functions
    get_gene_expression,
    get_gene_group_expression,
    
    # Chunked/viewport loading functions
    get_viewport_cells,
    get_chunk_cells,
    
    # Layout functions
    get_sample_meta,
    compute_layout,
    save_layout,
    delete_layout,
    load_layout,
)

from .region_functions import (
    run_dbscan,
    compute_alpha_shapes,
)

__all__ = [
    # Utils
    "_pack_coords_binary",
    "_b64", 
    "_serialize_result",
    
    # Callbacks
    "load_full_embedding",
    "set_embedding_spatial",
    "set_embedding_umap",
    "set_embedding_pca",
    "clear_plot",
    "clear_obs",
    "clear_gex",
    "get_obs_column",
    "save_obs_column",
    "save_color_scheme",
    "get_gene_expression",
    "get_gene_group_expression",
    "get_viewport_cells",
    "get_chunk_cells",
    "get_sample_meta",
    "compute_layout",
    "save_layout",
    "delete_layout",
    "load_layout",
    
    # Region functions
    "run_dbscan",
    "compute_alpha_shapes",
]