"""
Main interface for creating interactive AnnData visualizations.

This module provides the high-level API for creating scatter plot visualizations
of AnnData objects with support for spatial, UMAP, and PCA embeddings.
"""

from pathlib import Path
from typing import Optional, Tuple

from ..tools.callback_functions import (
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
    load_full_embedding,
    save_obs_column,
    save_color_scheme,
    compute_layout,
    save_layout,
    delete_layout,
    load_layout,
    get_sample_meta,
)
from ..tools.region_functions import (
    run_dbscan,
    compute_alpha_shapes,
)
from ..bridge.link_buttons import link_buttons_to_python


def _load_html_template() -> str:
    """Load the default HTML template for the control panel."""
    html_dir = Path(__file__).parent.parent / "html"
    template_path = html_dir / "controls_template.html"
    
    if not template_path.exists():
        raise FileNotFoundError(
            f"HTML template not found at {template_path}. "
            "Please ensure the html/ directory contains controls_template.html"
        )
    
    return template_path.read_text(encoding="utf-8")


def create_adata_interface(
    adata,
    figsize: Tuple[int, int] = (900, 600),
    debug: bool = False,
    html_template: Optional[str] = None,
    sample_id: Optional[str] = None,
    chunk_size: int = 250_000,
    max_result_size: int = 20_000_000,
):
    """
    Create an interactive visualization interface for an AnnData object.
    
    This function sets up a two-panel interface with:
    - Left panel: Control UI (embedding selector, gene input, obs column selector, etc.)
    - Right panel: WebGL scatter plot canvas with pan/zoom/rotate support
    
    Args:
        adata: AnnData object with spatial coordinates in obsm['spatial'] or obsm['X_spatial']
        figsize: Tuple of (width, height) in pixels for the visualization
        debug: If True, enables verbose logging in browser console
        html_template: HTML template string for the control panel UI. If None, loads the
                      default template from the html/ directory.
        sample_id: Name of obs column containing sample IDs (enables layout features)
        chunk_size: Number of cells per chunk for progressive loading (default 500K)
        max_result_size: Maximum size in bytes for callback results (default 30MB)
        
    Returns:
        str: The iframe ID for the created visualization
        
    Example:
        >>> import anndata as ad
        >>> from py_anndata_visualizer import create_adata_interface
        >>> adata = ad.read_h5ad("my_data.h5ad")
        >>> create_adata_interface(adata, figsize=(1000, 700), sample_id="sample")
    """
    # Auto-load HTML template if not provided
    if html_template is None:
        html_template = _load_html_template()
    
    # Unpack figsize tuple
    width, height = figsize
    
    # Detect categorical/string columns for layout dropdowns
    cat_columns = []
    for col in adata.obs.columns:
        if col.startswith("__"):
            continue
        if hasattr(adata.obs[col], 'cat') or adata.obs[col].dtype == object:
            cat_columns.append(col)
    
    # Get sample info if sample_id provided
    sample_info = None
    if sample_id and sample_id in adata.obs.columns:
        sample_info = {
            "column": sample_id,
            "samples": list(adata.obs[sample_id].astype(str).unique()),
        }
    
    # Detect existing layout embeddings in obsm (keys starting with X_ that aren't standard embeddings)
    standard_keys = {'X_spatial', 'X_umap', 'X_pca', 'X_tsne', 'spatial'}
    existing_layouts = [k for k in adata.obsm.keys() if k.startswith("X_") and k not in standard_keys]
    
    # Build initial data payload for JavaScript
    initial_data = {
        "obs_columns": list(adata.obs.columns),
        "var_names": list(adata.var_names[:1000]),  # First 1000 genes for autocomplete
        "cat_columns": cat_columns,
        "sample_id": sample_id,
        "sample_info": sample_info,
        "existing_layouts": existing_layouts,
    }

    # Create callback wrappers that inject sample_id
    def _compute_layout(data, adata=None, __sample_idx=None):
        return compute_layout(data, adata=adata, __sample_id__=sample_id)
    
    def _save_layout(data, adata=None, __sample_idx=None):
        return save_layout(data, adata=adata, __sample_id__=sample_id)
    
    def _load_layout(data, adata=None, __sample_idx=None):
        return load_layout(data, adata=adata, __sample_id__=sample_id)
    
    def _get_sample_meta(data, adata=None, __sample_idx=None):
        return get_sample_meta(data, adata=adata, __sample_id__=sample_id)
    
    def _run_dbscan(data, adata=None, __sample_idx=None):
        return run_dbscan(data, adata=adata, __sample_id__=sample_id)
    
    def _compute_alpha_shapes(data, adata=None, __sample_idx=None):
        return compute_alpha_shapes(data, adata=adata, __sample_id__=sample_id)

    return link_buttons_to_python(
        html_template,
        button_callbacks={
            # Obs column coloring
            "obsBtn": get_obs_column,
            
            # Gene expression
            "geneBtn": get_gene_expression,
            "geneGroupBtn": get_gene_group_expression,

            # Embedding buttons
            "spatialBtn": set_embedding_spatial,
            "umapBtn": set_embedding_umap,
            "pcaBtn": set_embedding_pca,

            # Clear buttons
            "clearObsBtn": clear_obs,
            "clearGexBtn": clear_gex,
            
            # CHUNKED: viewport cell loader (for spatial zoom detail)
            "viewportBtn": get_viewport_cells,
            
            # CHUNKED: load specific chunk (for UMAP/PCA progressive loading)
            "chunkBtn": get_chunk_cells,
            
            # LEGACY: full embedding loader (deprecated but kept for compat)
            "loadEmbeddingBtn": load_full_embedding,
            
            # Save selection folder to obs
            "__save_obs_column__": save_obs_column,
            
            # Save color scheme to adata.uns
            "saveColorsBtn": save_color_scheme,
            
            # Layout functions (with sample_id injected)
            "computeLayoutBtn": _compute_layout,
            "saveLayoutBtn": _save_layout,
            "obsmBtn": _save_layout,  # Alias for saving to obsm
            "deleteLayoutBtn": delete_layout,
            "loadLayoutBtn": _load_layout,
            "sampleMetaBtn": _get_sample_meta,
            
            # Region functions (with sample_id injected)
            "dbscanBtn": _run_dbscan,
            "alphaShapeBtn": _compute_alpha_shapes,
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
        chunk_size=chunk_size,
        max_result_size=max_result_size,
    )
