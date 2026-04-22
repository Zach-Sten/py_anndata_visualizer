"""
Main interface for creating interactive AnnData visualizations.

This module provides the high-level API for creating scatter plot visualizations
of AnnData objects with support for spatial, UMAP, and PCA embeddings.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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
    save_history,
    refresh_adata,
)
from ..tools.region_functions import (
    run_dbscan,
    compute_alpha_shapes,
    save_region_masks,
    load_region_masks,
    recompute_region_polygons,
    recapture_region_cells,
    save_manual_masks,
    load_manual_masks,
    save_region_group_to_obs,
    rename_region_mask,
    transform_manual_paths,
)
from ..tools.heatmap_functions import (
    compute_heatmap_bins,
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
    extra_obsm: Optional[List[str]] = None,
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
        chunk_size: Number of cells per chunk for progressive loading (default 250,000 cells).
        max_result_size: Maximum size in bytes for callback results (default 30MB)
        extra_obsm: Optional list of additional obsm keys to load as embeddings
                    (e.g. ['X_scANVI', 'X_harmony']). First 2 dims are used.
                    A button is added for each, and buttons for missing default
                    embeddings (spatial/umap/pca) are hidden automatically.
        
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
    
    # Validate sample_id early so the user gets a clear error before rendering
    if sample_id is not None and sample_id not in adata.obs.columns:
        raise ValueError(
            f"sample_id='{sample_id}' not found in adata.obs.columns.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Get sample info if sample_id provided
    sample_info = None
    if sample_id and sample_id in adata.obs.columns:
        sample_info = {
            "column": sample_id,
            "samples": list(adata.obs[sample_id].astype(str).unique()),
        }
    
    standard_keys = {'X_spatial', 'X_umap', 'X_pca', 'X_tsne', 'spatial'}

    # Detect which default embeddings exist
    has_spatial = "spatial" in adata.obsm or "X_spatial" in adata.obsm
    has_umap = "X_umap" in adata.obsm
    has_pca = "X_pca" in adata.obsm

    # Normalize extra_obsm: accept a plain string as a single-item list
    if isinstance(extra_obsm, str):
        extra_obsm = [extra_obsm]

    # Validate custom extra obsm keys
    custom_obsm_keys = []
    if extra_obsm:
        for key in extra_obsm:
            if key in adata.obsm and np.asarray(adata.obsm[key]).shape[1] >= 2:
                custom_obsm_keys.append(key)
            else:
                print(f"[Warning] extra_obsm key '{key}' not found or has < 2 dims, skipping")

    # Detect saved layout embeddings in obsm — exclude standard keys AND extra_obsm embedding keys
    embedding_keys = standard_keys | set(custom_obsm_keys)
    existing_layouts = [k for k in adata.obsm.keys() if k.startswith("X_") and k not in embedding_keys]

    # Build available_embeddings list for the JS UI
    available_embeddings = []
    if has_spatial:
        available_embeddings.append({"key": "spatial", "label": "Spatial"})
    if has_umap:
        available_embeddings.append({"key": "umap", "label": "UMAP"})
    if has_pca:
        available_embeddings.append({"key": "pca", "label": "PCA"})
    for key in custom_obsm_keys:
        label = key[2:] if key.startswith("X_") else key
        available_embeddings.append({"key": key, "label": label})

    # Detect any saved mask sources in uns (anything ending in _masks)
    existing_mask_sources = [
        {"key": k, "label": k.replace("_", " ").title()}
        for k in adata.uns.keys()
        if k.endswith("_masks")
    ]

    # Load existing history from adata.uns (stored as JSON string)
    _raw_history = adata.uns.get("__history__", "[]")
    if isinstance(_raw_history, str):
        try:
            existing_history = json.loads(_raw_history)
        except Exception:
            existing_history = []
    else:
        existing_history = list(_raw_history) if _raw_history else []

    # Build initial data payload for JavaScript
    initial_data = {
        "obs_columns": list(adata.obs.columns),
        "var_names": list(adata.var_names),  # All genes for autocomplete
        "cat_columns": cat_columns,
        "sample_id": sample_id,
        "sample_info": sample_info,
        "existing_layouts": existing_layouts,
        "available_embeddings": available_embeddings,
        "existing_mask_sources": existing_mask_sources,
        "history": existing_history,
    }

    def _active_sample_id(adata):
        """Return sample_id from init arg, falling back to runtime-switched value in adata.uns."""
        return sample_id or (adata.uns.get("__active_sample_id__") if adata is not None else None)

    # Create callback wrappers that inject sample_id
    def _compute_layout(data, adata=None, __sample_idx=None, **kwargs):
        return compute_layout(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _save_layout(data, adata=None, __sample_idx=None, **kwargs):
        return save_layout(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _load_layout(data, adata=None, __sample_idx=None, **kwargs):
        return load_layout(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _get_sample_meta(data, adata=None, __sample_idx=None, **kwargs):
        return get_sample_meta(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _run_dbscan(data, adata=None, __sample_idx=None, **kwargs):
        return run_dbscan(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _compute_alpha_shapes(data, adata=None, __sample_idx=None, **kwargs):
        return compute_alpha_shapes(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _save_region_masks(data, adata=None, __sample_idx=None, **kwargs):
        return save_region_masks(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _load_region_masks(data, adata=None, __sample_idx=None, **kwargs):
        return load_region_masks(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _recompute_region_polygons(data, adata=None, __sample_idx=None, **kwargs):
        return recompute_region_polygons(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _recapture_region_cells(data, adata=None, __sample_idx=None, **kwargs):
        return recapture_region_cells(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _compute_heatmap_bins(data, adata=None, __sample_idx=None, **kwargs):
        return compute_heatmap_bins(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _save_manual_masks(data, adata=None, __sample_idx=None, **kwargs):
        return save_manual_masks(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _load_manual_masks(data, adata=None, __sample_idx=None, **kwargs):
        return load_manual_masks(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _save_region_group_to_obs(data, adata=None, __sample_idx=None, **kwargs):
        return save_region_group_to_obs(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _rename_region_mask(data, adata=None, __sample_idx=None, **kwargs):
        return rename_region_mask(data, adata=adata, __sample_id__=_active_sample_id(adata))

    def _transform_manual_paths(data, adata=None, __sample_idx=None, **kwargs):
        return transform_manual_paths(data, adata=adata, __sample_id__=_active_sample_id(adata))

    return link_buttons_to_python(
        html_template,
        extra_obsm=custom_obsm_keys,
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
            "importLayoutFromObsmBtn": _load_layout,  # Import button uses same Python callback
            "sampleMetaBtn": _get_sample_meta,
            
            # Region functions (with sample_id injected)
            "dbscanBtn": _run_dbscan,
            "alphaShapeBtn": _compute_alpha_shapes,
            "saveRegionMasksBtn": _save_region_masks,
            "loadRegionMasksBtn": _load_region_masks,
            "recomputeRegionPolygonsBtn": _recompute_region_polygons,
            "recaptureRegionCellsPyBtn": _recapture_region_cells,

            # Heatmap functions
            "computeHeatmapBtn": _compute_heatmap_bins,
            
            # Manual selection mask functions
            "saveManualMasksBtn": _save_manual_masks,
            "loadManualMasksBtn": _load_manual_masks,

            # Region group → adata.obs (full indices from server-side cache)
            "saveRegionGroupToObsBtn": _save_region_group_to_obs,

            # Rename a region mask in all server-side caches
            "renameRegionMaskBtn": _rename_region_mask,

            # Transform manual selection paths to a new embedding
            "transformManualPathsBtn": _transform_manual_paths,

            # Refresh obs/obsm discovery and optionally switch sample_id column
            "refreshBtn": refresh_adata,

            # Persist UI history to adata.uns
            "saveHistoryBtn": save_history,
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
        chunk_size=chunk_size,
        max_result_size=max_result_size,
    )