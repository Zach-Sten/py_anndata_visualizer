"""
Main interface for creating AnnData visualizations.
"""
from pathlib import Path
from ..bridge.link_buttons import link_buttons_to_python
from ..tools.callback_functions import (
    get_obs_column,
    get_gene_expression,
    set_embedding_spatial,
    set_embedding_umap,
    set_embedding_pca,
    clear_plot,
    save_obs_column,
)


def create_adata_interface(adata, figsize=(900, 600), debug=False):
    """
    Create an interactive spatial plotting interface for AnnData objects.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with spatial coordinates and expression data
    figsize : tuple, optional
        (width, height) of the plot panel in pixels
    debug : bool, optional
        Enable debug logging
        
    Returns
    -------
    str
        Iframe ID for the created interface
    """
    # Load HTML template from the html directory
    html_path = Path(__file__).parent.parent / 'html' / 'controls_template.html'
    
    if not html_path.exists():
        raise FileNotFoundError(
            f"Could not find controls_template.html at {html_path}"
        )
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_template = f.read()
    
    # Unpack figsize tuple
    width, height = figsize
    
    initial_data = {
        "obs_columns": list(adata.obs.columns),
        "var_names": list(adata.var_names[:1000]),
    }
    
    return link_buttons_to_python(
        html_template,
        button_callbacks={
            "obsBtn": get_obs_column,
            "geneBtn": get_gene_expression,
            "spatialBtn": set_embedding_spatial,
            "umapBtn": set_embedding_umap,
            "pcaBtn": set_embedding_pca,
            "clearObsBtn": clear_plot,
            "clearGexBtn": clear_plot,
            "__save_obs_column__": save_obs_column,
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
    )