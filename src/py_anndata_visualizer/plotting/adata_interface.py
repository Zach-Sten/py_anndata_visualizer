def create_adata_interface(adata, figsize=(900, 600), debug=False, html_template: str = ""):
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
    
            # NEW: embedding buttons
            "spatialBtn": set_embedding_spatial,
            "umapBtn": set_embedding_umap,
            "pcaBtn": set_embedding_pca,
    
            # NEW: clear buttons (wire BOTH clear buttons to this)
            "clearObsBtn": clear_plot,
            "clearGexBtn": clear_plot,
            
            # NEW: save selection folder to obs
            "__save_obs_column__": save_obs_column,
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
    )