"""
Callback functions for handling user interactions with the visualizer.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Callback functions for adata:
def get_obs_column(data, adata=None, __sample_idx=None):
    if adata is None:
        return {"type": "error", "message": "adata not provided"}

    column = (data.get("column", "") or "").strip()
    if not column:
        return {"type": "error", "message": "Please enter a column name"}

    if column not in adata.obs.columns:
        return {
            "type": "error",
            "message": f"Column '{column}' not found",
            "available_columns": list(adata.obs.columns[:15]),
        }

    idx = np.asarray(__sample_idx if __sample_idx is not None else np.arange(adata.n_obs), dtype=int)
    vals = adata.obs[column].iloc[idx]

    # Numeric → continuous
    if pd.api.types.is_numeric_dtype(vals):
        v = pd.to_numeric(vals, errors="coerce").astype(float).fillna(np.nan)
        return {
            "type": "obs_values",
            "column": column,
            "mode": "continuous",
            "values": v.tolist(),
        }

    # Categorical → codes + categories + colors (prefer adata.uns palette)
    if pd.api.types.is_categorical_dtype(vals):
        categories = list(vals.cat.categories)
        codes = vals.cat.codes.to_numpy()
        codes = np.where(codes < 0, 0, codes)  # missing -> 0
        values_idx = codes.tolist()
    else:
        categories = list(pd.unique(vals))
        lut = {v: i for i, v in enumerate(categories)}
        values_idx = [lut[v] for v in vals]

    palette = None
    key = f"{column}_colors"
    if hasattr(adata, "uns") and key in adata.uns:
        try:
            pal = list(adata.uns[key])
            if len(pal) >= len(categories):
                palette = pal[:len(categories)]
        except Exception:
            palette = None

    return {
        "type": "obs_values",
        "column": column,
        "mode": "categorical",
        "values": values_idx,         # aligned to plotted points
        "categories": categories,
        "colors": palette,            # may be None
    }


def get_gene_expression(data, adata=None, __sample_idx=None):
    if adata is None:
        return {"type": "error", "message": "adata not provided"}

    gene = (data.get("gene", "") or "").strip()
    if not gene:
        return {"type": "error", "message": "Please enter a gene name"}

    if gene not in adata.var_names:
        similar = [g for g in adata.var_names if gene.upper() in g.upper()][:8]
        return {"type": "error", "message": f"Gene '{gene}' not found", "similar_genes": similar or None}

    idx = np.asarray(__sample_idx if __sample_idx is not None else np.arange(adata.n_obs), dtype=int)
    j = int(np.where(adata.var_names == gene)[0][0])

    X = adata.X
    if sp.issparse(X):
        v = np.asarray(X[idx, j].toarray()).ravel().astype(float)
    else:
        v = np.asarray(X[idx, j]).ravel().astype(float)

    # optional: log1p-like display scaling (comment out if you want raw)
    # v = np.log1p(np.maximum(v, 0))

    return {
        "type": "gex_values",
        "gene": gene,
        "values": v.tolist(),   # aligned to plotted points
    }

def set_embedding_spatial(data, adata=None, __sample_idx=None):
    return {"type": "set_embedding", "embedding": "spatial"}

def set_embedding_umap(data, adata=None, __sample_idx=None):
    return {"type": "set_embedding", "embedding": "umap"}

def set_embedding_pca(data, adata=None, __sample_idx=None):
    return {"type": "set_embedding", "embedding": "pca"}

def clear_plot(data, adata=None, __sample_idx=None):
    # Always reset to neutral grey in the parent plot
    return {"type": "clear_plot"}

def save_obs_column(data, adata=None, __sample_idx=None):
    """Save folder selections as a new categorical column in adata.obs"""
    import pandas as pd
    import numpy as np
    
    column_name = data.get("columnName", "selection_group")
    column_data = data.get("columnData", {})  # {index: selection_name}
    
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    # Create column with NaN for all cells
    n_cells = adata.n_obs
    new_column = pd.Series([np.nan] * n_cells, index=adata.obs.index, dtype='object')
    
    # Fill in selections
    for idx_str, sel_name in column_data.items():
        idx = int(idx_str)
        if 0 <= idx < n_cells:
            new_column.iloc[idx] = sel_name
    
    # Add to adata.obs
    adata.obs[column_name] = new_column
    
    # Convert to categorical for efficiency
    adata.obs[column_name] = pd.Categorical(adata.obs[column_name])
    
    print(f"✓ Saved '{column_name}' to adata.obs")
    print(f"  Categories: {list(adata.obs[column_name].cat.categories)}")
    print(f"  NaN count: {adata.obs[column_name].isna().sum()}")
    
    return {"type": "success"}