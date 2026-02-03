"""
Python callback functions for handling button clicks from the visualization interface.

These functions are called via the widget bridge when users interact with the UI.
They operate on adata objects and return JSON-serializable results.
"""

import base64
import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import _pack_coords_binary


# =============================================================================
# Embedding Functions
# =============================================================================

def load_full_embedding(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Load full embedding data (for UMAP/PCA - no streaming, load all cells)."""
    if adata is None:
        return {"type": "error", "message": "No data"}
    
    embedding = data.get("embedding", "umap")
    
    if embedding == "umap":
        if "X_umap" not in adata.obsm:
            return {"type": "error", "message": "No UMAP embedding"}
        coords = np.asarray(adata.obsm["X_umap"])[:, :2]
    elif embedding == "pca":
        if "X_pca" not in adata.obsm:
            return {"type": "error", "message": "No PCA embedding"}
        coords = np.asarray(adata.obsm["X_pca"])[:, :2]
    else:
        return {"type": "error", "message": f"Unknown embedding: {embedding}"}
    
    indices = np.arange(len(coords))
    print(f"[Full Embedding] Loading {embedding}: {len(indices):,} cells")
    
    coords_binary = base64.b64encode(coords.astype(np.float32).tobytes()).decode('ascii')
    
    return {
        "type": "full_embedding",
        "embedding": embedding,
        "indices": indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(indices)
    }


def set_embedding_spatial(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Switch to spatial embedding."""
    return {"type": "set_embedding", "embedding": "spatial"}


def set_embedding_umap(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Switch to UMAP embedding."""
    if adata is None or "X_umap" not in adata.obsm:
        return {"type": "error", "message": "No UMAP embedding"}
    return {"type": "set_embedding", "embedding": "umap"}


def set_embedding_pca(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Switch to PCA embedding."""
    if adata is None or "X_pca" not in adata.obsm:
        return {"type": "error", "message": "No PCA embedding"}
    return {"type": "set_embedding", "embedding": "pca"}


# =============================================================================
# Clear Functions
# =============================================================================

def clear_plot(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Clear all overlays."""
    return {"type": "clear_plot"}


def clear_obs(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Clear obs column coloring."""
    return {"type": "clear_obs"}


def clear_gex(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Clear gene expression coloring."""
    return {"type": "clear_gex"}


# =============================================================================
# Obs Column Functions
# =============================================================================

def get_obs_column(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Return category metadata AND compressed per-cell codes."""
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    column = (data.get("column", "") or "").strip()
    if not column:
        return {"type": "error", "message": "Please enter a column name"}
    if column not in adata.obs.columns:
        return {"type": "error", "message": f"Column '{column}' not found",
                "available_columns": list(adata.obs.columns[:15])}
    
    vals = adata.obs[column]
    idx = np.asarray(__sample_idx if __sample_idx is not None else np.arange(adata.n_obs), dtype=int)
    
    # Numeric → quantize to uint8
    if pd.api.types.is_numeric_dtype(vals):
        v = np.nan_to_num(vals.values[idx].astype(float), nan=0.0)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        codes = np.zeros(len(v), dtype=np.uint8)
        if vmax > vmin:
            codes[np.isfinite(v)] = np.clip(
                ((v[np.isfinite(v)] - vmin) / (vmax - vmin) * 254 + 1), 1, 255
            ).astype(np.uint8)
        compressed = zlib.compress(codes.tobytes(), level=6)
        b64 = base64.b64encode(compressed).decode("ascii")
        return {
            "type": "obs_values", "column": column, "mode": "continuous",
            "categories": None, "colors": None, "codes_b64": b64, "count": len(codes)
        }
    
    # Categorical → 1-indexed codes
    if pd.api.types.is_categorical_dtype(vals):
        categories = list(vals.cat.categories)
        raw_codes = vals.cat.codes.values[idx]
        codes = (raw_codes + 1).astype(np.int16)
        codes[raw_codes < 0] = 0
    else:
        categories = list(pd.unique(vals.dropna()))
        cat_to_idx = {c: i + 1 for i, c in enumerate(categories)}
        codes = np.array([cat_to_idx.get(v, 0) for v in vals.values[idx]], dtype=np.int16)
    
    # Get palette from adata.uns
    palette = None
    key = f"{column}_colors"
    if hasattr(adata, "uns") and key in adata.uns:
        try:
            pal = list(adata.uns[key])
            if len(pal) >= len(categories):
                palette = pal[:len(categories)]
        except Exception:
            pass
    
    codes_u8 = np.clip(codes, 0, 255).astype(np.uint8)
    compressed = zlib.compress(codes_u8.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    
    return {
        "type": "obs_values", "column": column, "mode": "categorical",
        "categories": categories, "colors": palette, "codes_b64": b64, "count": len(codes_u8)
    }


def save_obs_column(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Save folder selections as a new categorical column in adata.obs."""
    column_name = data.get("columnName", "selection_group")
    column_data = data.get("columnData", {})
    
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    n_cells = adata.n_obs
    new_column = pd.Series([np.nan] * n_cells, index=adata.obs.index, dtype='object')
    
    for idx_str, sel_name in column_data.items():
        idx = int(idx_str)
        if 0 <= idx < n_cells:
            new_column.iloc[idx] = sel_name
    
    adata.obs[column_name] = new_column
    adata.obs[column_name] = pd.Categorical(adata.obs[column_name])
    
    print(f"✓ Saved '{column_name}' to adata.obs")
    print(f"  Categories: {list(adata.obs[column_name].cat.categories)}")
    print(f"  NaN count: {adata.obs[column_name].isna().sum()}")
    
    return {"type": "success"}


def save_color_scheme(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Save color scheme to adata.uns['{column}_colors']."""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    column = data.get("column")
    colors = data.get("colors")
    categories = data.get("categories")
    
    if not column or not colors:
        return {"type": "error", "message": "Missing column or colors data"}
    
    uns_key = f"{column}_colors"
    adata.uns[uns_key] = list(colors)
    
    print(f"✓ Saved color scheme to adata.uns['{uns_key}']")
    print(f"  {len(colors)} colors for {len(categories) if categories else '?'} categories")
    
    return {"type": "success", "message": f"Colors saved to adata.uns['{uns_key}']"}


# =============================================================================
# Gene Expression Functions
# =============================================================================

def get_gene_expression(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Get gene expression values, quantized and compressed."""
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
        v = np.asarray(X[idx, j].toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(X[idx, j]).ravel().astype(np.float32)
    
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
    vmax = float(v.max())
    
    quantized = np.zeros(len(v), dtype=np.uint8)
    if vmax > 0:
        quantized[v > 0] = np.clip((v[v > 0] / vmax * 254 + 1), 1, 255).astype(np.uint8)
    
    compressed = zlib.compress(quantized.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    
    return {
        "type": "gex_values",
        "gene": gene,
        "count": len(quantized),
        "vmax": round(vmax, 4),
        "values_b64": b64,
    }


def get_gene_group_expression(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Compute geometric mean expression for a group of genes."""
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    genes = data.get("genes", [])
    method = data.get("method", "geometric_mean")
    group_name = data.get("groupName", "group")
    
    if not genes or len(genes) == 0:
        return {"type": "error", "message": "No genes specified"}
    
    valid_genes = [g for g in genes if g in adata.var_names]
    if not valid_genes:
        return {"type": "error", "message": f"None of the genes found: {genes}"}
    
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        print(f"[GeneGroup] Warning: genes not found: {missing}")
    
    idx = np.asarray(__sample_idx if __sample_idx is not None else np.arange(adata.n_obs), dtype=int)
    X = adata.X
    
    gene_values = []
    for gene in valid_genes:
        j = int(np.where(adata.var_names == gene)[0][0])
        if sp.issparse(X):
            v = np.asarray(X[idx, j].toarray()).ravel().astype(np.float32)
        else:
            v = np.asarray(X[idx, j]).ravel().astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.maximum(v, 0.0)
        gene_values.append(v)
    
    stacked = np.stack(gene_values, axis=0)
    
    # Geometric mean using log transform for numerical stability
    epsilon = 1e-9
    log_vals = np.log(stacked + epsilon)
    combined = np.exp(np.mean(log_vals, axis=0)) - epsilon
    combined = np.maximum(combined, 0.0)
    
    vmax = float(combined.max())
    
    quantized = np.zeros(len(combined), dtype=np.uint8)
    if vmax > 0:
        quantized[combined > 0] = np.clip(
            (combined[combined > 0] / vmax * 254 + 1), 1, 255
        ).astype(np.uint8)
    
    compressed = zlib.compress(quantized.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    
    print(f"[GeneGroup] {group_name}: geometric_mean({valid_genes}) → vmax={vmax:.4f}")
    
    return {
        "type": "gex_values",
        "gene": group_name,
        "genes_in_group": valid_genes,
        "method": "geometric_mean",
        "count": len(quantized),
        "vmax": round(vmax, 4),
        "values_b64": b64,
    }


# =============================================================================
# Chunked/Viewport Loading Functions
# =============================================================================

def get_viewport_cells(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """
    CHUNKED STREAMING: Load cells in viewport from NON-CHUNK-0 cells.
    Chunk 0 is always pre-loaded, this adds detail from other chunks when zooming.
    Max 30K cells returned at once with Gaussian center-weighting.
    """
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    embedding = data.get("embedding", "spatial")
    view_minX = data.get("viewMinX")
    view_maxX = data.get("viewMaxX")
    view_minY = data.get("viewMinY")
    view_maxY = data.get("viewMaxY")
    
    if view_minX is None or view_maxX is None or view_minY is None or view_maxY is None:
        return {"type": "error", "message": "Invalid viewport bounds"}
    
    active_column = data.get("activeColumn", None)
    active_gene = data.get("activeGene", None)
    
    # Get embedding coordinates
    if embedding == "spatial":
        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"])[:, :2]
        elif "X_spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["X_spatial"])[:, :2]
        else:
            return {"type": "error", "message": "No spatial embedding"}
    elif embedding == "umap":
        if "X_umap" not in adata.obsm:
            return {"type": "error", "message": "No UMAP embedding"}
        coords = np.asarray(adata.obsm["X_umap"])[:, :2]
    elif embedding == "pca":
        if "X_pca" not in adata.obsm:
            return {"type": "error", "message": "No PCA embedding"}
        coords = np.asarray(adata.obsm["X_pca"])[:, :2]
    else:
        return {"type": "error", "message": f"Unknown embedding: {embedding}"}
    
    # CIRCULAR viewport for rotation-invariant loading
    center_x = (view_minX + view_maxX) / 2
    center_y = (view_minY + view_maxY) / 2
    radius = (view_maxX - view_minX) / 2
    
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)
    
    in_circle = distances <= radius
    
    # Only load from NON-CHUNK-0 cells
    if "__chunk__" in adata.obs.columns:
        chunk_assignments = adata.obs["__chunk__"].values
        non_chunk0 = (chunk_assignments != 0)
        in_circle = in_circle & non_chunk0
    
    candidate_indices = np.where(in_circle)[0]
    
    if len(candidate_indices) == 0:
        return {
            "type": "viewport_cells",
            "embedding": embedding,
            "indices": [],
            "coords_binary": "",
            "coords_count": 0,
            "message": "No additional cells in viewport (chunk 0 covers this area)"
        }
    
    normalized_dist = distances[candidate_indices] / radius
    sigma = 0.5
    probabilities = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
    
    MAX_CELLS = 30_000
    if len(candidate_indices) > MAX_CELLS:
        probabilities = probabilities / probabilities.sum()
        visible_indices = np.random.choice(
            candidate_indices, size=MAX_CELLS, replace=False, p=probabilities
        )
        visible_indices.sort()
    else:
        visible_indices = candidate_indices
    
    print(f"[Streaming] Viewport {embedding}: {len(visible_indices):,} cells (circular, rotation-safe)")
    
    visible_coords = coords[visible_indices]
    coords_binary = _pack_coords_binary(visible_coords)
    
    chunks = None
    if "__chunk__" in adata.obs.columns:
        chunks = adata.obs["__chunk__"].iloc[visible_indices].tolist()
    
    response = {
        "type": "viewport_cells",
        "embedding": embedding,
        "indices": visible_indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(visible_indices),
        "chunks": chunks
    }
    
    # Add obs color data if requested
    if active_column and active_column in adata.obs.columns:
        vals = adata.obs[active_column].iloc[visible_indices]
        
        if pd.api.types.is_numeric_dtype(vals):
            v = pd.to_numeric(vals, errors="coerce").astype(float).fillna(np.nan)
            response["obs_values"] = v.tolist()
            response["obs_mode"] = "continuous"
            response["obs_column"] = active_column
        else:
            if pd.api.types.is_categorical_dtype(vals):
                categories = list(vals.cat.categories)
                codes = vals.cat.codes.to_numpy()
                codes = np.where(codes < 0, 0, codes)
                values_idx = codes.tolist()
            else:
                categories = list(pd.unique(vals))
                lut = {v: i for i, v in enumerate(categories)}
                values_idx = [lut[v] for v in vals]
            
            palette = None
            key = f"{active_column}_colors"
            if hasattr(adata, "uns") and key in adata.uns:
                try:
                    pal = list(adata.uns[key])
                    if len(pal) >= len(categories):
                        palette = pal[:len(categories)]
                except:
                    pass
            
            response["obs_values"] = values_idx
            response["obs_mode"] = "categorical"
            response["obs_column"] = active_column
            response["obs_categories"] = categories
            response["obs_colors"] = palette
    
    # Add gene expression if requested
    if active_gene and active_gene in adata.var_names:
        j = int(np.where(adata.var_names == active_gene)[0][0])
        X = adata.X
        if sp.issparse(X):
            v = np.asarray(X[visible_indices, j].toarray()).ravel().astype(float)
        else:
            v = np.asarray(X[visible_indices, j]).ravel().astype(float)
        response["gex_values"] = v.tolist()
        response["gex_gene"] = active_gene
    
    return response


def get_chunk_cells(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """
    CHUNKED: Load cells from a specific chunk with ALL embeddings (spatial, UMAP, PCA).
    This enables instant switching between embeddings on the client side.
    """
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    if "__chunk__" not in adata.obs.columns:
        return {"type": "error", "message": "Chunk assignments not found - run initialization first"}
    
    chunk_id = data.get("chunk", 1)
    request_id = data.get("requestId", None)
    active_column = data.get("activeColumn", None)
    active_gene = data.get("activeGene", None)
    
    print(f"[Chunk] Python processing request for chunk {chunk_id}, reqId={request_id}")
    
    chunk_mask = (adata.obs["__chunk__"].values == chunk_id)
    chunk_indices = np.where(chunk_mask)[0]
    
    if len(chunk_indices) == 0:
        return {"type": "error", "message": f"Chunk {chunk_id} is empty"}
    
    USE_COMPRESSION = True
    
    spatial_binary = None
    umap_binary = None
    pca_binary = None
    
    # Spatial
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])[chunk_indices, :2]
        spatial_binary = _pack_coords_binary(coords, compress=USE_COMPRESSION)
    elif "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"])[chunk_indices, :2]
        spatial_binary = _pack_coords_binary(coords, compress=USE_COMPRESSION)
    
    # UMAP
    if "X_umap" in adata.obsm:
        coords = np.asarray(adata.obsm["X_umap"])[chunk_indices, :2]
        umap_binary = _pack_coords_binary(coords, compress=USE_COMPRESSION)
    
    # PCA
    if "X_pca" in adata.obsm:
        coords = np.asarray(adata.obsm["X_pca"])[chunk_indices, :2]
        pca_binary = _pack_coords_binary(coords, compress=USE_COMPRESSION)
    
    response = {
        "type": "chunk_data",
        "chunk": chunk_id,
        "requestId": request_id,
        "indices": chunk_indices.tolist(),
        "spatial_binary": spatial_binary,
        "umap_binary": umap_binary,
        "pca_binary": pca_binary,
        "count": len(chunk_indices)
    }
    
    # Add per-cell sample IDs if available
    if "__cell_sample_ids__" in adata.uns:
        sids = adata.uns["__cell_sample_ids__"][chunk_indices]
        response["sids_b64"] = base64.b64encode(
            zlib.compress(sids.astype(np.uint16).tobytes(), level=6)
        ).decode("ascii")
    
    # Add obs color data if requested
    if active_column and active_column in adata.obs.columns:
        vals = adata.obs[active_column].iloc[chunk_indices]
        
        if pd.api.types.is_numeric_dtype(vals):
            v = pd.to_numeric(vals, errors="coerce").astype(float).fillna(np.nan)
            response["obs_values"] = v.tolist()
            response["obs_mode"] = "continuous"
            response["obs_column"] = active_column
        else:
            if pd.api.types.is_categorical_dtype(vals):
                categories = list(vals.cat.categories)
                codes = vals.cat.codes.to_numpy()
                codes = np.where(codes < 0, 0, codes)
                values_idx = codes.tolist()
            else:
                categories = list(pd.unique(adata.obs[active_column]))
                lut = {v: i for i, v in enumerate(categories)}
                values_idx = [lut.get(v, 0) for v in vals]
            
            palette = None
            key = f"{active_column}_colors"
            if hasattr(adata, "uns") and key in adata.uns:
                try:
                    pal = list(adata.uns[key])
                    if len(pal) >= len(categories):
                        palette = pal[:len(categories)]
                except:
                    pass
            
            response["obs_values"] = values_idx
            response["obs_mode"] = "categorical"
            response["obs_column"] = active_column
            response["obs_categories"] = categories
            response["obs_colors"] = palette
    
    # Add gene expression if requested
    if active_gene and active_gene in adata.var_names:
        j = int(np.where(adata.var_names == active_gene)[0][0])
        X = adata.X
        if sp.issparse(X):
            v = np.asarray(X[chunk_indices, j].toarray()).ravel().astype(float)
        else:
            v = np.asarray(X[chunk_indices, j]).ravel().astype(float)
        response["gex_values"] = v.tolist()
        response["gex_gene"] = active_gene
    
    return response


# =============================================================================
# Layout Functions
# =============================================================================

def get_sample_meta(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """Fetch per-sample metadata for a given obs column (on-demand)."""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    col = data.get("column", "")
    sample_col = __sample_id__ or data.get("sample_id")
    
    if not col or not col.strip():
        return {"type": "sample_meta", "column": "", "valid": False,
                "message": "No column specified"}
    if col not in adata.obs.columns:
        return {"type": "sample_meta", "column": col, "valid": False,
                "message": f"Column '{col}' not found in obs"}
    if not sample_col or sample_col not in adata.obs.columns:
        return {"type": "error", "message": f"Sample column not set"}
    
    sample_names = adata.uns.get("__sample_names__", [])
    if not sample_names:
        return {"type": "error", "message": "No sample names stored"}
    
    samp_arr = adata.obs[sample_col].astype(str).values
    col_vals = adata.obs[col].astype(str).values
    
    per_sample_val = []
    for s in sample_names:
        mask = (samp_arr == s)
        uniq = set(col_vals[mask])
        if len(uniq) > 1:
            return {"type": "sample_meta", "column": col, "valid": False,
                    "message": f"Column '{col}' has multiple values per sample"}
        per_sample_val.append(uniq.pop() if uniq else "__NA__")
    
    cats = sorted(set(per_sample_val))
    cat_to_code = {c: i for i, c in enumerate(cats)}
    codes = [cat_to_code[v] for v in per_sample_val]
    
    return {
        "type": "sample_meta",
        "column": col,
        "valid": True,
        "cats": cats,
        "codes": codes,
    }


def compute_layout(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """Compute grid layout coordinates for all cells based on sample grouping."""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    sample_col = __sample_id__ or data.get("sample_id")
    if not sample_col or sample_col not in adata.obs.columns:
        return {"type": "error", "message": f"Sample column '{sample_col}' not found"}
    
    group_col = data.get("group_by")
    group_ncols = int(data.get("group_ncols", 2))
    sample_ncols = int(data.get("sample_ncols", 3))
    sort_col = data.get("sort_by") or None
    order_col = data.get("order_by") or None
    gap_x = float(data.get("gap_x", 1.0))
    gap_y = float(data.get("gap_y", 1.0))
    
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])[:, :2].copy()
    elif "X_spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["X_spatial"])[:, :2].copy()
    else:
        return {"type": "error", "message": "No spatial coordinates"}
    
    samples_arr = adata.obs[sample_col].astype(str).values
    unique_samples = list(pd.Series(samples_arr).unique())
    
    # Per-sample centroid and bounding box
    sample_centroids = {}
    sample_sizes = {}
    for s in unique_samples:
        mask = (samples_arr == s)
        coords = spatial[mask]
        if len(coords) == 0:
            continue
        sample_centroids[s] = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        sample_sizes[s] = (float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])))
    
    max_w = max((sz[0] for sz in sample_sizes.values()), default=1)
    max_h = max((sz[1] for sz in sample_sizes.values()), default=1)
    cell_w = max_w * (1 + gap_x * 0.3)
    cell_h = max_h * (1 + gap_y * 0.3)
    
    # Group samples
    if group_col and group_col in adata.obs.columns:
        groups_ser = adata.obs.groupby(sample_col)[group_col].first()
        sample_to_group = groups_ser.astype(str).to_dict()
    else:
        sample_to_group = {s: "all" for s in unique_samples}
    
    group_names = sorted(set(sample_to_group.values()))
    
    # Sort samples within each group
    groups = {}
    for gname in group_names:
        members = [s for s in unique_samples if sample_to_group.get(s) == gname]
        if sort_col and sort_col in adata.obs.columns:
            sort_ser = adata.obs.groupby(sample_col)[sort_col].first()
            if order_col and order_col in adata.obs.columns:
                order_ser = adata.obs.groupby(sample_col)[order_col].first()
                sort_df = pd.DataFrame({"sort": sort_ser, "order": order_ser})
                sort_df = sort_df.loc[[s for s in members if s in sort_df.index]]
                sort_df = sort_df.sort_values(["sort", "order"])
                members = list(sort_df.index)
            else:
                members = sorted(members, key=lambda s: str(sort_ser.get(s, s)))
        else:
            members = sorted(members)
        groups[gname] = members
    
    # Grid geometry
    max_sample_rows = max(((len(v) + sample_ncols - 1) // sample_ncols for v in groups.values()), default=1)
    block_w = sample_ncols * cell_w
    block_h = max_sample_rows * cell_h
    group_gap_x = cell_w * 1.5
    group_gap_y = cell_h * 1.5
    
    sample_grid_pos = {}
    for gi, gname in enumerate(group_names):
        g_row, g_col = gi // group_ncols, gi % group_ncols
        gx0 = g_col * (block_w + group_gap_x)
        gy0 = g_row * (block_h + group_gap_y)
        for si, sname in enumerate(groups[gname]):
            s_row, s_col = si // sample_ncols, si % sample_ncols
            sample_grid_pos[sname] = (gx0 + s_col * cell_w + cell_w / 2,
                                       gy0 + s_row * cell_h + cell_h / 2)
    
    # Shift each cell
    new_coords = spatial.copy()
    sample_labels = []
    sample_label_positions = []
    for s in unique_samples:
        if s not in sample_grid_pos or s not in sample_centroids:
            continue
        mask = (samples_arr == s)
        cx, cy = sample_centroids[s]
        tx, ty = sample_grid_pos[s]
        new_coords[mask, 0] += (tx - cx)
        new_coords[mask, 1] += (ty - cy)
        sample_labels.append(s)
        sample_label_positions.append([float(tx), float(ty)])
    
    minX, maxX = float(new_coords[:, 0].min()), float(new_coords[:, 0].max())
    minY, maxY = float(new_coords[:, 1].min()), float(new_coords[:, 1].max())
    
    coords_binary = _pack_coords_binary(new_coords.astype(np.float32), compress=True)
    
    print(f"[Layout] {len(unique_samples)} samples, {len(group_names)} groups, grid {group_ncols} cols")
    return {
        "type": "layout_coords",
        "coords_binary": coords_binary,
        "count": len(new_coords),
        "bounds": {"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY, "count": len(new_coords)},
        "sample_labels": sample_labels,
        "sample_label_positions": sample_label_positions,
        "group_names": group_names,
        "groups": {g: groups[g] for g in group_names},
    }


def save_layout(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """Save layout coords to adata.obsm['X_layout_<n>'].
    
    Receives sample centroids from JS and reconstructs full cell positions
    by applying the same offset logic used in the viewer.
    """
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    name = data.get("name", "default")
    centroids_b64 = data.get("centroids_b64")
    sample_labels = data.get("sample_labels", [])
    n_samples = data.get("n_samples", 0)
    
    # Legacy support for old format
    if not centroids_b64:
        coords_b64 = data.get("coords_b64")
        if coords_b64:
            raw = base64.b64decode(coords_b64)
            try:
                raw = zlib.decompress(raw)
            except Exception:
                pass
            coords = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
            if coords.shape[0] != adata.n_obs:
                return {"type": "error", "message": f"Count mismatch: {coords.shape[0]} vs {adata.n_obs}"}
            key = name if name.startswith("X_") else f"X_{name}"
            adata.obsm[key] = coords
            print(f"[Layout] Saved '{name}' to adata.obsm['{key}'] shape={coords.shape}")
            return {"type": "layout_obsm_saved", "name": name, "key": key}
        return {"type": "error", "message": "No coordinates provided"}
    
    # New format: sample centroids + labels
    raw = base64.b64decode(centroids_b64)
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    
    centroids = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    print(f"[Layout] Received {centroids.shape[0]} sample centroids for '{name}'")
    
    if centroids.shape[0] != len(sample_labels):
        return {"type": "error", "message": f"Centroid/label mismatch: {centroids.shape[0]} vs {len(sample_labels)}"}
    
    sample_to_centroid = {label: centroids[i] for i, label in enumerate(sample_labels)}
    
    # Get sample IDs from adata
    sample_id_col = __sample_id__
    if sample_id_col is None or sample_id_col not in adata.obs.columns:
        for col in ['sample_id', 'sample', 'Sample', 'SampleID', 'sample_name', 'core_id']:
            if col in adata.obs.columns:
                sample_id_col = col
                break
    
    if sample_id_col is None or sample_id_col not in adata.obs.columns:
        return {"type": "error", "message": f"Sample ID column '{__sample_id__}' not found in adata.obs"}
    
    sample_ids = adata.obs[sample_id_col].astype(str).values
    
    # Get spatial coordinates
    spatial_key = None
    for key in ['X_spatial', 'spatial']:
        if key in adata.obsm:
            spatial_key = key
            break
    
    if spatial_key is None:
        return {"type": "error", "message": "No spatial coordinates found in adata.obsm"}
    
    spatial = np.asarray(adata.obsm[spatial_key])[:, :2].astype(np.float32)
    
    # Compute per-sample spatial centroids
    unique_samples = np.unique(sample_ids)
    sample_spatial_centroids = {}
    for s in unique_samples:
        mask = sample_ids == s
        sample_spatial_centroids[s] = spatial[mask].mean(axis=0)
    
    # Reconstruct full cell positions
    layout_coords = np.zeros_like(spatial)
    for i, (sid, sp) in enumerate(zip(sample_ids, spatial)):
        if sid in sample_to_centroid:
            layout_centroid = sample_to_centroid[sid]
            spatial_centroid = sample_spatial_centroids[sid]
            offset = layout_centroid - spatial_centroid
            layout_coords[i] = sp + offset
        else:
            layout_coords[i] = sp
    
    key = name if name.startswith("X_") else f"X_{name}"
    adata.obsm[key] = layout_coords
    print(f"[Layout] Saved '{name}' to adata.obsm['{key}'] shape={layout_coords.shape}")
    return {"type": "layout_obsm_saved", "name": name, "key": key}


def delete_layout(data: Dict, adata=None, __sample_idx=None, **kwargs) -> Dict:
    """Delete a layout from adata.obsm."""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    name = data.get("name", "")
    key = name if name.startswith("X_") else f"X_{name}"
    if key in adata.obsm:
        del adata.obsm[key]
        print(f"Deleted layout '{name}' from adata.obsm")
        return {"type": "layout_deleted", "name": name}
    return {"type": "error", "message": f"Layout '{name}' not found"}


def load_layout(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """Load a saved layout from adata.obsm and return binary coords."""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    name = data.get("name", "")
    key = name if name.startswith("X_") else f"X_{name}"
    if key not in adata.obsm:
        return {"type": "error", "message": f"Layout '{name}' not found"}
    
    coords = np.asarray(adata.obsm[key])[:, :2].astype(np.float32)
    minX, maxX = float(coords[:, 0].min()), float(coords[:, 0].max())
    minY, maxY = float(coords[:, 1].min()), float(coords[:, 1].max())
    
    sample_labels = []
    sample_label_positions = []
    sample_col = __sample_id__ or kwargs.get("__sample_id__")
    if sample_col and sample_col in adata.obs.columns:
        samps = adata.obs[sample_col].astype(str).values
        for s in pd.Series(samps).unique():
            mask = (samps == s)
            sample_labels.append(s)
            sample_label_positions.append([float(coords[mask, 0].mean()), float(coords[mask, 1].mean())])
    
    coords_binary = _pack_coords_binary(coords, compress=True)
    return {
        "type": "layout_coords",
        "coords_binary": coords_binary,
        "count": len(coords),
        "bounds": {"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY, "count": len(coords)},
        "sample_labels": sample_labels,
        "sample_label_positions": sample_label_positions,
        "layout_name": name,
    }
