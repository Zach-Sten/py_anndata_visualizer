"""
Spatial heatmap functions for the visualization interface.

Provides cell binning along a user-defined ribbon path and computes
mean gene expression per bin for spatial heatmap visualization.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np


def compute_heatmap_bins(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Given a ribbon path geometry and a list of genes, compute mean expression
    per bin along the ribbon.
    
    The ribbon is defined by a bezier center spine with variable width.
    We divide it into N bins along the spine, find cells in each bin,
    and compute mean expression for each gene.
    
    Expects data keys (via payload JSON string):
        - ribbon: {
            start: {x, y},
            end: {x, y},
            controlPoints: [{x, y}, {x, y}],  # bezier control points
            widthStart: float,
            widthMid: float,
            widthEnd: float,
          }
        - genes: [gene_name, ...]
        - numBins: int (default 15)
        - enabledCategories: optional filter — {column, enabled: [cat1, cat2, ...]}
        - embedding: optional embedding key for coordinates
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    # Parse payload
    payload_str = data.get("payload", "{}")
    if isinstance(payload_str, str):
        try:
            payload = json.loads(payload_str)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse heatmap data: {e}"}
    else:
        payload = payload_str
    
    ribbon = payload.get("ribbon", {})
    genes = payload.get("genes", [])
    num_bins = int(payload.get("numBins", 15))
    enabled_filter = payload.get("enabledCategories", None)
    embedding_key = payload.get("embedding", None)
    
    if not ribbon:
        return {"type": "error", "message": "No ribbon geometry provided"}
    if not genes:
        return {"type": "error", "message": "No genes specified"}
    
    # Get spatial coordinates
    spatial = _get_coordinates(adata, embedding_key)
    if spatial is None:
        return {"type": "error", "message": "No valid coordinate embedding found"}
    
    # Apply category filter if provided (respects Color By toggles)
    cell_mask = np.ones(adata.n_obs, dtype=bool)
    if enabled_filter:
        col = enabled_filter.get("column")
        enabled_cats = enabled_filter.get("enabled", [])
        if col and col in adata.obs.columns and enabled_cats:
            cell_mask = adata.obs[col].astype(str).isin(enabled_cats).values
    
    # Parse ribbon geometry
    sx, sy = ribbon["start"]["x"], ribbon["start"]["y"]
    ex, ey = ribbon["end"]["x"], ribbon["end"]["y"]
    cp = ribbon.get("controlPoints", [])
    cp1 = cp[0] if len(cp) > 0 else {"x": sx + (ex - sx) / 3, "y": sy + (ey - sy) / 3}
    cp2 = cp[1] if len(cp) > 1 else {"x": sx + 2 * (ex - sx) / 3, "y": sy + 2 * (ey - sy) / 3}
    
    w_start = float(ribbon.get("widthStart", 50))
    w_mid = float(ribbon.get("widthMid", 50))
    w_end = float(ribbon.get("widthEnd", 50))
    
    # Generate points along the bezier spine
    # Cubic bezier: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    t_values = np.linspace(0, 1, num_bins + 1)
    t_mids = (t_values[:-1] + t_values[1:]) / 2  # bin centers
    
    p0 = np.array([sx, sy])
    p1 = np.array([cp1["x"], cp1["y"]])
    p2 = np.array([cp2["x"], cp2["y"]])
    p3 = np.array([ex, ey])
    
    # Compute bin boundaries along the spine
    bin_edges = []
    for t in t_values:
        pt = _bezier_point(t, p0, p1, p2, p3)
        tangent = _bezier_tangent(t, p0, p1, p2, p3)
        normal = np.array([-tangent[1], tangent[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        
        # Interpolate width: quadratic through start, mid, end
        w = _interpolate_width(t, w_start, w_mid, w_end)
        
        bin_edges.append({
            "point": pt,
            "normal": normal,
            "width": w,
            "t": t,
        })
    
    # For each bin, create a quadrilateral from adjacent edges and find cells inside
    bin_results = []
    bin_cell_indices = []
    
    for i in range(num_bins):
        edge_a = bin_edges[i]
        edge_b = bin_edges[i + 1]
        
        # Quad corners: top-left, top-right, bottom-right, bottom-left
        half_wa = edge_a["width"] / 2
        half_wb = edge_b["width"] / 2
        
        quad = np.array([
            edge_a["point"] + edge_a["normal"] * half_wa,
            edge_b["point"] + edge_b["normal"] * half_wb,
            edge_b["point"] - edge_b["normal"] * half_wb,
            edge_a["point"] - edge_a["normal"] * half_wa,
        ])
        
        # Find cells inside this quad (using point-in-polygon)
        inside = _points_in_quad(spatial[:, :2], quad) & cell_mask
        cell_indices = np.where(inside)[0].tolist()
        bin_cell_indices.append(cell_indices)
        
        # Compute centroid of the bin on the spine (for labeling)
        t_center = (edge_a["t"] + edge_b["t"]) / 2
        center_pt = _bezier_point(t_center, p0, p1, p2, p3)
        
        bin_results.append({
            "bin": i,
            "cell_count": len(cell_indices),
            "center_x": float(center_pt[0]),
            "center_y": float(center_pt[1]),
            "quad": quad.tolist(),
        })
    
    # Compute mean expression for each gene in each bin
    heatmap_data = {}
    for gene in genes:
        if gene not in adata.var_names:
            print(f"[Heatmap] Gene '{gene}' not found, skipping")
            continue
        
        gene_idx = list(adata.var_names).index(gene)
        
        row = []
        for cell_indices in bin_cell_indices:
            if len(cell_indices) == 0:
                row.append(0.0)
            else:
                # Handle sparse and dense matrices
                expr = adata.X[cell_indices, gene_idx]
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.asarray(expr).flatten()
                row.append(float(np.mean(expr)))
        
        heatmap_data[gene] = row
    
    total_cells = sum(b["cell_count"] for b in bin_results)
    print(f"[Heatmap] Computed {len(heatmap_data)} genes × {num_bins} bins "
          f"({total_cells} total cells)")
    
    return {
        "type": "heatmap_result",
        "genes": list(heatmap_data.keys()),
        "heatmap": heatmap_data,
        "bins": bin_results,
        "numBins": num_bins,
        "totalCells": total_cells,
    }


def _get_coordinates(adata, embedding_key=None):
    """Get 2D coordinates from adata."""
    if embedding_key and embedding_key in adata.obsm:
        return np.array(adata.obsm[embedding_key])[:, :2]
    for key in ["spatial", "X_spatial", "X_umap", "X_pca"]:
        if key in adata.obsm:
            return np.array(adata.obsm[key])[:, :2]
    return None


def _bezier_point(t, p0, p1, p2, p3):
    """Evaluate cubic bezier at parameter t."""
    u = 1 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def _bezier_tangent(t, p0, p1, p2, p3):
    """Compute tangent of cubic bezier at parameter t."""
    u = 1 - t
    tangent = (3 * u**2 * (p1 - p0) +
               6 * u * t * (p2 - p1) +
               3 * t**2 * (p3 - p2))
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm
    return tangent


def _interpolate_width(t, w_start, w_mid, w_end):
    """
    Quadratic interpolation of width through three control values.
    w(0) = w_start, w(0.5) = w_mid, w(1) = w_end
    Using Lagrange interpolation through (0, w_start), (0.5, w_mid), (1, w_end)
    """
    # Lagrange basis polynomials evaluated at t:
    # L0(t) = (t - 0.5)(t - 1) / ((0 - 0.5)(0 - 1)) = (t-0.5)(t-1) / 0.5
    # L1(t) = (t - 0)(t - 1) / ((0.5 - 0)(0.5 - 1)) = t(t-1) / (-0.25)
    # L2(t) = (t - 0)(t - 0.5) / ((1 - 0)(1 - 0.5)) = t(t-0.5) / 0.5
    L0 = (t - 0.5) * (t - 1.0) / 0.5
    L1 = t * (t - 1.0) / (-0.25)
    L2 = t * (t - 0.5) / 0.5
    return w_start * L0 + w_mid * L1 + w_end * L2


def _points_in_quad(points, quad):
    """
    Test which points are inside a convex quadrilateral.
    Uses cross-product winding test.
    
    quad: (4, 2) array of vertices in order
    points: (N, 2) array of test points
    """
    n = len(quad)
    inside = np.ones(len(points), dtype=bool)
    
    for i in range(n):
        edge_start = quad[i]
        edge_end = quad[(i + 1) % n]
        
        # Cross product of edge vector with point-to-start vector
        edge = edge_end - edge_start
        to_point = points - edge_start
        cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
        
        # All points should be on the same side (left/right) for convex polygon
        # We check that cross product is >= 0 (left side, counterclockwise winding)
        # But quad might be clockwise, so we check consistency
        if i == 0:
            sign = cross >= 0
            inside &= sign
        else:
            inside &= (cross >= 0) == sign[0] if np.any(sign) else (cross >= 0)
    
    # Simpler approach: use the standard winding test
    # Re-implement with proper sign handling
    inside = _robust_point_in_quad(points, quad)
    return inside


def _robust_point_in_quad(points, quad):
    """
    Robust point-in-convex-polygon test using cross products.
    Returns boolean array of which points are inside the quad.
    """
    n = len(quad)
    
    # Compute cross products for all edges
    crosses = []
    for i in range(n):
        v1 = quad[(i + 1) % n] - quad[i]
        v2 = points - quad[i]
        cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
        crosses.append(cross)
    
    # Point is inside if all cross products have the same sign
    crosses = np.array(crosses)  # (4, N)
    all_pos = np.all(crosses >= 0, axis=0)
    all_neg = np.all(crosses <= 0, axis=0)
    return all_pos | all_neg
