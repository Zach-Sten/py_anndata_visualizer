"""
Region detection functions for the visualization interface.

Provides DBSCAN-based spatial clustering and alpha shape boundary
computation for defining regions from cell type selections.
"""

import base64
import json
import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Coordinate helpers
# =============================================================================

def _get_embedding_coords(adata, embedding: str = None):
    """
    Return (n_cells, 2) array for the requested embedding key.

    Resolution order:
      1. Try the key as-is in adata.obsm           (e.g. "spatial", "X_umap")
      2. Try with "X_" prepended                    (e.g. "layout" → "X_layout")
      3. Fall back to 'spatial' / 'X_spatial'

    Returns None if no valid key is found.
    """
    if embedding:
        if embedding in adata.obsm:
            return np.asarray(adata.obsm[embedding])[:, :2]
        xkey = embedding if embedding.startswith("X_") else "X_" + embedding
        if xkey in adata.obsm:
            return np.asarray(adata.obsm[xkey])[:, :2]
        print(f"[Regions] Embedding '{embedding}' not found in adata.obsm — falling back to spatial")

    for key in ["spatial", "X_spatial"]:
        if key in adata.obsm:
            return np.asarray(adata.obsm[key])[:, :2]
    return None


# =============================================================================
# Region Functions (DBSCAN + Alpha Shape)
# =============================================================================

def run_dbscan(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Run DBSCAN clustering one sample at a time, streaming progress back to JS.

    First call: supply column/category/eps/min_samples/embedding. Python
    determines the full sample list, processes the first sample, and returns a
    'dbscan_progress' response. JS fires continuation calls (continue=true)
    until Python returns a final 'dbscan_result'.

    Full cluster indices are cached in adata.uns['_dbscan_tmp'] — never sent
    through the bridge.
    """
    from sklearn.cluster import DBSCAN
    import json as _json

    if adata is None:
        return {"type": "error", "message": "No adata provided"}

    is_continuation = data.get("continue", False)

    # ── FRESH RUN: validate inputs and build the sample queue ──────────────────
    if not is_continuation:
        column = data.get("column", "")
        category = data.get("category", "")
        eps = float(data.get("eps", 30))
        min_samples = int(data.get("min_samples", 20))
        embedding = data.get("embedding", None)

        all_cells_mode = (column == "all_cells" or not column)

        if not all_cells_mode and (not column or column not in adata.obs.columns):
            return {"type": "error", "message": f"Column '{column}' not found in adata.obs"}
        if not all_cells_mode and not category:
            return {"type": "error", "message": "No category specified"}

        spatial = _get_embedding_coords(adata, embedding)
        if spatial is None:
            return {"type": "error", "message": "No spatial coordinates found"}

        if all_cells_mode:
            type_mask = np.ones(adata.n_obs, dtype=bool)
            category = "all_cells"
        else:
            obs_vals = adata.obs[column].astype(str).values
            type_mask = (obs_vals == str(category))
            if type_mask.sum() == 0:
                return {"type": "error", "message": f"No cells found for {column}='{category}'"}

        sample_col = __sample_id__
        if sample_col and sample_col in adata.obs.columns:
            sample_ids = adata.obs[sample_col].astype(str).values
        else:
            sample_col = None
            sample_ids = np.array(["all"] * adata.n_obs)

        pending_samples = np.unique(sample_ids[type_mask]).tolist()
        print(f"[Regions] DBSCAN starting: {len(pending_samples)} samples, embedding='{embedding}'")

        # Initialise fresh state and cluster cache
        adata.uns["_dbscan_tmp"] = _json.dumps({})
        adata.uns["_dbscan_state"] = _json.dumps({
            "pending": pending_samples,
            "params": {
                "column": column, "category": category,
                "eps": eps, "min_samples": min_samples,
                "embedding": embedding, "sample_col": sample_col,
                "all_cells_mode": all_cells_mode,
            },
            "total_samples": len(pending_samples),
            "done_count": 0,
            "noise_count": 0,
            "cluster_info": [],
        })

    # ── LOAD STATE ─────────────────────────────────────────────────────────────
    raw = adata.uns.get("_dbscan_state")
    if raw is None:
        return {"type": "error", "message": "No DBSCAN state — please re-run DBSCAN"}
    state = _json.loads(raw) if isinstance(raw, str) else raw
    p = state["params"]

    # Reload spatial coords and masks from adata (avoids storing large arrays)
    spatial = _get_embedding_coords(adata, p["embedding"])
    if spatial is None:
        return {"type": "error", "message": "No spatial coordinates found"}

    if p["all_cells_mode"]:
        type_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        type_mask = (adata.obs[p["column"]].astype(str).values == str(p["category"]))

    sc = p.get("sample_col")
    sample_ids = (adata.obs[sc].astype(str).values
                  if sc and sc in adata.obs.columns
                  else np.array(["all"] * adata.n_obs))

    # ── PROCESS NEXT SAMPLE ────────────────────────────────────────────────────
    pending = state["pending"]
    sample = pending.pop(0)

    sample_mask = type_mask & (sample_ids == sample)
    global_indices = np.where(sample_mask)[0]

    sample_cluster_info = []
    sample_noise = 0

    raw_tmp = adata.uns.get("_dbscan_tmp", "{}")
    cached = _json.loads(raw_tmp) if isinstance(raw_tmp, str) else raw_tmp

    if len(global_indices) >= p["min_samples"]:
        coords = spatial[global_indices]
        db = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit(coords)
        labels = db.labels_
        unique_labels = sorted(set(labels) - {-1})
        sample_noise = int((labels == -1).sum())

        print(f"[Regions] Sample '{sample}': {len(global_indices)} cells → "
              f"{len(unique_labels)} clusters, {sample_noise} noise")

        for label in unique_labels:
            name = f"{sample}_{label}_{p['category']}"
            cell_indices = global_indices[labels == label].tolist()
            cached[name] = cell_indices
            cx = float(spatial[cell_indices, 0].mean())
            cy = float(spatial[cell_indices, 1].mean())
            info = {"name": name, "count": len(cell_indices), "centroid_x": cx, "centroid_y": cy}
            sample_cluster_info.append(info)
            state["cluster_info"].append(info)
    else:
        print(f"[Regions] Skipping '{sample}': {len(global_indices)} cells < min_samples={p['min_samples']}")

    state["done_count"] += 1
    state["noise_count"] = state.get("noise_count", 0) + sample_noise
    state["pending"] = pending

    adata.uns["_dbscan_tmp"] = _json.dumps(cached)
    adata.uns["_dbscan_state"] = _json.dumps(state)

    # ── MORE SAMPLES REMAINING → progress response ─────────────────────────────
    if pending:
        return {
            "type": "dbscan_progress",
            "sample": sample,
            "clusters_this_sample": sample_cluster_info,
            "sample_idx": state["done_count"],
            "total_samples": state["total_samples"],
            "noise_this_sample": sample_noise,
        }

    # ── ALL DONE → final result ────────────────────────────────────────────────
    total_clusters = len(state["cluster_info"])
    if total_clusters == 0:
        return {
            "type": "error",
            "message": f"DBSCAN found no clusters. Try lowering eps or min_samples. "
                       f"({state['noise_count']} cells classified as noise)",
        }

    print(f"[Regions] DBSCAN complete: {total_clusters} clusters across "
          f"{state['total_samples']} samples (noise={state['noise_count']})")

    return {
        "type": "dbscan_result",
        "column": p["column"],
        "category": p["category"],
        "eps": p["eps"],
        "min_samples": p["min_samples"],
        "clusters": state["cluster_info"],
        "total_clusters": total_clusters,
        "noise_count": state["noise_count"],
    }


def compute_alpha_shapes(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Compute alpha shape boundaries for DBSCAN clusters.

    Expects data keys:
        - clusters: list of {name} — indices are looked up from adata.uns['_dbscan_tmp']
        - alpha: alpha parameter for alphashape (float).
                 Higher = tighter fit, lower = looser/more convex.

    Returns polygon vertices per cluster. Includes a capped sample of indices
    (≤ _MAX_CANVAS_INDICES) for centroid-tracking in the canvas; full indices
    are stored server-side in adata.uns['_dbscan_tmp'].
    """
    import json as _json
    import alphashape
    from shapely.geometry import Polygon, MultiPolygon

    _MAX_CANVAS_INDICES = 2000  # cap sent to JS for translateRegionPolygons

    if adata is None:
        return {"type": "error", "message": "No adata provided"}

    clusters_data = data.get("clusters", [])
    alpha_val = float(data.get("alpha", 0.05))
    embedding = data.get("embedding", None)

    if not clusters_data:
        return {"type": "error", "message": "No clusters provided"}

    # Load full indices from server-side cache
    raw_tmp = adata.uns.get("_dbscan_tmp")
    if raw_tmp is None:
        return {"type": "error", "message": "No DBSCAN results cached — please re-run DBSCAN"}
    try:
        cached_clusters = _json.loads(raw_tmp) if isinstance(raw_tmp, str) else raw_tmp
    except Exception as e:
        return {"type": "error", "message": f"Failed to read DBSCAN cache: {e}"}

    # Get coordinates for the active embedding
    spatial = _get_embedding_coords(adata, embedding)
    if spatial is None:
        return {"type": "error", "message": "No spatial coordinates found"}

    regions = []
    failed = []

    for cluster in clusters_data:
        name = cluster["name"]
        indices = cached_clusters.get(name)
        if indices is None:
            failed.append({"name": name, "reason": "Cluster not found in DBSCAN cache — re-run DBSCAN"})
            continue
        
        if len(indices) < 3:
            failed.append({"name": name, "reason": "Too few points (< 3)"})
            continue
        
        coords = spatial[indices]
        
        try:
            shape = alphashape.alphashape(coords, alpha_val)
            
            if shape is None or shape.is_empty:
                failed.append({"name": name, "reason": "Alpha shape is empty (try lower alpha)"})
                continue
            
            # Extract polygon(s) — could be Polygon or MultiPolygon
            polygons = []
            if isinstance(shape, Polygon):
                if shape.is_valid and not shape.is_empty:
                    # Exterior ring as list of [x, y]
                    exterior = list(shape.exterior.coords)
                    polygons.append(exterior)
                    # Include holes if any
                    for interior in shape.interiors:
                        polygons.append(list(interior.coords))
            elif isinstance(shape, MultiPolygon):
                for poly in shape.geoms:
                    if poly.is_valid and not poly.is_empty:
                        exterior = list(poly.exterior.coords)
                        polygons.append(exterior)
                        for interior in poly.interiors:
                            polygons.append(list(interior.coords))
            else:
                # LineString or other — skip
                failed.append({"name": name, "reason": f"Unexpected shape type: {type(shape).__name__}"})
                continue
            
            if not polygons:
                failed.append({"name": name, "reason": "No valid polygons extracted"})
                continue
            
            # Convert polygon coords to flat lists for JSON
            polygon_data = []
            for poly_coords in polygons:
                polygon_data.append([[float(x), float(y)] for x, y in poly_coords])

            # Capped evenly-spaced sample sent to JS for translateRegionPolygons.
            # IMPORTANT: centroid_x/y must be computed from canvas_indices, not
            # from all cells. translateRegionPolygons computes the centroid from
            # canvas_indices on the JS side; if the stored centroid was computed
            # from all cells the delta would be nonzero even on the same embedding,
            # shifting the polygon away from its cells.
            n = len(indices)
            if n > _MAX_CANVAS_INDICES:
                step = max(1, n // _MAX_CANVAS_INDICES)
                canvas_indices = indices[::step][:_MAX_CANVAS_INDICES]
                canvas_coords = spatial[canvas_indices]
            else:
                canvas_indices = indices
                canvas_coords = coords

            regions.append({
                "name": name,
                "indices": canvas_indices,
                "polygons": polygon_data,
                "centroid_x": float(canvas_coords[:, 0].mean()),
                "centroid_y": float(canvas_coords[:, 1].mean()),
            })
            
        except Exception as e:
            failed.append({"name": name, "reason": str(e)})
            continue
    
    if not regions and failed:
        fail_reasons = "; ".join(f"{f['name']}: {f['reason']}" for f in failed[:5])
        return {
            "type": "error",
            "message": f"All alpha shapes failed. {fail_reasons}. Try adjusting alpha."
        }
    
    print(f"[Regions] Alpha shapes: {len(regions)} succeeded, {len(failed)} failed (alpha={alpha_val})")
    if failed:
        for f in failed[:3]:
            print(f"  ⚠ {f['name']}: {f['reason']}")
    
    return {
        "type": "alpha_shapes",
        "regions": regions,
        "failed": failed,
        "alpha": alpha_val,
        "total_regions": len(regions),
    }


def save_region_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Save region masks (polygon geometries + cell indices) to adata.uns['region_masks'].
    
    Expects data keys:
        - groups: dict of { groupName: { selections: [regionName, ...], expanded, visible } }
        - regions: dict of { regionName: { indices: [...], visible, tool } }
        - polygons: list of { name, polygons: [[[x,y],...]], centroid_x, centroid_y }
        - metadata: { column, fill_opacity }
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    # Data comes as a single JSON payload to avoid bridge serialization issues
    payload_str = data.get("payload", "{}")
    if isinstance(payload_str, str):
        import json as _json
        try:
            payload = _json.loads(payload_str)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse region mask data: {e}"}
    else:
        payload = payload_str
    
    groups_data = payload.get("groups", {})
    regions_data = payload.get("regions", {})
    polygons_data = payload.get("polygons", [])
    metadata = payload.get("metadata", {})

    # Load full indices from server-side DBSCAN cache (JS only has a capped sample)
    import json as _json
    raw_tmp = adata.uns.get("_dbscan_tmp")
    cached_indices = {}
    if raw_tmp is not None:
        try:
            cached_indices = _json.loads(raw_tmp) if isinstance(raw_tmp, str) else raw_tmp
        except Exception:
            pass

    # Build a clean structure for storage
    masks_store = {
        "metadata": {
            "column": metadata.get("column"),
            "fill_opacity": metadata.get("fill_opacity", 0.1),
            "outline_weight": metadata.get("outline_weight", 2),
            "alpha": metadata.get("alpha", 0.05),
            "sample_id": __sample_id__,
        },
        "groups": {},
        "regions": {},
    }

    for gname, gdata in groups_data.items():
        masks_store["groups"][gname] = {
            "selections": gdata.get("selections", []),
            "expanded": gdata.get("expanded", True),
            "visible": gdata.get("visible", True),
        }

    # Build a lookup from polygon name to polygon data
    poly_lookup = {}
    for p in polygons_data:
        poly_lookup[p["name"]] = {
            "polygons": p.get("polygons", []),
            "centroid_x": p.get("centroid_x"),
            "centroid_y": p.get("centroid_y"),
        }

    # Store regions with full indices (from cache) and polygon geometry
    for rname, rdata in regions_data.items():
        # Prefer full indices from server-side cache; fall back to whatever JS sent
        full_indices = cached_indices.get(rname) or rdata.get("indices", [])
        region_entry = {
            "indices": full_indices,
            "visible": rdata.get("visible", True),
            "tool": rdata.get("tool", "region"),
        }
        # Attach polygon data if available
        if rname in poly_lookup:
            region_entry["polygons"] = poly_lookup[rname]["polygons"]
            region_entry["centroid_x"] = poly_lookup[rname]["centroid_x"]
            region_entry["centroid_y"] = poly_lookup[rname]["centroid_y"]

        masks_store["regions"][rname] = region_entry
    
    # Save to adata.uns as a JSON string (avoids h5ad ragged array issues with polygon data)
    import json as _json
    adata.uns["region_masks"] = _json.dumps(masks_store)
    
    total_regions = len(masks_store["regions"])
    total_groups = len(masks_store["groups"])
    total_cells = sum(len(r.get("indices", [])) for r in masks_store["regions"].values())
    
    print(f"[Regions] Saved {total_regions} region masks ({total_groups} groups, {total_cells} cells) to adata.uns['region_masks']")
    
    return {
        "type": "region_masks_saved",
        "total_regions": total_regions,
        "total_groups": total_groups,
        "total_cells": total_cells,
    }


def load_region_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Load region masks from adata.uns['region_masks'] and return them to the UI.
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    raw = adata.uns.get("region_masks")
    if raw is None:
        return {"type": "error", "message": "No saved region masks found in adata.uns['region_masks']"}
    
    # Parse JSON string (saved as string to avoid h5ad ragged array issues)
    import json as _json
    if isinstance(raw, str):
        try:
            masks_store = _json.loads(raw)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse region masks: {e}"}
    else:
        # Legacy: might be a dict if not yet re-saved
        masks_store = raw
    
    print(f"[Regions] Loading region masks from adata.uns['region_masks']")
    
    groups = masks_store.get("groups", {})
    regions = masks_store.get("regions", {})
    metadata = masks_store.get("metadata", {})
    
    # Rebuild polygon list for the canvas
    polygons_for_canvas = []
    for rname, rdata in regions.items():
        if "polygons" in rdata and rdata["polygons"]:
            polygons_for_canvas.append({
                "name": rname,
                "polygons": rdata["polygons"],
                "centroid_x": rdata.get("centroid_x"),
                "centroid_y": rdata.get("centroid_y"),
                "indices": rdata.get("indices", []),
            })
    
    total_regions = len(regions)
    total_groups = len(groups)
    print(f"[Regions] Loaded {total_regions} regions in {total_groups} groups")
    
    return {
        "type": "region_masks_loaded",
        "groups": groups,
        "regions": {rname: {"indices": r.get("indices", []), "visible": r.get("visible", True), "tool": r.get("tool", "region")} for rname, r in regions.items()},
        "polygons": polygons_for_canvas,
        "metadata": metadata,
        "total_regions": total_regions,
        "total_groups": total_groups,
    }


def recompute_region_polygons(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Recompute alpha shape polygons for existing regions using current coordinates.
    
    Called when layout/embedding changes so region masks track their cells.
    
    Expects data keys:
        - regions: list of { name, indices } — the region cell indices
        - alpha: alpha parameter for alpha shapes
        - embedding: current embedding key (e.g., "spatial", "X_umap") or null for current
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    from alphashape import alphashape as compute_alphashape
    from shapely.geometry import Polygon, MultiPolygon
    
    # Parse payload (same pattern as save)
    payload_str = data.get("payload", "{}")
    if isinstance(payload_str, str):
        import json as _json
        try:
            payload = _json.loads(payload_str)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse recompute data: {e}"}
    else:
        payload = payload_str
    
    regions_list = payload.get("regions", [])
    alpha_val = float(payload.get("alpha", 0.05))
    embedding = payload.get("embedding", None)
    
    # Get the coordinates from the current embedding
    if embedding and embedding in adata.obsm:
        spatial = np.array(adata.obsm[embedding])[:, :2]
    elif "spatial" in adata.obsm:
        spatial = np.array(adata.obsm["spatial"])[:, :2]
    elif "X_spatial" in adata.obsm:
        spatial = np.array(adata.obsm["X_spatial"])[:, :2]
    else:
        # Try the first available embedding
        for key in adata.obsm.keys():
            if adata.obsm[key].shape[1] >= 2:
                spatial = np.array(adata.obsm[key])[:, :2]
                break
        else:
            return {"type": "error", "message": "No valid coordinate embedding found"}
    
    print(f"[Regions] Recomputing {len(regions_list)} region polygons with alpha={alpha_val}")
    
    results = []
    failed = []
    
    for region in regions_list:
        name = region["name"]
        indices = region["indices"]
        
        if len(indices) < 3:
            failed.append({"name": name, "reason": f"Too few cells ({len(indices)})"})
            continue
        
        coords = spatial[indices]
        
        try:
            shape = compute_alphashape(coords, alpha=alpha_val)
            
            if shape is None or shape.is_empty:
                failed.append({"name": name, "reason": "Empty alpha shape"})
                continue
            
            polygons_list = []
            geoms = [shape] if isinstance(shape, Polygon) else (shape.geoms if isinstance(shape, MultiPolygon) else [])
            
            for geom in geoms:
                if isinstance(geom, Polygon) and not geom.is_empty:
                    exterior = list(geom.exterior.coords)
                    polygons_list.append([[float(x), float(y)] for x, y in exterior])
            
            if polygons_list:
                cx = float(coords[:, 0].mean())
                cy = float(coords[:, 1].mean())
                results.append({
                    "name": name,
                    "polygons": polygons_list,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "indices": indices,
                })
            else:
                failed.append({"name": name, "reason": "No valid polygons"})
                
        except Exception as e:
            failed.append({"name": name, "reason": str(e)})
    
    print(f"[Regions] Recomputed: {len(results)} succeeded, {len(failed)} failed")
    
    return {
        "type": "region_polygons_recomputed",
        "regions": results,
        "failed": failed,
        "alpha": alpha_val,
        "total_regions": len(results),
    }


# ────────────────────────────────────────────────────────────
#  Manual Selection Masks — save/load to adata.uns['manual_masks']
# ────────────────────────────────────────────────────────────

def save_manual_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Save manual selection masks (cell indices per selection) to adata.uns['manual_masks'].
    Supports delta-encoded indices (compressed=true) for efficient widget bridge transport.
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    payload_str = data.get("payload", "{}")
    print(f"[Manual] Raw payload length: {len(payload_str) if isinstance(payload_str, str) else 'not string'}")
    
    if isinstance(payload_str, str):
        import json as _json
        try:
            payload = _json.loads(payload_str)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse manual mask data: {e}"}
    else:
        payload = payload_str
    
    groups_data = payload.get("groups", {})
    selections_data = payload.get("selections", {})
    compressed = payload.get("compressed", False)
    
    print(f"[Manual] Parsed payload: {len(groups_data)} groups, {len(selections_data)} selections, compressed={compressed}")
    
    masks_store = {
        "metadata": {
            "sample_id": __sample_id__,
        },
        "groups": {},
        "selections": {},
    }
    
    for gname, gdata in groups_data.items():
        masks_store["groups"][gname] = {
            "selections": gdata.get("selections", []),
            "expanded": gdata.get("expanded", True),
        }
    
    for sname, sdata in selections_data.items():
        if compressed and "deltas" in sdata:
            # Decode delta-encoded indices
            deltas_str = sdata["deltas"]
            if deltas_str:
                deltas = [int(x) for x in deltas_str.split(",")]
                indices = []
                running = 0
                for d in deltas:
                    running += d
                    indices.append(running)
                # Fix: first delta IS the first index, rest are differences
                indices = []
                val = 0
                for i, d in enumerate(deltas):
                    if i == 0:
                        val = d
                    else:
                        val += d
                    indices.append(val)
            else:
                indices = []
        else:
            indices = sdata.get("indices", [])
        
        masks_store["selections"][sname] = {
            "indices": indices,
            "tool": sdata.get("tool", "manual"),
        }
    
    import json as _json
    adata.uns["manual_masks"] = _json.dumps(masks_store)
    
    total_selections = len(masks_store["selections"])
    total_groups = len(masks_store["groups"])
    total_cells = sum(len(s.get("indices", [])) for s in masks_store["selections"].values())
    
    print(f"[Manual] Saved {total_selections} selections ({total_groups} groups, {total_cells} cells) to adata.uns['manual_masks']")
    
    return {
        "type": "manual_masks_saved",
        "total_selections": total_selections,
        "total_groups": total_groups,
        "total_cells": total_cells,
    }


def load_manual_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Load manual selection masks from adata.uns['manual_masks'] and return them to the UI.
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}
    
    raw = adata.uns.get("manual_masks")
    if raw is None:
        return {"type": "error", "message": "No saved manual masks found in adata.uns['manual_masks']"}
    
    import json as _json
    if isinstance(raw, str):
        try:
            masks_store = _json.loads(raw)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse manual masks: {e}"}
    else:
        masks_store = raw
    
    print(f"[Manual] Loading manual masks from adata.uns['manual_masks']")
    
    groups = masks_store.get("groups", {})
    selections = masks_store.get("selections", {})
    
    total_selections = len(selections)
    total_groups = len(groups)
    print(f"[Manual] Loaded {total_selections} selections in {total_groups} groups")
    
    return {
        "type": "manual_masks_loaded",
        "groups": groups,
        "selections": {sname: {"indices": s.get("indices", []), "tool": s.get("tool", "manual")} for sname, s in selections.items()},
        "total_selections": total_selections,
        "total_groups": total_groups,
    }
