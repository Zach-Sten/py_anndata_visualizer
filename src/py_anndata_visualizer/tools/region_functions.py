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

        # Load any existing cluster cache so regions from prior runs survive
        existing_tmp = adata.uns.get("_dbscan_tmp", "{}")
        adata.uns["_dbscan_tmp"] = existing_tmp if isinstance(existing_tmp, str) else _json.dumps(existing_tmp)
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
                "count": len(indices),  # full cell count (indices may be capped for canvas)
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
            "embedding": data.get("embedding"),  # coordinate space the polygons were computed in
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
        "uns_key": "region_masks",
        "total_regions": total_regions,
        "total_groups": total_groups,
        "total_cells": total_cells,
    }


def _detect_stored_embedding(polygons_for_canvas, full_indices, adata):
    """
    Heuristic: guess the embedding that stored polygon vertices are in by comparing
    the first polygon's centroid to the coordinate ranges of all obsm keys.
    Falls back to "spatial" if unable to determine.
    """
    if not polygons_for_canvas:
        return "spatial"

    # Pick a sample polygon centroid
    entry = polygons_for_canvas[0]
    if not entry.get("polygons"):
        return "spatial"
    verts = np.array(entry["polygons"][0])
    cx, cy = verts[:, 0].mean(), verts[:, 1].mean()

    best_key = "spatial"
    best_dist = float("inf")
    name = entry["name"]
    indices = full_indices.get(name, [])
    if not indices:
        return "spatial"

    for key in adata.obsm.keys():
        coords = np.asarray(adata.obsm[key])
        if coords.ndim < 2 or coords.shape[1] < 2:
            continue
        idx_arr = np.asarray(indices[:500], dtype=int)
        idx_arr = idx_arr[idx_arr < coords.shape[0]]
        if idx_arr.size == 0:
            continue
        center = coords[idx_arr, :2].mean(axis=0)
        dist = (cx - center[0])**2 + (cy - center[1])**2
        if dist < best_dist:
            best_dist = dist
            best_key = key

    print(f"[Regions] Detected stored polygon embedding: '{best_key}' (centroid dist²={best_dist:.1f})")
    return best_key


def _transform_polygons_to_embedding(polygons_for_canvas, full_indices, adata,
                                      source_embedding, target_embedding):
    """
    Translate stored polygon vertices from source_embedding → target_embedding space.

    For each region, the per-region translation offset is:
        offset = mean(target_coords[indices]) - mean(source_coords[indices])
    This correctly handles any source/target embedding pair (spatial, layout, umap, etc.).
    If source == target, no transform is applied.
    """
    if source_embedding == target_embedding:
        return polygons_for_canvas

    source_coords = _get_embedding_coords(adata, source_embedding)
    target_coords = _get_embedding_coords(adata, target_embedding)

    if source_coords is None or target_coords is None:
        print(f"[Regions] Cannot transform polygons: missing '{source_embedding}' or '{target_embedding}' coords")
        return polygons_for_canvas

    n_cells = min(source_coords.shape[0], target_coords.shape[0])

    transformed = []
    for entry in polygons_for_canvas:
        name = entry["name"]
        indices = full_indices.get(name, [])
        if not indices or not entry.get("polygons"):
            transformed.append(entry)
            continue

        idx_arr = np.asarray(indices, dtype=int)
        idx_arr = idx_arr[idx_arr < n_cells]
        if idx_arr.size == 0:
            transformed.append(entry)
            continue

        src_center = source_coords[idx_arr].mean(axis=0)
        tg_center = target_coords[idx_arr].mean(axis=0)
        offset = tg_center - src_center

        new_polygons = []
        for ring in entry["polygons"]:
            new_ring = [[v[0] + float(offset[0]), v[1] + float(offset[1])] for v in ring]
            new_polygons.append(new_ring)

        transformed.append({
            **entry,
            "polygons": new_polygons,
            "centroid_x": float(tg_center[0]),
            "centroid_y": float(tg_center[1]),
        })

    return transformed


def load_region_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Load region masks from adata.uns and return all at once.

    data["source"] picks the uns key; format is auto-detected by content.
    data["embedding"] is the current embedding key from JS (used to transform
    polygon vertices from spatial space into the current view's coordinate space).
    JS handles progressive rendering client-side for the progress bar.
    """
    import json as _json

    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    source = data.get("source", "region_masks")
    embedding = data.get("embedding", None)  # current view's embedding key from JS
    raw = adata.uns.get(source)
    if raw is None:
        return {"type": "error", "message": f"No saved masks found in adata.uns['{source}']"}
    try:
        masks_store = _json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        return {"type": "error", "message": f"Failed to parse masks from '{source}': {e}"}

    _MAX_IDX = 2000  # cap per region sent to JS (keeps payload small)

    print(f"[Regions] Loading masks from adata.uns['{source}'] (embedding='{embedding}')")
    groups = masks_store.get("groups", {})

    if "selections" in masks_store:
        raw_items = masks_store.get("selections", {})
        full_indices = {n: s.get("indices", []) for n, s in raw_items.items()}
        regions = {
            n: {"indices": idx[:_MAX_IDX], "count": len(idx), "visible": True, "tool": "region"}
            for n, idx in full_indices.items()
        }
        polygons_for_canvas = []
        metadata = {"fill_opacity": 0.1}
    else:
        raw_regions = masks_store.get("regions", {})
        metadata = masks_store.get("metadata", {})
        full_indices = {n: r.get("indices", []) for n, r in raw_regions.items()}
        regions = {
            n: {"indices": full_indices[n][:_MAX_IDX], "count": len(full_indices[n]), "visible": r.get("visible", True), "tool": r.get("tool", "region")}
            for n, r in raw_regions.items()
        }
        polygons_for_canvas = [
            {"name": n, "polygons": r["polygons"],
             "centroid_x": r.get("centroid_x"), "centroid_y": r.get("centroid_y"),
             "indices": full_indices[n][:_MAX_IDX]}
            for n, r in raw_regions.items() if r.get("polygons")
        ]

    # Transform polygon vertices from stored embedding → current embedding if they differ
    stored_embedding = metadata.get("embedding") or _detect_stored_embedding(
        polygons_for_canvas, full_indices, adata
    )
    if embedding and stored_embedding and stored_embedding != embedding and polygons_for_canvas:
        print(f"[Regions] Transforming polygons: '{stored_embedding}' → '{embedding}'")
        polygons_for_canvas = _transform_polygons_to_embedding(
            polygons_for_canvas, full_indices, adata,
            source_embedding=stored_embedding,
            target_embedding=embedding,
        )

    # Cache full indices server-side for save-to-obs operations
    adata.uns["_sel_idx_cache_"] = _json.dumps(full_indices)

    total_regions = len(regions)
    total_groups = len(groups)
    print(f"[Regions] Loaded {total_regions} regions in {total_groups} groups")

    return {
        "type": "region_masks_loaded",
        "groups": groups,
        "regions": regions,
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

    # Load full indices from caches — JS may send capped (2000) indices, but full set is better
    import json as _json
    full_idx_cache = {}
    for src_key in ("_sel_idx_cache_", "_dbscan_tmp"):
        raw = adata.uns.get(src_key)
        if raw is not None:
            try:
                full_idx_cache.update(_json.loads(raw) if isinstance(raw, str) else raw)
            except Exception:
                pass

    results = []
    failed = []

    for region in regions_list:
        name = region["name"]
        # Prefer full cached indices over the (possibly capped) JS-sent ones
        indices = full_idx_cache.get(name) or region["indices"]

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


def recapture_region_cells(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Re-derive cell membership for loaded polygon masks using point-in-polygon testing.

    This is called after importing masks to find all cells whose coordinates fall
    within each stored polygon boundary in the current embedding coordinate space.
    Avoids recomputing alpha shapes from scratch.

    Expects data keys (JSON in data["payload"]):
        - polygons: list of { name, polygons: [[[x,y],...]] }
        - embedding: current embedding key (e.g., "X_TMA_grid", "spatial")
    """
    from matplotlib.path import Path as MplPath

    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    import json as _json
    payload_str = data.get("payload", "{}")
    try:
        payload = _json.loads(payload_str) if isinstance(payload_str, str) else payload_str
    except Exception as e:
        return {"type": "error", "message": f"Failed to parse recapture payload: {e}"}

    polygons_list = payload.get("polygons", [])
    embedding = payload.get("embedding", None)

    if not polygons_list:
        return {"type": "error", "message": "No polygons provided for recapture"}

    coords = _get_embedding_coords(adata, embedding)
    if coords is None:
        return {"type": "error", "message": "No valid coordinate embedding found"}

    all_coords = coords  # shape (n_cells, 2)
    n_cells = all_coords.shape[0]

    print(f"[Regions] Recapturing cells for {len(polygons_list)} masks using embedding='{embedding}' ({n_cells} total cells)")

    results = {}
    failed = []

    for entry in polygons_list:
        name = entry.get("name", "")
        poly_rings = entry.get("polygons", [])  # list of rings, each a list of [x, y]

        if not poly_rings:
            failed.append({"name": name, "reason": "No polygon vertices"})
            continue

        # Union of all rings: a cell is "inside" if it's inside ANY ring
        in_mask = np.zeros(n_cells, dtype=bool)
        for ring in poly_rings:
            if len(ring) < 3:
                continue
            verts = np.array(ring, dtype=float)
            try:
                path = MplPath(verts)
                in_ring = path.contains_points(all_coords)
                in_mask |= in_ring
            except Exception as e:
                print(f"[Regions] Recapture polygon error for '{name}': {e}")

        indices = list(np.where(in_mask)[0].astype(int))
        count = len(indices)

        if count == 0:
            failed.append({"name": name, "reason": "No cells found within polygon"})
            continue

        cx = float(all_coords[indices, 0].mean())
        cy = float(all_coords[indices, 1].mean())
        results[name] = {
            "indices": indices,
            "count": count,
            "centroid_x": cx,
            "centroid_y": cy,
        }

    # Update _sel_idx_cache_ with the newly recaptured full indices
    full_cache = {name: rd["indices"] for name, rd in results.items()}
    adata.uns["_sel_idx_cache_"] = _json.dumps(full_cache)

    print(f"[Regions] Recaptured: {len(results)} succeeded, {len(failed)} failed")

    return {
        "type": "region_cells_recaptured",
        "results": results,
        "failed": failed,
        "total_recaptured": len(results),
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
            "embedding": payload.get("embedding") or data.get("embedding"),  # coordinate space selections were made in
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
            "path": sdata.get("path", []),  # polygon path for layout tracking
        }
    
    import json as _json
    adata.uns["manual_masks"] = _json.dumps(masks_store)
    
    total_selections = len(masks_store["selections"])
    total_groups = len(masks_store["groups"])
    total_cells = sum(len(s.get("indices", [])) for s in masks_store["selections"].values())
    
    print(f"[Manual] Saved {total_selections} selections ({total_groups} groups, {total_cells} cells) to adata.uns['manual_masks']")
    
    return {
        "type": "manual_masks_saved",
        "uns_key": "manual_masks",
        "total_selections": total_selections,
        "total_groups": total_groups,
        "total_cells": total_cells,
    }


def load_manual_masks(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Load manual selection masks from adata.uns and return all at once.

    data["source"] picks the uns key; format is auto-detected by content.
    data["embedding"] is the current embedding key from JS — used to transform
    stored polygon paths to the current coordinate space.

    If the source is region_masks format (has "regions" key with polygon geometry),
    the stored polygons are translated source_embedding → current_embedding using
    per-region cell-index offsets. No convex hull computation is done.
    """
    import json as _json

    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    source = data.get("source", "manual_masks")
    current_embedding = data.get("embedding", None)
    raw = adata.uns.get(source)
    if raw is None:
        return {"type": "error", "message": f"No saved masks found in adata.uns['{source}']"}
    try:
        masks_store = _json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        return {"type": "error", "message": f"Failed to parse masks from '{source}': {e}"}

    _MAX_IDX = 2000  # cap per selection sent to JS (keeps payload small)

    stored_embedding = masks_store.get("metadata", {}).get("embedding")
    print(f"[Manual] Loading masks from adata.uns['{source}'] "
          f"(stored_embedding='{stored_embedding}', current='{current_embedding}')")
    groups = masks_store.get("groups", {})

    # ── Detect format ─────────────────────────────────────────────────────────
    if "regions" in masks_store:
        # region_masks format: has polygon geometry per region
        raw_regions = masks_store.get("regions", {})
        full_indices = {n: r.get("indices", []) for n, r in raw_regions.items()}
        tool_map = {n: "polygon" for n in raw_regions}
        # Build polygon entries in the same shape _transform_polygons_to_embedding expects
        polygons_for_transform = [
            {"name": n, "polygons": r["polygons"],
             "centroid_x": r.get("centroid_x"), "centroid_y": r.get("centroid_y"),
             "indices": full_indices[n][:_MAX_IDX]}
            for n, r in raw_regions.items() if r.get("polygons")
        ]
        has_polygon_data = True
    else:
        raw_selections = masks_store.get("selections", {})
        full_indices = {n: s.get("indices", []) for n, s in raw_selections.items()}
        tool_map = {n: s.get("tool", "manual") for n, s in raw_selections.items()}
        # Extract saved polygon paths (saved since v2 of manual_masks format)
        saved_paths = {n: s["path"] for n, s in raw_selections.items() if s.get("path")}
        if saved_paths:
            polygons_for_transform = [
                {"name": n, "polygons": [path], "centroid_x": None, "centroid_y": None}
                for n, path in saved_paths.items()
            ]
            has_polygon_data = True
        else:
            polygons_for_transform = []
            has_polygon_data = False

    # ── Transform stored polygons to current embedding ────────────────────────
    path_map: Dict[str, list] = {}
    if has_polygon_data and current_embedding and polygons_for_transform:
        if not stored_embedding:
            stored_embedding = _detect_stored_embedding(polygons_for_transform, full_indices, adata)
        if stored_embedding != current_embedding:
            print(f"[Manual] Transforming polygon paths: '{stored_embedding}' → '{current_embedding}'")
            polygons_for_transform = _transform_polygons_to_embedding(
                polygons_for_transform, full_indices, adata,
                source_embedding=stored_embedding,
                target_embedding=current_embedding,
            )
        # Build path_map: first ring of each polygon as the editable path
        for entry in polygons_for_transform:
            if entry.get("polygons"):
                path_map[entry["name"]] = [[float(v[0]), float(v[1])]
                                            for v in entry["polygons"][0]]

    # Cache full indices server-side
    adata.uns["_sel_idx_cache_"] = _json.dumps(full_indices)

    # Build selections for the JS payload.
    # If a polygon path is available, omit saved indices entirely — the canvas
    # will recompute cell membership via its own point-in-polygon logic when the
    # selection is activated (same as drawing a polygon manually).
    selections = {}
    for n, idx in full_indices.items():
        if n in path_map:
            # Path-only: canvas will find cells itself
            sel = {
                "indices": [],
                "count": 0,
                "tool": tool_map[n],
                "path": path_map[n],
            }
        else:
            # No polygon stored — fall back to saved indices
            sel = {
                "indices": idx[:_MAX_IDX],
                "count": len(idx),
                "tool": tool_map[n],
            }
        selections[n] = sel

    total_selections = len(selections)
    total_groups = len(groups)
    print(f"[Manual] Loaded {total_selections} selections in {total_groups} groups "
          f"({len(path_map)} with polygon paths)")

    return {
        "type": "manual_masks_loaded",
        "groups": groups,
        "selections": selections,
        "total_selections": total_selections,
        "total_groups": total_groups,
    }


def transform_manual_paths(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Transform manual selection polygon paths from one embedding to another.

    Used when the user switches to a different layout/embedding and existing
    manual selection paths need to be repositioned to match the new coordinate space.

    Expects data keys:
        - selections: list of {name, path: [[x,y],...], tool, indices: [int,...]}
        - source_embedding: embedding key the paths are currently in
        - target_embedding: embedding key to transform paths into

    Returns:
        {type: "manual_paths_transformed", selections: [{name, path, tool}]}
    """
    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    import json as _json

    raw = data.get("selections", [])
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception as e:
            return {"type": "error", "message": f"Failed to parse selections: {e}"}

    source_embedding = data.get("source_embedding")
    target_embedding = data.get("target_embedding")

    if not source_embedding or not target_embedding:
        return {"type": "error", "message": "source_embedding and target_embedding are required"}

    if source_embedding == target_embedding:
        return {
            "type": "manual_paths_transformed",
            "selections": [{"name": s["name"], "path": s["path"], "tool": s.get("tool", "polygon")} for s in raw],
        }

    # Wrap into the format _transform_polygons_to_embedding expects
    # polygons = list of {name, polygons: [[ring]], centroid_x, centroid_y}
    # full_indices = {name: [int, ...]}
    polygons_for_canvas = [
        {
            "name": s["name"],
            "polygons": [s["path"]],   # each path is a single ring
            "centroid_x": None,
            "centroid_y": None,
        }
        for s in raw
    ]
    full_indices = {s["name"]: s.get("indices", []) for s in raw}
    tool_map = {s["name"]: s.get("tool", "polygon") for s in raw}

    print(f"[Manual] Transforming {len(raw)} selection paths: '{source_embedding}' → '{target_embedding}'")
    transformed = _transform_polygons_to_embedding(
        polygons_for_canvas, full_indices, adata,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
    )

    result_selections = []
    for entry in transformed:
        name = entry["name"]
        new_path = [[float(v[0]), float(v[1])] for v in entry["polygons"][0]] if entry.get("polygons") else []
        result_selections.append({"name": name, "path": new_path, "tool": tool_map.get(name, "polygon")})

    return {
        "type": "manual_paths_transformed",
        "selections": result_selections,
    }


def rename_region_mask(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Rename a region mask in all server-side caches so that save-to-obs and
    save-to-uns keep working after a JS-side rename.

    Expects data keys:
        - old_name: current (old) region name
        - new_name: desired new region name
    """
    import json as _json

    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    old_name = data.get("old_name", "")
    new_name = data.get("new_name", "")

    if not old_name or not new_name or old_name == new_name:
        return {"type": "rename_region_mask_ok", "old_name": old_name, "new_name": new_name}

    def _rename_in_json_uns(key):
        raw = adata.uns.get(key)
        if raw is None:
            return
        try:
            cache = _json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return
        if old_name in cache:
            cache[new_name] = cache.pop(old_name)
            adata.uns[key] = _json.dumps(cache)

    # Patch flat caches: _dbscan_tmp and _sel_idx_cache_ are {name: [indices]}
    _rename_in_json_uns("_dbscan_tmp")
    _rename_in_json_uns("_sel_idx_cache_")

    # Patch region_masks uns store: regions dict is {name: {...}}
    raw_masks = adata.uns.get("region_masks")
    if raw_masks is not None:
        try:
            masks_store = _json.loads(raw_masks) if isinstance(raw_masks, str) else raw_masks
            regions = masks_store.get("regions", {})
            if old_name in regions:
                regions[new_name] = regions.pop(old_name)
                # Also update group membership
                for gdata in masks_store.get("groups", {}).values():
                    gdata["selections"] = [
                        new_name if s == old_name else s
                        for s in gdata.get("selections", [])
                    ]
                adata.uns["region_masks"] = _json.dumps(masks_store)
        except Exception:
            pass

    print(f"[RegionRename] '{old_name}' → '{new_name}' updated in server caches")
    return {"type": "rename_region_mask_ok", "old_name": old_name, "new_name": new_name}


def save_region_group_to_obs(data: Dict, adata=None, __sample_idx=None, __sample_id__=None, **kwargs) -> Dict:
    """
    Write a region group's cell assignments to adata.obs as a categorical column.

    Looks up full cell indices server-side (DBSCAN cache → region_masks fallback)
    so the JS-side 2000-cap on canvas_indices never truncates the result.

    Expects data keys:
        - group_name: name of the obs column to create
        - region_names: list of region names in the group
    """
    import json as _json

    if adata is None:
        return {"type": "error", "message": "No adata object available"}

    group_name = data.get("group_name", "")
    region_names = data.get("region_names", [])

    if not group_name:
        return {"type": "error", "message": "No group_name provided"}
    if not region_names:
        return {"type": "error", "message": "No region_names provided"}

    # Load full indices from DBSCAN cache
    raw_tmp = adata.uns.get("_dbscan_tmp", "{}")
    try:
        dbscan_cache = _json.loads(raw_tmp) if isinstance(raw_tmp, str) else raw_tmp
    except Exception:
        dbscan_cache = {}

    # Load full indices from import cache (populated by load_region_masks / load_manual_masks)
    sel_idx_cache = {}
    raw_sel = adata.uns.get("_sel_idx_cache_")
    if raw_sel is not None:
        try:
            sel_idx_cache = _json.loads(raw_sel) if isinstance(raw_sel, str) else raw_sel
        except Exception:
            pass

    # Load full indices from saved region_masks as final fallback
    region_masks_cache = {}
    raw_masks = adata.uns.get("region_masks")
    if raw_masks is not None:
        try:
            masks_store = _json.loads(raw_masks) if isinstance(raw_masks, str) else raw_masks
            for rname, rdata in masks_store.get("regions", {}).items():
                region_masks_cache[rname] = rdata.get("indices", [])
        except Exception:
            pass

    n_cells = adata.n_obs
    new_column = pd.Series([np.nan] * n_cells, index=adata.obs.index, dtype="object")
    total_labeled = 0

    for rname in region_names:
        # Prefer DBSCAN cache → import cache → saved region_masks
        indices = dbscan_cache.get(rname) or sel_idx_cache.get(rname) or region_masks_cache.get(rname, [])
        if not indices:
            print(f"[RegionToObs] Warning: no indices found for region '{rname}'")
            continue
        for idx in indices:
            if 0 <= idx < n_cells:
                new_column.iloc[idx] = rname
        total_labeled += len(indices)

    adata.obs[group_name] = pd.Categorical(new_column)

    labeled_count = int(new_column.notna().sum())
    print(f"[RegionToObs] Saved '{group_name}' to adata.obs: {labeled_count} cells labeled across {len(region_names)} regions")

    return {
        "type": "region_group_obs_saved",
        "group_name": group_name,
        "labeled_cells": labeled_count,
        "region_count": len(region_names),
    }
