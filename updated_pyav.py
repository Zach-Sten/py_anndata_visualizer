# PART 1: Main body:
# STREAMING VERSION - Load only visible cells (max 100K at a time)

import zlib

def _pack_coords_binary(coords_array, compress=False) -> str:
    """Pack Nx2 float32 coordinates as base64, optionally with zlib compression."""
    import base64
    import numpy as np
    raw_bytes = coords_array.astype(np.float32).tobytes()
    if compress:
        raw_bytes = zlib.compress(raw_bytes, level=6)  # level 6 is good balance of speed/size
    return base64.b64encode(raw_bytes).decode('ascii')

def _b64(s: str) -> str:
    """Base64 encode a string."""
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _serialize_result(result: Any) -> Dict:
    """Safely serialize callback results to JSON-compatible format."""
    if isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    elif isinstance(result, (list, tuple)):
        return [_serialize_result(item) for item in result]
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, (np.integer, np.floating)):
        return result.item()
    elif isinstance(result, pd.Series):
        return result.tolist()
    elif isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    elif hasattr(result, "toarray"):
        return result.toarray().tolist()
    elif isinstance(result, (str, int, float, bool, type(None))):
        return result
    else:
        return str(result)


def link_buttons_to_python(
    html_content: str,
    button_callbacks: Dict[str, Callable],
    callback_args: Optional[Dict[str, Any]] = None,
    height: int = 600,
    width: int = 900,
    debug: bool = False,
    max_result_size: int = 30_000_000,
    initial_data: Optional[Dict] = None,
    chunk_size: int = 500_000,
) -> str:
    """
    Link HTML buttons to Python callbacks using widget-based communication.
    Adds a 2-panel layout:
      - Left: iframe UI (buttons/inputs)
      - Right: plot canvas (transparent) that can:
          * render embedding points: spatial/umap/pca (loaded upfront)
          * color points by continuous (GEX/obs numeric) or categorical (obs)
          * clear overlay back to neutral grey
    """
    import uuid
    import os
    
    if callback_args is None:
        callback_args = {}
    if initial_data is None:
        initial_data = {}

    iframe_id = f"iframe_{uuid.uuid4().hex}"
    output = widgets.Output()

    button_ids = list(button_callbacks.keys())
    button_ids_js = json.dumps(button_ids)
    initial_data_js = json.dumps(initial_data)

    # ----------------------------
    # Create data bridge widgets (one per button)
    # ----------------------------
    data_bridges: Dict[str, widgets.Text] = {}
    for bid in button_ids:
        bridge_name = f"_data_{iframe_id}_{bid}"
        bridge = widgets.Text(
            value="",
            description=bridge_name,
            placeholder=f"Data bridge for {bid}",
            layout=widgets.Layout(width="0px", height="0px", visibility="hidden", display="none"),
        )
        bridge.add_class(f"data-bridge-{iframe_id}")
        bridge.add_class(f"data-bridge-{bid}")
        data_bridges[bid] = bridge
        display(bridge)

    # ----------------------------
    # JS injected inside iframe:
    # - captures button clicks + input fields
    # - posts to parent
    # - receives python responses and re-dispatches as CustomEvent('pythonResponse')
    # ----------------------------
    communication_script = f"""
<script>
(function() {{
  const iframeId = {json.dumps(iframe_id)};
  const buttonIds = {button_ids_js};
  const initialData = {initial_data_js};
  let pendingRequest = null;

  function log(...args) {{ {"console.log('[iframe]', ...args);" if debug else ""} }}

  window.INITIAL_DATA = initialData;
  window._iframeId = iframeId;
  log("Initial data loaded:", initialData);

  // Buttons that are triggered programmatically (not UI buttons in iframe)
  const skipBridgeButtons = new Set([
    "computeLayoutBtn", "deleteLayoutBtn", "loadLayoutBtn", "saveLayoutBtn",
    "viewportBtn", "chunkBtn", "loadEmbeddingBtn", "__save_obs_column__"
  ]);
  
  buttonIds.forEach((bid) => {{
    if (skipBridgeButtons.has(bid)) return;
    const button = document.getElementById(bid);
    if (!button) {{
      log("Button not found:", bid);
      return;
    }}

    button.addEventListener('click', () => {{
      if (pendingRequest === bid) {{
        log("Request already pending for", bid);
        return;
      }}

      pendingRequest = bid;
      log("Button clicked:", bid);
      button.disabled = true;

      const data = {{}};

      // data-* attributes on the button
      for (let attr of button.attributes) {{
        if (attr.name.startsWith('data-')) {{
          data[attr.name.substring(5)] = attr.value;
        }}
      }}

      // inputs associated to this button via data-for="buttonId"
      const inputs = document.querySelectorAll('[data-for="' + bid + '"]');
      inputs.forEach(input => {{
        const key = input.getAttribute('data-key') || input.id || input.name;
        if (key) {{
          data[key] = input.value;
        }}
      }});

      log("Sending data:", data);

      window.parent.postMessage({{
        type: 'button_click',
        iframeId: iframeId,
        buttonId: bid,
        data: data,
        timestamp: Date.now()
      }}, '*');
    }});
  }});

  // Track processed chunks to avoid duplicates
  const processedChunks = new Set();
  
  window.addEventListener('message', (event) => {{
    if (event.data && event.data.type === 'python_response') {{
      const data = event.data.data;
      
      // Log chunk_data but don't block duplicates - allow reload after reconnect
      if (data && data.type === 'chunk_data') {{
        log("Chunk data received:", data.chunk);
      }}
      
      log("Received Python response:", data);

      if (pendingRequest) {{
        const button = document.getElementById(pendingRequest);
        if (button) button.disabled = false;
        pendingRequest = null;
      }}

      // Forward sample_meta responses to parent for layout computation
      if (data && data.type === 'sample_meta') {{
        window.parent.postMessage(data, '*');
      }}

      const customEvent = new CustomEvent('pythonResponse', {{
        detail: data
      }});
      window.dispatchEvent(customEvent);
    }}
  }});
}})();
</script>
"""

    if "</body>" in html_content.lower():
        idx = html_content.lower().rfind("</body>")
        full_html = html_content[:idx] + communication_script + html_content[idx:]
    else:
        full_html = html_content + communication_script

    payload_b64 = _b64(full_html)

    # ----------------------------
    # CHUNKED LOADING: Stream chunks via websocket with struct-of-arrays on JS side
    # ----------------------------
    adata = callback_args.get("adata", None)
    USE_COMPRESSION = True
    
    spatial = None
    if adata is not None:
        if "spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["spatial"])
        elif "X_spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["X_spatial"])

    if spatial is None or spatial.ndim != 2 or spatial.shape[1] < 2:
        sample_idx = np.array([], dtype=int)
        embeds_js = json.dumps({"streaming": True, "chunked": True, "spatial": None, "umap": None, "pca": None})
        sample_meta_js = "null"
    else:
        n = spatial.shape[0]
        sample_idx = np.arange(n, dtype=int)
        
        # ----------------------------
        # DYNAMIC CHUNK SIZING: configurable cells per chunk (default 500K)
        # ----------------------------
        CELLS_PER_CHUNK = chunk_size
        NUM_CHUNKS = max(1, (n + CELLS_PER_CHUNK - 1) // CELLS_PER_CHUNK)  # Ceiling division
        NUM_CHUNKS = min(NUM_CHUNKS, 50)  # Cap at 50 chunks max
        print(f"[Chunked Loading] {n:,} cells → {NUM_CHUNKS} chunks (~{n // NUM_CHUNKS:,} cells each)")
        
        # ----------------------------
        # CHUNK ASSIGNMENT
        # ----------------------------
        need_reassign = "__chunk__" not in adata.obs.columns
        if not need_reassign:
            max_chunk = adata.obs["__chunk__"].max()
            if max_chunk != NUM_CHUNKS - 1:
                need_reassign = True
        
        if need_reassign:
            np.random.seed(42)
            chunk_assignments = np.random.randint(0, NUM_CHUNKS, size=n)
            adata.obs["__chunk__"] = chunk_assignments
        else:
            chunk_assignments = adata.obs["__chunk__"].values
        
        # Calculate bounds
        spatial_coords = spatial[:, :2]
        bounds = {
            "minX": float(spatial_coords[:, 0].min()),
            "maxX": float(spatial_coords[:, 0].max()),
            "minY": float(spatial_coords[:, 1].min()),
            "maxY": float(spatial_coords[:, 1].max()),
            "count": n
        }
        
        # Get UMAP/PCA if available
        umap_coords = None
        pca_coords = None
        if "X_umap" in adata.obsm:
            umap_coords = np.asarray(adata.obsm["X_umap"])[:, :2]
        if "X_pca" in adata.obsm:
            pca_coords = np.asarray(adata.obsm["X_pca"])[:, :2]
        
        # ----------------------------
        # CHUNK 0: Embed in HTML for instant display
        # ----------------------------
        chunk0_mask = (chunk_assignments == 0)
        chunk0_indices = np.where(chunk0_mask)[0]
        chunk0_binary = _pack_coords_binary(spatial_coords[chunk0_indices], compress=USE_COMPRESSION)
        
        chunk0_umap_binary = None
        chunk0_pca_binary = None
        if umap_coords is not None:
            chunk0_umap_binary = _pack_coords_binary(umap_coords[chunk0_indices], compress=USE_COMPRESSION)
        if pca_coords is not None:
            chunk0_pca_binary = _pack_coords_binary(pca_coords[chunk0_indices], compress=USE_COMPRESSION)
        
        # Sample for minimap (use chunk 0 - it's already a representative sample!)
        minimap_sample_size = min(50_000, len(chunk0_indices))
        chunk0_spatial = spatial_coords[chunk0_indices]
        if len(chunk0_indices) > minimap_sample_size:
            minimap_subset = np.random.choice(len(chunk0_indices), size=minimap_sample_size, replace=False)
            minimap_coords = chunk0_spatial[minimap_subset]
        else:
            minimap_coords = chunk0_spatial
        minimap_binary = base64.b64encode(minimap_coords.astype(np.float32).tobytes()).decode('ascii')
        
        # Embedding bounds
        umap_bounds = None
        pca_bounds = None
        if umap_coords is not None:
            umap_bounds = {
                "minX": float(umap_coords[:, 0].min()),
                "maxX": float(umap_coords[:, 0].max()),
                "minY": float(umap_coords[:, 1].min()),
                "maxY": float(umap_coords[:, 1].max()),
                "count": n
            }
        if pca_coords is not None:
            pca_bounds = {
                "minX": float(pca_coords[:, 0].min()),
                "maxX": float(pca_coords[:, 0].max()),
                "minY": float(pca_coords[:, 1].min()),
                "maxY": float(pca_coords[:, 1].max()),
                "count": n
            }
        
        embeds_js = json.dumps({
            "streaming": True,
            "chunked": True,
            "compressed": USE_COMPRESSION,
            "numChunks": NUM_CHUNKS,
            "hasUmap": umap_coords is not None,
            "hasPca": pca_coords is not None,
            "spatial": bounds,
            "umap": umap_bounds,
            "pca": pca_bounds,
            "chunk0": {
                "indices": chunk0_indices.tolist(),
                "spatial_binary": chunk0_binary,
                "umap_binary": chunk0_umap_binary,
                "pca_binary": chunk0_pca_binary,
                "count": len(chunk0_indices)
            },
            "minimap": {
                "coords_binary": minimap_binary,
                "count": len(minimap_coords)
            }
        })
        
        # ----------------------------
        # SAMPLE METADATA for JS-side layout computation
        # Lightweight: just names + spatial centroids + bboxes
        # Group/sort metadata fetched on-demand via bridge
        # ----------------------------
        sample_id_col = initial_data.get("sample_id") if initial_data else None
        sample_meta_js = "null"
        if sample_id_col and sample_id_col in adata.obs.columns:
            samp_arr = adata.obs[sample_id_col].astype(str).values
            unique_samps = list(pd.Series(samp_arr).unique())
            n_samps = len(unique_samps)
            samp_to_idx = {s: i for i, s in enumerate(unique_samps)}
            
            # Per-cell sample index (uint16)
            cell_sample_ids = np.array([samp_to_idx[s] for s in samp_arr], dtype=np.uint16)
            
            # Per-sample spatial metadata (compact arrays)
            s_cx = np.zeros(n_samps, dtype=np.float32)
            s_cy = np.zeros(n_samps, dtype=np.float32)
            s_w = np.zeros(n_samps, dtype=np.float32)
            s_h = np.zeros(n_samps, dtype=np.float32)
            for si, s in enumerate(unique_samps):
                mask = (samp_arr == s)
                sc = spatial_coords[mask]
                s_cx[si] = sc[:, 0].mean()
                s_cy[si] = sc[:, 1].mean()
                s_w[si] = np.ptp(sc[:, 0])
                s_h[si] = np.ptp(sc[:, 1])
            
            # Chunk0 sample IDs (compressed)
            chunk0_sids_b64 = base64.b64encode(
                zlib.compress(cell_sample_ids[chunk0_indices].tobytes(), level=6)
            ).decode("ascii")
            
            sample_meta_js = json.dumps({
                "names": [s for s in unique_samps],
                "cx": s_cx.tolist(), "cy": s_cy.tolist(),
                "w": s_w.tolist(), "h": s_h.tolist(),
                "chunk0_sids_b64": chunk0_sids_b64,
            })
            
            adata.uns["__cell_sample_ids__"] = cell_sample_ids
            adata.uns["__sample_names__"] = unique_samps
            print(f"[Layout] {n_samps} samples")

    callback_args = dict(callback_args)
    callback_args["__sample_idx"] = sample_idx.tolist()

    # ----------------------------
    # Parent container: 2 panels + plot canvas (transparent background)
    # ----------------------------
    container_html = f"""
<div style="display:flex; gap:10px; width:100%; height:{int(height)}px; margin:0; padding:0; box-sizing:border-box;">
  <!-- Left panel: controls iframe -->
  <div id="left_{iframe_id}" style="width: 420px; height:100%; flex: 0 0 420px; box-sizing:border-box;">
    <iframe
      id="{iframe_id}"
      style="width:100%; height:100%; border:1px solid #ccc; border-radius:6px; background:white; box-sizing:border-box;"
    ></iframe>
  </div>

  <!-- Right panel: plot area - flex grow to fill remaining space -->
  <div id="plot_panel_{iframe_id}"
       tabindex="0"
       style="flex: 1 1 auto; min-width:400px; height:100%; border:1px solid rgba(128,128,128,0.2); border-radius:6px;
              background: inherit; position: relative; overflow:hidden; outline:none; box-sizing:border-box;">
    <canvas id="plot_canvas_{iframe_id}"
            style="width:100%; height:100%; display:block; background: inherit;"></canvas>

    <!-- Loading overlay - shown until all chunks loaded -->
    <div id="loading_overlay_{iframe_id}"
         style="position:absolute; top:0; left:0; right:0; bottom:0;
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                background: rgba(255,255,255,0.85); z-index:100;
                pointer-events: all;">
      <div style="text-align:center;">
        <div style="font-size:14px; color:#333; margin-bottom:12px; font-family: ui-monospace, monospace;">
          Loading cells...
        </div>
        <div style="width:200px; height:8px; background:#e0e0e0; border-radius:4px; overflow:hidden;">
          <div id="loading_bar_{iframe_id}" 
               style="width:3%; height:100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); 
                      border-radius:4px; transition: width 0.2s ease;"></div>
        </div>
        <div id="loading_text_{iframe_id}" 
             style="font-size:11px; color:#666; margin-top:8px; font-family: ui-monospace, monospace;">
          0 / 3,432,339 cells
        </div>
      </div>
    </div>
    
    <!-- Layout loading overlay - shown during layout computation -->
    <div id="layout_loading_{iframe_id}"
         style="position:absolute; top:0; left:0; right:0; bottom:0;
                display:none; flex-direction:column; align-items:center; justify-content:center;
                background: rgba(255,255,255,0.75); z-index:150;
                pointer-events: all;">
      <div style="text-align:center;">
        <div style="font-size:14px; color:#333; font-family: ui-monospace, monospace;">
          Configuring layout...
        </div>
      </div>
    </div>
    
    <!-- GEX loading overlay - shown during gene expression loading -->
    <div id="gex_loading_{iframe_id}"
         style="position:absolute; top:0; left:0; right:0; bottom:0;
                display:none; flex-direction:column; align-items:center; justify-content:center;
                background: rgba(255,255,255,0.75); z-index:150;
                pointer-events: all;">
      <div style="text-align:center;">
        <div style="font-size:14px; color:#333; font-family: ui-monospace, monospace;">
          Loading gene expression...
        </div>
      </div>
    </div>

    <!-- Hidden label (needed for state tracking but not displayed) -->
    <div id="plot_label_{iframe_id}" style="display:none;">Embedding: spatial</div>

    <!-- Camera button in top left -->
    <button id="camera_btn_{iframe_id}"
            style="position:absolute; left:10px; top:10px;
                   width:36px; height:36px;
                   border-radius:8px;
                   border:1px solid rgba(0,0,0,0.12);
                   background: rgba(255,255,255,0.9);
                   cursor:pointer;
                   display:flex;
                   align-items:center;
                   justify-content:center;
                   transition: all 0.15s ease;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
            title="Download plot as PNG">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
        <circle cx="12" cy="13" r="4"></circle>
      </svg>
    </button>
    
    <!-- Minimap in bottom left -->
    <canvas id="minimap_{iframe_id}"
            style="position:absolute; bottom:10px; left:10px;
                   width:120px; height:120px;
                   background:#000;
                   border:2px solid rgba(255,255,255,0.3);
                   border-radius:8px;
                   display:block;"></canvas>
    
    <!-- 2D overlay canvas for sample labels (on top of WebGL) -->
    <canvas id="label_overlay_{iframe_id}"
            style="position:absolute; top:0; left:0; width:100%; height:100%;
                   pointer-events:none; z-index:10;"></canvas>
  </div>
</div>

<style>
  /* Constrain Jupyter output cell padding */
  .jp-OutputArea-output:has(#plot_panel_{iframe_id}) {{
    padding: 0 !important;
    margin: 0 !important;
  }}
  .output_subarea:has(#plot_panel_{iframe_id}) {{
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
  }}
  #camera_btn_{iframe_id}:hover {{
    background: rgba(141,236,245,0.2);
    border-color: rgba(141,236,245,0.6);
    box-shadow: 0 4px 8px rgba(0,0,0,0.12);
  }}
</style>

<script>
(function() {{
  const iframeId = {json.dumps(iframe_id)};

  const iframe = document.getElementById({json.dumps(iframe_id)});
  const b64 = {json.dumps(payload_b64)};
  const html = decodeURIComponent(escape(atob(b64)));

  // ---- write iframe content ----
  const doc = iframe.contentWindow.document;
  doc.open();
  doc.write(html);
  doc.close();

  // ---- request queue for button bridge ----
  window["_requests_" + iframeId] = [];

  function log(...args) {{ {"console.log('[parent]', ...args);" if debug else ""} }}

  // Guard against duplicate listener on reruns for this iframe_id
  const MSG_GUARD = "_msg_listener_installed_" + iframeId;
  if (!window[MSG_GUARD]) {{
    window[MSG_GUARD] = true;
    window.addEventListener('message', (event) => {{
      if (!event.data) return;
      
      // Handle GEX loading overlay messages
      if (event.data.type === "show_gex_loading") {{
        showGexLoading();
        return;
      }}
      if (event.data.type === "hide_gex_loading") {{
        hideGexLoading();
        return;
      }}
      
      if (event.data.iframeId === iframeId && event.data.type === "button_click") {{
        log("Queued button click:", event.data.buttonId, "with data:", event.data.data);
        window["_requests_" + iframeId].push(event.data);
      }}
      // Gene switch from iframe chip — use cache
      if (event.data.type === "switch_gene" && event.data.gene) {{
        const gene = event.data.gene;
        if (activateCachedGene(gene)) {{
          console.log(`[GEX] Cache hit: ${{gene}}`);
          const iframeEl = document.getElementById(iframeId);
          if (iframeEl && iframeEl.contentWindow) {{
            iframeEl.contentWindow.postMessage({{ type: "gex_loaded", gene: gene, vmax: currentGexVmax }}, "*");
          }}
          setLabel("GEX: " + gene); markGPUDirty(); draw();
        }} else {{
          // Show loading overlay for cache miss
          showGexLoading();
          const iframeEl = document.getElementById(iframeId);
          if (iframeEl && iframeEl.contentWindow) {{
            iframeEl.contentWindow.postMessage({{ type: "gex_cache_miss", gene: gene }}, "*");
          }}
        }}
      }}
      // Gene group expression request — send to Python
      if (event.data.type === "gene_group_expression") {{
        showGexLoading();  // Show loading for gene group too
        console.log("[GEX] Gene group requested:", event.data.groupName, event.data.genes, event.data.method);
        window["_requests_" + iframeId].push({{
          type: "button_click",
          iframeId: iframeId,
          buttonId: "geneGroupBtn",
          data: {{
            groupName: event.data.groupName,
            genes: event.data.genes,
            method: event.data.method
          }},
          timestamp: Date.now()
        }});
      }}
      // Listen to ui_state updates from iframe (for legend toggles, opacity, etc)
      if (event.data.type === "ui_state") {{
        const s = window["_plotState_" + iframeId];
        let colorsDirty = false;
        
        // Sync obs toggles/colors/opacity
        if (s && event.data.obs) {{
          if (event.data.obs.enabled) {{
            const prev = JSON.stringify(s.obs.enabled);
            s.obs.enabled = event.data.obs.enabled;
            if (JSON.stringify(s.obs.enabled) !== prev) colorsDirty = true;
          }}
          const newOp = event.data.obs.opacity != null ? event.data.obs.opacity : 1.0;
          if (s.obs.opacity !== newOp) {{ s.obs.opacity = newOp; colorsDirty = true; }}
          if (event.data.obs.colors) {{ s.obs.colors = event.data.obs.colors; currentPalette = event.data.obs.colors; colorsDirty = true; }}
        }}
        
        // Sync gex colormap/opacity
        if (s && event.data.gex) {{
          const newOp = event.data.gex.opacity != null ? event.data.gex.opacity : 1.0;
          if (s.gex.opacity !== newOp) {{ s.gex.opacity = newOp; colorsDirty = true; }}
          if (event.data.gex.colormap && s.gex.colormap !== event.data.gex.colormap) {{
            s.gex.colormap = event.data.gex.colormap; currentColormap = event.data.gex.colormap; colorsDirty = true;
          }}
          // Gene chip toggled OFF
          if (!event.data.gex.active && currentGexGene) {{
            gexValues.fill(0); currentGexGene = null; currentGexVmax = 0; colorsDirty = true;
          }}
          if (event.data.gex.active && event.data.gex.active !== currentGexGene) {{
            if (activateCachedGene(event.data.gex.active)) {{
              const iframeEl = document.getElementById(iframeId);
              if (iframeEl && iframeEl.contentWindow) {{
                iframeEl.contentWindow.postMessage({{ type: "gex_loaded", gene: event.data.gex.active, vmax: currentGexVmax }}, "*");
              }}
              colorsDirty = true;
            }}
          }}
        }}
        
        if (s && event.data.selection) {{
          const oldSel = s.selectionIndices;
          const newSel = event.data.selection.indices || null;
          s.selectionIndices = newSel;
          s.selectionPath = event.data.selection.path || null;  // Stored in DATA coords
          s.selectionTool = event.data.selection.tool || null;
          
          // Restore path to canvas for editing with transform handles
          // Path is stored in DATA coordinates, convert to CANVAS coords
          if (s.selectionPath && s.selectionPath.length > 0) {{
            window["_selectionPathData_" + iframeId] = s.selectionPath.map(p => [...p]);  // Store data coords
            window["_selectionPath_" + iframeId] = pathDataToCanvas(s.selectionPath);  // Convert to canvas coords
            window["_selectionTool_" + iframeId] = s.selectionTool;
            window["_isDrawing_" + iframeId] = false;  // Not drawing, just editing
            drawSelectionOutline();
          }} else {{
            // Clear canvas selection if no active selection
            window["_selectionPath_" + iframeId] = [];
            window["_selectionPathData_" + iframeId] = null;
            window["_selectionBounds_" + iframeId] = null;
            window["_selectionHandles_" + iframeId] = null;
          }}
          
          // Mark GPU dirty if selection changed
          if ((oldSel && !newSel) || (!oldSel && newSel) || 
              (oldSel && newSel && oldSel.length !== newSel.length)) {{
            colorsDirty = true;
          }}
        }}
        if (s && typeof event.data.pointSize === 'number') s.pointSize = event.data.pointSize;
        
        // Label
        if (s) {{
          const ho = currentObsColumn != null, hg = currentGexGene != null;
          s.label = hg && ho ? "obs: "+currentObsColumn+" + GEX: "+currentGexGene : hg ? "GEX: "+currentGexGene : ho ? "obs: "+currentObsColumn : "Embedding: "+currentEmbedding;
          setLabel(s.label);
        }}
        
        if (colorsDirty) markGPUDirty();
        draw();
      }}
      
      // Listen for selection_tool messages from iframe
      if (event.data.type === "selection_tool") {{
        const tool = event.data.tool;
        window["_selectionTool_" + iframeId] = tool;
        window["_selectionPath_" + iframeId] = [];
        window["_isDrawing_" + iframeId] = false;
        log("Selection tool changed to:", tool);
        draw(); // Redraw to clear any previous selection outline
      }}
      
      // Listen for save_obs_column message from iframe
      if (event.data.type === "save_obs_column" && event.data.iframeId === iframeId) {{
        log("Saving obs column:", event.data.columnName);
        window["_requests_" + iframeId].push({{
          buttonId: "__save_obs_column__",
          data: {{
            columnName: event.data.columnName,
            columnData: event.data.columnData
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Layout compute request from iframe — route directly to updatePlot (no Python)
      if (event.data.type === "compute_layout") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
      
      // Toggle sample labels
      if (event.data.type === "toggle_sample_labels") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
      
      // Layout save request, switch, delete, obsm, and ADJUST (live gap/transpose)
      if (event.data.type === "save_layout_request" || 
          event.data.type === "switch_to_saved_layout" ||
          event.data.type === "delete_saved_layout" ||
          event.data.type === "save_to_obsm" ||
          event.data.type === "adjust_layout") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
      
      // Forward sample_meta from iframe to updatePlot for layout metadata handling
      if (event.data.type === "sample_meta") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
    }});
  }}

  window["sendToIframe_" + iframeId] = function(data) {{
    const iframe2 = document.getElementById(iframeId);
    if (iframe2 && iframe2.contentWindow) {{
      iframe2.contentWindow.postMessage({{
        type: "python_response",
        data: data
      }}, "*");
    }}
  }};

  // ============================================================
  // STRUCT-OF-ARRAYS ARCHITECTURE: No per-cell objects, flat typed arrays
  // This avoids GC hell and enables O(1) embedding switches
  // ============================================================
  const METADATA = {embeds_js};
  const SAMPLE_META = {sample_meta_js};
  window["_sampleMeta_" + iframeId] = SAMPLE_META;
  let currentEmbedding = "spatial";
  
  // Total cells we'll load
  const TOTAL_CELLS = METADATA.spatial ? METADATA.spatial.count : 0;
  
  // STRUCT-OF-ARRAYS: Flat typed arrays instead of Map of objects
  const cellIds = new Int32Array(TOTAL_CELLS);
  const posSpatial = new Float32Array(TOTAL_CELLS * 2);
  const posUmap = new Float32Array(TOTAL_CELLS * 2);
  const posPca = new Float32Array(TOTAL_CELLS * 2);
  const posLayout = new Float32Array(TOTAL_CELLS * 2);
  const posLayoutSnapshot = new Float32Array(TOTAL_CELLS * 2);  // For layout-to-layout transitions
  const cellSampleId = new Uint16Array(TOTAL_CELLS);  // Per-cell sample index
  let layoutHasData = false;
  let layoutSampleLabels = [];
  let layoutLabelPositions = [];
  let showSampleLabels = true;  // Toggle for label visibility
  let layoutParams = null;  // Current layout parameters
  
  // Group and axis annotations for layout
  let layoutGroupLabels = [];      // Array of {{name, x, y}} objects - group name labels
  let layoutAxisInfo = null;       // {{columns: "patient_id", rows: "sample_order"}} - axis descriptors
  let layoutColLabels = [];        // Array of {{name, x, y}} - column header labels
  let layoutRowLabels = [];        // Array of {{name, x, y}} - row header labels
  
  // In-memory saved layouts (name -> snapshot of posLayout + metadata)
  const savedLayouts = {{}};
  let activeLayoutName = null;
  const obsValues = new Float32Array(TOTAL_CELLS);       // For coloring
  const gexValues = new Float32Array(TOTAL_CELLS);       // For gene expression
  let loadedCount = 0;  // Write head - how many cells loaded so far
  
  // Index lookup: cellId -> position in arrays (for color updates)
  const cellIdToIndex = new Map();
  
  // Color state
  let currentPalette = null;
  let currentCategories = null;
  let currentObsColumn = null;
  let currentGexGene = null;
  
  // Chunk tracking - SEQUENTIAL loading (more stable)
  const CHUNKS_LOADED = new Set();
  const NUM_CHUNKS = METADATA.numChunks || 10;
  let isLoadingChunk = false;  // One at a time
  let isFullyLoaded = false;
  
  // Loading overlay elements
  const loadingOverlay = document.getElementById("loading_overlay_" + iframeId);
  const loadingBar = document.getElementById("loading_bar_" + iframeId);
  const loadingText = document.getElementById("loading_text_" + iframeId);
  const layoutLoadingOverlay = document.getElementById("layout_loading_" + iframeId);
  const gexLoadingOverlay = document.getElementById("gex_loading_" + iframeId);
  
  // Show/hide layout loading overlay
  function showLayoutLoading() {{
    if (layoutLoadingOverlay) {{
      layoutLoadingOverlay.style.display = "flex";
    }}
  }}
  function hideLayoutLoading() {{
    if (layoutLoadingOverlay) {{
      layoutLoadingOverlay.style.display = "none";
    }}
  }}
  
  // Show/hide GEX loading overlay
  function showGexLoading() {{
    if (gexLoadingOverlay) {{
      gexLoadingOverlay.style.display = "flex";
    }}
  }}
  function hideGexLoading() {{
    if (gexLoadingOverlay) {{
      gexLoadingOverlay.style.display = "none";
    }}
  }}
  
  // Update loading progress
  function updateLoadingProgress() {{
    const progress = Math.round(CHUNKS_LOADED.size / NUM_CHUNKS * 100);
    
    if (loadingBar) {{
      loadingBar.style.width = progress + "%";
    }}
    if (loadingText) {{
      loadingText.textContent = `${{loadedCount.toLocaleString()}} / ${{TOTAL_CELLS.toLocaleString()}} cells`;
    }}
  }}
  
  // Hide loading overlay when complete
  function hideLoadingOverlay() {{
    if (loadingOverlay) {{
      loadingOverlay.style.opacity = "0";
      loadingOverlay.style.transition = "opacity 0.3s ease";
      setTimeout(() => {{
        loadingOverlay.style.display = "none";
      }}, 300);
    }}
    isFullyLoaded = true;
    console.log(`[Loading] Complete! ${{loadedCount.toLocaleString()}} cells loaded`);
  }}
  
  // ----------------------------
  // INSTANT LOAD: Decode Chunk 0 with ALL embeddings
  // Appends directly to typed arrays (no objects!)
  // ----------------------------
  function loadChunk0() {{
    if (!METADATA.chunk0) return;
    
    const chunk0 = METADATA.chunk0;
    const indices = chunk0.indices;
    const count = chunk0.count;
    
    // Decode ALL embedding coordinates
    const spatialCoords = chunk0.spatial_binary ? decodeBinaryCoords(chunk0.spatial_binary, count) : null;
    const umapCoords = chunk0.umap_binary ? decodeBinaryCoords(chunk0.umap_binary, count) : null;
    const pcaCoords = chunk0.pca_binary ? decodeBinaryCoords(chunk0.pca_binary, count) : null;
    
    // Decode sample IDs for chunk0
    let chunk0Sids = null;
    if (SAMPLE_META && SAMPLE_META.chunk0_sids_b64) {{
      try {{
        const raw = atob(SAMPLE_META.chunk0_sids_b64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        const inflated = (typeof pako !== 'undefined') ? pako.inflate(bytes) : bytes;
        // Copy to aligned buffer for Uint16Array
        const aligned = new ArrayBuffer(count * 2);
        new Uint8Array(aligned).set(inflated.subarray(0, count * 2));
        chunk0Sids = new Uint16Array(aligned);
      }} catch(e) {{ console.warn("[Chunk0] Failed to decode sample IDs:", e); }}
    }}
    
    if (!spatialCoords && !umapCoords && !pcaCoords) {{
      console.warn("[Chunk0] No coordinate data available");
      return;
    }}
    
    // Reset state
    loadedCount = 0;
    cellIdToIndex.clear();
    CHUNKS_LOADED.clear();
    
    // STRUCT-OF-ARRAYS: Append directly to flat typed arrays (no objects!)
    for (let i = 0; i < count; i++) {{
      const idx = indices[i];
      const writePos = loadedCount;
      
      // Store cell ID and its position in our arrays
      cellIds[writePos] = idx;
      cellIdToIndex.set(idx, writePos);
      
      // Store sample assignment
      if (chunk0Sids) cellSampleId[writePos] = chunk0Sids[i];
      
      // Copy coordinates directly (no object allocation!)
      if (spatialCoords) {{
        posSpatial[writePos * 2] = spatialCoords[i * 2];
        posSpatial[writePos * 2 + 1] = spatialCoords[i * 2 + 1];
      }}
      if (umapCoords) {{
        posUmap[writePos * 2] = umapCoords[i * 2];
        posUmap[writePos * 2 + 1] = umapCoords[i * 2 + 1];
      }}
      if (pcaCoords) {{
        posPca[writePos * 2] = pcaCoords[i * 2];
        posPca[writePos * 2 + 1] = pcaCoords[i * 2 + 1];
      }}
      
      loadedCount++;
    }}
    
    CHUNKS_LOADED.add(0);
    
    // Update loading progress bar
    updateLoadingProgress();
    
    // Start SEQUENTIAL chunk loading (one at a time for stability)
    setTimeout(() => requestNextChunk(), 100);
  }}
  
  // ----------------------------
  // SEQUENTIAL CHUNK LOADING: One chunk at a time for stability
  // ----------------------------
  
  // Process chunk data - STRUCT-OF-ARRAYS version (no objects!)
  let lastChunkTime = 0;
  let chunkProcessingLock = false;
  
  function processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, count, chunkSids, responseRequestId) {{
    const now = Date.now();
    
    // Check if this is a stale response (from an old request)
    if (responseRequestId !== undefined && responseRequestId !== null && 
        lastChunkRequestId !== null && responseRequestId !== lastChunkRequestId) {{
      console.log("[Chunk] Stale response - got reqId", responseRequestId, "but expected", lastChunkRequestId, "- ignoring and re-requesting");
      isLoadingChunk = false;
      setTimeout(() => requestNextChunk(), 200);  // Re-request the chunk we actually need
      return;
    }}
    
    // Also handle case where response has no requestId (old cached response)
    if ((responseRequestId === undefined || responseRequestId === null) && CHUNKS_LOADED.has(chunkId)) {{
      console.log("[Chunk] Old cached response for chunk", chunkId, "- ignoring and re-requesting");
      isLoadingChunk = false;
      setTimeout(() => requestNextChunk(), 200);
      return;
    }}
    
    // If we already have this chunk, skip it and request next
    if (CHUNKS_LOADED.has(chunkId)) {{
      console.log("[Chunk] Already have chunk", chunkId, "- requesting next");
      isLoadingChunk = false;
      setTimeout(() => requestNextChunk(), 50);
      return;
    }}
    
    lastChunkTime = now;
    
    // Prevent concurrent processing
    if (chunkProcessingLock) {{
      console.log("[Chunk] Processing locked, queuing:", chunkId);
      setTimeout(() => processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, count, chunkSids), 50);
      return;
    }}
    
    chunkProcessingLock = true;
    CHUNKS_LOADED.add(chunkId);
    isLoadingChunk = false;
    
    console.log("[Chunk] Processing chunk", chunkId, "with", count, "cells");
    
    for (let i = 0; i < count; i++) {{
      const idx = indices[i];
      if (cellIdToIndex.has(idx)) continue;
      
      const writePos = loadedCount;
      cellIds[writePos] = idx;
      cellIdToIndex.set(idx, writePos);
      
      // Store sample assignment
      if (chunkSids) cellSampleId[writePos] = chunkSids[i];
      
      if (spatialCoords) {{
        posSpatial[writePos * 2] = spatialCoords[i * 2];
        posSpatial[writePos * 2 + 1] = spatialCoords[i * 2 + 1];
      }}
      if (umapCoords) {{
        posUmap[writePos * 2] = umapCoords[i * 2];
        posUmap[writePos * 2 + 1] = umapCoords[i * 2 + 1];
      }}
      if (pcaCoords) {{
        posPca[writePos * 2] = pcaCoords[i * 2];
        posPca[writePos * 2 + 1] = pcaCoords[i * 2 + 1];
      }}
      
      loadedCount++;
    }}
    
    // If layout is active, recompute with new cells
    if (layoutHasData && layoutParams && currentEmbedding === "layout") {{
      computeLayoutJS(layoutParams);
    }}
    
    updateLoadingProgress();
    markGPUDirty();
    draw();
    
    // Unlock processing
    chunkProcessingLock = false;
    
    // Check if all chunks loaded
    if (CHUNKS_LOADED.size >= NUM_CHUNKS) {{
      console.log("[Chunk] All", NUM_CHUNKS, "chunks loaded successfully");
      hideLoadingOverlay();
      isFullyLoaded = true;
    }} else {{
      // Request next missing chunk
      setTimeout(() => requestNextChunk(), 50);
    }}
  }}
  
  // Request next MISSING chunk (not just sequential)
  let lastRequestedChunk = null;  // Track which chunk we actually requested
  let chunkRequestId = 0;  // Unique ID for each request
  let lastChunkRequestId = null;  // The ID of our current pending request
  
  function requestNextChunk() {{
    if (isLoadingChunk) return;
    if (CHUNKS_LOADED.size >= NUM_CHUNKS) {{
      hideLoadingOverlay();
      isFullyLoaded = true;
      return;
    }}
    
    // Find first missing chunk
    let nextChunk = null;
    for (let c = 1; c < NUM_CHUNKS; c++) {{
      if (!CHUNKS_LOADED.has(c)) {{
        nextChunk = c;
        break;
      }}
    }}
    
    if (nextChunk === null) {{
      hideLoadingOverlay();
      isFullyLoaded = true;
      return;
    }}
    
    isLoadingChunk = true;
    lastRequestedChunk = nextChunk;
    chunkRequestId++;
    lastChunkRequestId = chunkRequestId;
    console.log("[Chunk] Requesting chunk", nextChunk, "reqId:", chunkRequestId);
    
    const requestData = {{
      chunk: nextChunk,
      requestId: chunkRequestId,  // Echo this back
      activeColumn: currentObsColumn,
      activeGene: currentGexGene
    }};
    
    window["_requests_" + iframeId].push({{
      type: 'button_click',
      iframeId: iframeId,
      buttonId: 'chunkBtn',
      data: requestData,
      timestamp: Date.now()
    }});
  }}
  
  // NOTE: loadChunk0() is called AFTER WebGL initialization below
  
  // Decode minimap sample for global positioning
  let MINIMAP_POINTS = [];
  if (METADATA.minimap && METADATA.minimap.coords_binary) {{
    const binary = atob(METADATA.minimap.coords_binary);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const floats = new Float32Array(bytes.buffer);
    
    MINIMAP_POINTS = [];
    for (let i = 0; i < floats.length; i += 2) {{
      MINIMAP_POINTS.push([floats[i], floats[i + 1]]);
    }}
  }}
  
  console.log("[Chunked Streaming] Metadata loaded:", METADATA);
  
  // Load pako for zlib decompression if needed
  const USE_COMPRESSION = METADATA.compressed || false;
  let pakoLoaded = false;
  
  if (USE_COMPRESSION && typeof pako === 'undefined') {{
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js';
    script.onload = () => {{ pakoLoaded = true; console.log("[Compression] pako loaded"); }};
    document.head.appendChild(script);
  }} else if (typeof pako !== 'undefined') {{
    pakoLoaded = true;
  }}
  
  // Decode binary coordinates helper (with optional zlib decompression)
  function decodeBinaryCoords(binaryStr, count) {{
    if (!binaryStr || count === 0) return null;
    const binary = atob(binaryStr);
    let bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    if (USE_COMPRESSION && typeof pako !== 'undefined') {{
      try {{ bytes = pako.inflate(bytes); }} catch (e) {{ console.error("[Compression] fail:", e); return null; }}
    }}
    return new Float32Array(bytes.buffer);
  }}

  // Decode zlib-compressed base64 uint8 array (GEX/obs data)
  function decodeBinaryUint8(b64, expectedCount) {{
    if (!b64) return null;
    try {{
      const raw = atob(b64);
      let bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      if (typeof pako !== 'undefined') bytes = pako.inflate(bytes);
      return bytes;  // Uint8Array
    }} catch(e) {{ console.error("[Binary] Decode failed:", e); return null; }}
  }}
  
  // Store current GEX vmax for colorbar labels
  let currentGexVmax = 0;

  // GEX cache for instant gene switching
  const gexCache = new Map();
  const GEX_CACHE_MAX = 10;
  function cacheGex(gene, arr, vmax) {{
    if (gexCache.size >= GEX_CACHE_MAX) gexCache.delete(gexCache.keys().next().value);
    gexCache.set(gene, {{ u8: arr, vm: vmax }});
  }}
  function activateCachedGene(gene) {{
    const entry = gexCache.get(gene);
    if (!entry) return false;
    currentGexGene = gene;
    currentGexVmax = entry.vm;
    gexValues.fill(0);
    const d = entry.u8;
    for (let k = 0; k < d.length; k++) {{
      if (d[k] > 0) {{
        const ap = cellIdToIndex.get(k);
        if (ap !== undefined) gexValues[ap] = d[k];
      }}
    }}
    return true;
  }}
  
  let points = [];  // Keep for compatibility but not used

  // ----------------------------
  // Navigation state (zoom, pan, rotation)
  // ----------------------------
  let zoom = 1.0;
  let lastZoomLevel = 1.0;  // Track for zoom-out detection
  let panX = 0;
  let panY = 0;
  let rotation = 0;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  let rotationMode = false;
  let lastRotationKeyPress = 0;
  
  // ----------------------------
  // Animation state
  // ----------------------------
  let isAnimating = false;
  let animationStartTime = 0;
  let animationDuration = 800; // 0.8 second for embedding transitions
  let startPoints = null;
  let endPoints = null;
  
  // Embedding animation state
  let animationSourceEmbedding = null;
  let animationTargetEmbedding = null;
  
  // Color animation
  let colorAnimating = false;
  let colorAnimStart = 0;
  const COLOR_ANIM_MS = 500; // 0.5 seconds
  let oldColors = null;
  let newColors = null;
  
  // Helper function for smooth interpolation
  function lerp(a, b, t) {{
    return a + (b - a) * t;
  }}
  
  // Easing function (ease-in-out)
  function easeInOutQuad(t) {{
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }}
  
  // Get position array for an embedding
  function getEmbeddingPositions(embeddingName) {{
    if (embeddingName === "spatial") return posSpatial;
    if (embeddingName === "umap") return posUmap;
    if (embeddingName === "pca") return posPca;
    if (embeddingName === "layout") return posLayout;
    if (embeddingName === "layout_snapshot") return posLayoutSnapshot;
    return posSpatial;
  }}

  // ----------------------------
  // Plot state (supports layered rendering: obs + GEX overlay)
  // ----------------------------
  window["_plotState_" + iframeId] = {{
    // Obs layer (base)
    obs: {{
      mode: null,           // "categorical" | "continuous" | null
      values: null,         // per-point values
      colors: null,         // categorical palette
      categories: null,
      enabled: null,        // boolean array for categorical toggles
      opacity: 1.0
    }},
    // GEX layer (overlay)
    gex: {{
      values: null,
      opacity: 1.0,
      colormap: "viridis"
    }},
    label: "Embedding: spatial",
    pointSize: 1.1,
    selectionIndices: null  // array of selected point indices (acts as mask)
  }};

  function setLabel(text) {{
    const lab = document.getElementById("plot_label_" + iframeId);
    if (lab) lab.textContent = text;
  }}

  // Camera button - download canvas as PNG with custom filename
  const cameraBtn = document.getElementById("camera_btn_" + iframeId);
  if (cameraBtn) {{
    cameraBtn.addEventListener("click", () => {{
      // Prompt for filename
      const defaultName = "spatial_plot_" + new Date().toISOString().slice(0,10);
      const filename = prompt("Enter filename for PNG (without extension):", defaultName);
      
      // If user cancelled or entered empty string, don't download
      if (!filename || filename.trim() === "") {{
        return;
      }}
      
      // Force redraw labels to ensure they're up to date
      drawSampleLabels();
      
      // Create composite canvas with WebGL + labels
      const compositeCanvas = document.createElement("canvas");
      compositeCanvas.width = canvas.width;
      compositeCanvas.height = canvas.height;
      const compCtx = compositeCanvas.getContext("2d");
      
      // Transparent background (WebGL canvas already has transparent bg)
      compCtx.clearRect(0, 0, compositeCanvas.width, compositeCanvas.height);
      
      // Draw WebGL canvas
      compCtx.drawImage(canvas, 0, 0);
      
      // Draw label overlay on top
      if (labelOverlay) {{
        compCtx.drawImage(labelOverlay, 0, 0);
      }}
      
      // Download with user's filename
      const link = document.createElement("a");
      link.download = filename.trim() + ".png";
      link.href = compositeCanvas.toDataURL("image/png");
      link.click();
    }});
  }}

  // ----------------------------
  // Plot updater: receives Python callback payloads
  // ----------------------------
  window["updatePlot_" + iframeId] = function(payload) {{
    if (!payload || !payload.type) return;

    // CHUNKED: Handle chunk data with ALL embeddings - uses processChunkData
    if (payload.type === "chunk_data") {{
      const chunkId = payload.chunk;
      
      // Decode ALL embedding coordinates
      const spatialCoords = payload.spatial_binary ? decodeBinaryCoords(payload.spatial_binary, payload.count) : null;
      const umapCoords = payload.umap_binary ? decodeBinaryCoords(payload.umap_binary, payload.count) : null;
      const pcaCoords = payload.pca_binary ? decodeBinaryCoords(payload.pca_binary, payload.count) : null;
      
      if (!spatialCoords && !umapCoords && !pcaCoords) {{
        console.error("Failed to decode chunk coordinates");
        isLoadingChunk = false;
        setTimeout(() => requestNextChunk(), 100);
        return;
      }}
      
      // Convert indices to Int32Array for processChunkData
      const indices = new Int32Array(payload.indices);
      
      // Use shared processChunkData (struct-of-arrays)
      // Decode sample IDs if present
      let chunkSids = null;
      if (payload.sids_b64) {{
        try {{
          const raw = atob(payload.sids_b64);
          const bytes = new Uint8Array(raw.length);
          for (let j = 0; j < raw.length; j++) bytes[j] = raw.charCodeAt(j);
          const inflated = (typeof pako !== 'undefined') ? pako.inflate(bytes) : bytes;
          const aligned = new ArrayBuffer(payload.count * 2);
          new Uint8Array(aligned).set(inflated.subarray(0, payload.count * 2));
          chunkSids = new Uint16Array(aligned);
        }} catch(e) {{ console.warn("[Chunk] Failed to decode sample IDs"); }}
      }}
      processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, payload.count, chunkSids, payload.requestId);
      
      // Store palette if provided
      if (payload.obs_colors) {{
        currentPalette = payload.obs_colors;
        currentCategories = payload.obs_categories;
        currentObsColumn = payload.obs_column;
      }}
      if (payload.gex_gene) {{
        currentGexGene = payload.gex_gene;
      }}
      
      return;
    }}

    // LAYOUT: JS-side compute request from iframe
    if (payload.type === "compute_layout") {{
      showLayoutLoading();
      
      // Check which columns need metadata
      const neededCols = [payload.group_by, payload.columns, payload.rows].filter(c => c && c.trim());
      const missingCols = neededCols.filter(c => !sampleMetaCache[c]);
      
      if (missingCols.length > 0) {{
        // Fetch missing metadata via bridge, then compute
        let fetched = 0;
        const pendingParams = payload;
        
        // Store a listener for sample_meta responses
        const metaHandler = (metaPayload) => {{
          if (metaPayload.type === "sample_meta" && metaPayload.valid) {{
            sampleMetaCache[metaPayload.column] = {{
              cats: metaPayload.cats,
              codes: metaPayload.codes
            }};
          }}
          fetched++;
          if (fetched >= missingCols.length) {{
            // All fetched — now compute
            window["_layoutMetaHandler_" + iframeId] = null;
            doLayoutCompute(pendingParams);
            hideLayoutLoading();
          }}
        }};
        window["_layoutMetaHandler_" + iframeId] = metaHandler;
        
        // Request each missing column
        for (const col of missingCols) {{
          const iframeEl = document.getElementById(iframeId);
          if (iframeEl && iframeEl.contentWindow) {{
            // Click hidden sampleMetaBtn with column data
            const reqId = "sampleMetaBtn";
            window["_requests_" + iframeId].push({{
              buttonId: reqId,
              data: {{ column: col }},
              type: "button_click",
              iframeId: iframeId
            }});
          }}
        }}
      }} else {{
        // All cached — compute immediately
        doLayoutCompute(payload);
        hideLayoutLoading();
      }}
      return;
    }}
    
    // Handle sample_meta response from Python
    if (payload.type === "sample_meta") {{
      const handler = window["_layoutMetaHandler_" + iframeId];
      if (handler) {{
        handler(payload);
      }} else if (payload.valid) {{
        sampleMetaCache[payload.column] = {{
          cats: payload.cats,
          codes: payload.codes
        }};
      }}
      return;
    }}
    
    // Toggle sample labels
    if (payload.type === "toggle_sample_labels") {{
      showSampleLabels = !!payload.show;
      draw();
      return;
    }}
    
    // LAYOUT: Python-returned coords (for loading saved layouts from obsm)
    if (payload.type === "layout_coords") {{
      const decoded = decodeBinaryCoords(payload.coords_binary, payload.count);
      if (!decoded || decoded.length !== TOTAL_CELLS * 2) {{
        console.error("[Layout] Coordinate count mismatch");
        return;
      }}
      posLayout.set(decoded);
      layoutHasData = true;
      if (payload.bounds) METADATA["layout"] = payload.bounds;
      layoutSampleLabels = payload.sample_labels || [];
      layoutLabelPositions = payload.sample_label_positions || [];
      
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "layout_applied",
          sample_labels: layoutSampleLabels,
          layout_name: payload.layout_name || null,
        }}, "*");
      }}
      
      if (currentEmbedding === "layout" && !isAnimating) {{
        markGPUDirty(); draw();
      }} else {{
        animationSourceEmbedding = currentEmbedding;
        animationTargetEmbedding = "layout";
        currentEmbedding = "layout";
        isAnimating = true;
        animationStartTime = performance.now();
        zoom = 1.0; panX = 0; panY = 0; rotation = 0;
        requestAnimationFrame(animateEmbeddingTransition);
      }}
      setLabel("Layout");
      return;
    }}

    // LAYOUT SAVE: store current layout in JS memory (no Python round-trip)
    if (payload.type === "save_layout_request") {{
      if (!layoutHasData) return;
      const name = payload.name;
      savedLayouts[name] = {{
        positions: new Float32Array(posLayout.subarray(0, loadedCount * 2)),
        labels: [...layoutSampleLabels],
        labelPositions: layoutLabelPositions.map(p => [...p]),
        groupLabels: layoutGroupLabels.map(g => ({{...g}})),
        colLabels: layoutColLabels.map(c => ({{...c}})),
        rowLabels: layoutRowLabels.map(r => ({{...r}})),
        axisInfo: layoutAxisInfo ? {{...layoutAxisInfo}} : null,
        bounds: METADATA["layout"] ? {{...METADATA["layout"]}} : null,
        params: layoutParams ? {{...layoutParams}} : null,
      }};
      activeLayoutName = name;
      console.log("[Layout] Saved:", name, "(" + Object.keys(savedLayouts).length + " total)");
      
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "layout_saved",
          name: name,
          all_names: Object.keys(savedLayouts),
        }}, "*");
      }}
      return;
    }}
    
    // LAYOUT SWITCH: restore a saved layout from JS memory with animation
    if (payload.type === "switch_to_saved_layout") {{
      const name = payload.name;
      const saved = savedLayouts[name];
      if (!saved) return;
      
      // Snapshot current layout positions for animation source
      if (layoutHasData && currentEmbedding === "layout") {{
        posLayoutSnapshot.set(posLayout.subarray(0, loadedCount * 2));
        METADATA["layout_snapshot"] = METADATA["layout"] ? {{...METADATA["layout"]}} : null;
      }}
      
      // Load the saved layout into posLayout (target)
      posLayout.set(saved.positions);
      layoutHasData = true;
      layoutSampleLabels = [...saved.labels];
      layoutLabelPositions = saved.labelPositions.map(p => [...p]);
      layoutGroupLabels = saved.groupLabels ? saved.groupLabels.map(g => ({{...g}})) : [];
      layoutColLabels = saved.colLabels ? saved.colLabels.map(c => ({{...c}})) : [];
      layoutRowLabels = saved.rowLabels ? saved.rowLabels.map(r => ({{...r}})) : [];
      layoutAxisInfo = saved.axisInfo ? {{...saved.axisInfo}} : null;
      if (saved.bounds) METADATA["layout"] = {{...saved.bounds}};
      if (saved.params) layoutParams = {{...saved.params}};
      activeLayoutName = name;
      
      // Animate: if already on layout, animate from snapshot to new layout
      if (currentEmbedding === "layout" && METADATA["layout_snapshot"] && !isAnimating) {{
        animationSourceEmbedding = "layout_snapshot";
        animationTargetEmbedding = "layout";
        isAnimating = true;
        animationStartTime = performance.now();
        requestAnimationFrame(animateEmbeddingTransition);
      }} else if (currentEmbedding !== "layout") {{
        // Animate from current embedding to layout
        animationSourceEmbedding = currentEmbedding;
        animationTargetEmbedding = "layout";
        currentEmbedding = "layout";
        isAnimating = true;
        animationStartTime = performance.now();
        zoom = 1.0; panX = 0; panY = 0; rotation = 0;
        requestAnimationFrame(animateEmbeddingTransition);
      }} else {{
        // Already on layout, just redraw
        markGPUDirty(); draw();
      }}
      setLabel("Layout: " + name);
      
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "layout_applied",
          layout_name: name,
          sample_labels: layoutSampleLabels,
        }}, "*");
      }}
      return;
    }}
    
    // LAYOUT DELETE: remove from JS memory
    if (payload.type === "delete_saved_layout") {{
      const name = payload.name;
      delete savedLayouts[name];
      if (activeLayoutName === name) activeLayoutName = null;
      
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "layout_deleted",
          name: name,
        }}, "*");
      }}
      return;
    }}
    
    // SAVE TO OBSM: send sample centroids to Python (not all cell positions)
    // Python will reconstruct full cell positions using spatial offsets
    if (payload.type === "save_to_obsm") {{
      if (!layoutHasData) {{
        console.error("[Layout] save_to_obsm: no layout data");
        return;
      }}
      const name = payload.name;
      const src = savedLayouts[name] || null;
      
      // We need to compute sample centroids from the layout
      // Use layoutLabelPositions which stores [x,y] for each sample
      let samplePositions;
      if (src && src.labelPositions && src.labelPositions.length > 0) {{
        samplePositions = src.labelPositions;
        console.log("[Layout] save_to_obsm: using saved layout", name, "samples:", samplePositions.length);
      }} else if (layoutLabelPositions && layoutLabelPositions.length > 0) {{
        samplePositions = layoutLabelPositions;
        console.log("[Layout] save_to_obsm: using current layout samples:", samplePositions.length);
      }} else {{
        console.error("[Layout] save_to_obsm: no sample positions available");
        return;
      }}
      
      // Create Float32Array of sample centroids [x0,y0,x1,y1,...]
      const nSamples = samplePositions.length;
      const centroidData = new Float32Array(nSamples * 2);
      for (let i = 0; i < nSamples; i++) {{
        centroidData[i * 2] = samplePositions[i][0];
        centroidData[i * 2 + 1] = samplePositions[i][1];
      }}
      
      // Compress and encode
      const centroidBytes = new Uint8Array(centroidData.buffer);
      console.log("[Layout] save_to_obsm: centroid bytes:", centroidBytes.length);
      
      let compressedB64 = "";
      try {{
        if (typeof pako !== 'undefined') {{
          const compressed = pako.deflate(centroidBytes);
          let binary = "";
          for (let i = 0; i < compressed.length; i++) binary += String.fromCharCode(compressed[i]);
          compressedB64 = btoa(binary);
        }} else {{
          let binary = "";
          for (let i = 0; i < centroidBytes.length; i++) binary += String.fromCharCode(centroidBytes[i]);
          compressedB64 = btoa(binary);
        }}
        console.log("[Layout] save_to_obsm: compressed to", compressedB64.length, "chars");
      }} catch(e) {{ 
        console.error("[Layout] Encoding failed:", e); 
        return; 
      }}
      
      // Also send the sample labels in order so Python can match them
      const sampleLabels = src ? src.labels : layoutSampleLabels;
      
      window["_requests_" + iframeId].push({{
        buttonId: "obsmBtn",
        data: {{ 
          name: name, 
          centroids_b64: compressedB64,
          sample_labels: sampleLabels,
          n_samples: nSamples
        }},
        type: "button_click",
        iframeId: iframeId
      }});
      console.log("[Layout] save_to_obsm: queued request for", name, "with", nSamples, "samples");
      return;
    }}
    
    // ADJUST LAYOUT: live gap/transpose changes without full recompute
    if (payload.type === "adjust_layout") {{
      if (!layoutHasData || !layoutParams) return;
      // Merge new gap/transpose into existing params
      const newParams = {{...layoutParams}};
      if (payload.gap != null) newParams.gap = payload.gap;
      if (payload.transpose != null) newParams.transpose = payload.transpose;
      
      // Snapshot current for smooth animation
      posLayoutSnapshot.set(posLayout.subarray(0, loadedCount * 2));
      METADATA["layout_snapshot"] = METADATA["layout"] ? {{...METADATA["layout"]}} : null;
      
      // Recompute with adjusted params
      computeLayoutJS(newParams);
      
      // Animate the adjustment
      if (!isAnimating) {{
        animationSourceEmbedding = "layout_snapshot";
        animationTargetEmbedding = "layout";
        isAnimating = true;
        animationStartTime = performance.now();
        requestAnimationFrame(animateEmbeddingTransition);
      }}
      return;
    }}

    // LAYOUT: saved/deleted/obsm confirmation (forward from Python bridge to iframe)
    if (payload.type === "layout_obsm_saved") {{
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage(payload, "*");
      }}
      return;
    }}

    // Switch embedding - SMOOTH ANIMATED TRANSITION
    if (payload.type === "set_embedding") {{
      const name = (payload.embedding || "spatial");
      
      // Don't switch if already on this embedding
      if (currentEmbedding === name) return;
      
      // Don't animate if we're still loading or if animation already in progress
      if (!isFullyLoaded || isAnimating) {{
        currentEmbedding = name;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        rotation = 0;
        markGPUDirty();
        draw();
        return;
      }}
      
      console.log(`[Embedding] Animating transition: ${{currentEmbedding}} → ${{name}}`);
      
      // Store the source embedding for animation
      animationSourceEmbedding = currentEmbedding;
      animationTargetEmbedding = name;
      
      // Start animation
      isAnimating = true;
      animationStartTime = performance.now();
      
      // Reset view for the transition
      zoom = 1.0;
      panX = 0;
      panY = 0;
      rotation = 0;
      
      // Start animation loop
      requestAnimationFrame(animateEmbeddingTransition);
      return;
    }}

    // Clear overlay (back to neutral grey)
    if (payload.type === "clear_plot") {{
      obsValues.fill(0); gexValues.fill(0);
      currentObsColumn = null; currentPalette = null; currentCategories = null; currentGexGene = null; currentGexVmax = 0;
      setLabel("Embedding: " + currentEmbedding); markGPUDirty(); draw(); return;
    }}
    if (payload.type === "clear_obs") {{
      obsValues.fill(0); currentObsColumn = null; currentPalette = null; currentCategories = null;
      setLabel(currentGexGene ? "GEX: "+currentGexGene : "Embedding: "+currentEmbedding);
      markGPUDirty(); draw(); return;
    }}
    if (payload.type === "clear_gex") {{
      gexValues.fill(0); currentGexGene = null; currentGexVmax = 0;
      setLabel(currentObsColumn ? "obs: "+currentObsColumn : "Embedding: "+currentEmbedding);
      markGPUDirty(); draw(); return;
    }}

    // GEX values — decode compressed uint8 into gexValues[], forward to iframe
    if (payload.type === "gex_values") {{
      currentGexGene = payload.gene || null;
      currentGexVmax = payload.vmax || 0;
      gexValues.fill(0);
      const decoded = decodeBinaryUint8(payload.values_b64, payload.count || 0);
      if (decoded) {{
        cacheGex(payload.gene, decoded, payload.vmax || 0);
        for (let k = 0; k < decoded.length; k++) {{
          if (decoded[k] > 0) {{
            const ap = cellIdToIndex.get(k);
            if (ap !== undefined) gexValues[ap] = decoded[k];
          }}
        }}
        console.log(`[GEX] ${{payload.gene}}: cached (${{gexCache.size}}/${{GEX_CACHE_MAX}})`);
      }}
      setLabel("GEX: " + (payload.gene || ""));
      // Forward gene name + vmax to iframe (no big arrays — iframe tracks names only)
      const iframeEl = document.querySelector("iframe");
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "gex_loaded", gene: payload.gene, vmax: payload.vmax
        }}, "*");
      }}
      hideGexLoading();  // Hide loading overlay
      markGPUDirty(); draw(); return;
    }}

    // Obs values — decode compressed uint8 category codes
    if (payload.type === "obs_values") {{
      currentObsColumn = payload.column;
      currentPalette = payload.colors;
      currentCategories = payload.categories;
      obsValues.fill(0);
      const decoded = decodeBinaryUint8(payload.codes_b64, payload.count || 0);
      if (decoded) {{
        for (let k = 0; k < decoded.length; k++) {{
          if (decoded[k] > 0) {{
            const ap = cellIdToIndex.get(k);
            if (ap !== undefined) obsValues[ap] = decoded[k];
          }}
        }}
        console.log(`[OBS] ${{payload.column}}: ${{decoded.length}} cells`);
      }}
      markGPUDirty(); draw(); return;
    }}
  }};

  // ----------------------------
  // WebGL Plotting (GPU-accelerated, handles millions of points at 60fps)
  // ----------------------------
  const canvas = document.getElementById("plot_canvas_" + iframeId);
  const panel = document.getElementById("plot_panel_" + iframeId);
  
  // Initialize WebGL context
  const gl = canvas.getContext("webgl", {{ antialias: true, alpha: true, preserveDrawingBuffer: true }}) || 
             canvas.getContext("experimental-webgl", {{ antialias: true, alpha: true, preserveDrawingBuffer: true }});
  
  if (!gl) {{
    console.error("WebGL not supported!");
  }}
  
  // Keep 2D context for minimap only
  const minimap = document.getElementById("minimap_" + iframeId);
  const minimapCtx = minimap.getContext("2d");
  minimap.width = 120;
  minimap.height = 120;
  
  // Label overlay for sample names (2D canvas on top of WebGL)
  const labelOverlay = document.getElementById("label_overlay_" + iframeId);
  const labelCtx = labelOverlay ? labelOverlay.getContext("2d") : null;

  // ----------------------------
  // WebGL Shaders
  // ----------------------------
  const vertexShaderSource = `
    attribute vec2 a_position;
    attribute vec3 a_color;
    
    uniform mat3 u_matrix;
    uniform float u_pointSize;
    
    varying vec3 v_color;
    
    void main() {{
      vec3 pos = u_matrix * vec3(a_position, 1.0);
      gl_Position = vec4(pos.xy, 0.0, 1.0);
      gl_PointSize = u_pointSize;
      v_color = a_color;
    }}
  `;
  
  const fragmentShaderSource = `
    precision mediump float;
    varying vec3 v_color;
    uniform float u_opacity;
    void main() {{
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;
      float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
      gl_FragColor = vec4(v_color, alpha * u_opacity);
    }}
  `;
  
  // Compile shader
  function compileShader(gl, type, source) {{
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
      console.error("Shader compile error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }}
    return shader;
  }}
  
  // Create program
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
    console.error("Program link error:", gl.getProgramInfoLog(program));
  }}
  
  // Get attribute/uniform locations
  const a_position = gl.getAttribLocation(program, "a_position");
  const a_color = gl.getAttribLocation(program, "a_color");
  const u_matrix = gl.getUniformLocation(program, "u_matrix");
  const u_pointSize = gl.getUniformLocation(program, "u_pointSize");
  const u_opacity = gl.getUniformLocation(program, "u_opacity");
  
  // Create buffers
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  
  // GPU data tracking
  let gpuPointCount = 0;
  let gpuDataDirty = true;  // Flag to rebuild GPU buffers
  
  // Mark GPU data as needing rebuild
  function markGPUDirty() {{
    gpuDataDirty = true;
  }}

  function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}
  const COLORMAPS_GL = {{
    viridis:[[0,[.267,.004,.329]],[.13,[.278,.173,.478]],[.25,[.231,.318,.545]],[.38,[.173,.443,.557]],[.5,[.129,.565,.553]],[.63,[.153,.678,.506]],[.75,[.361,.784,.388]],[.88,[.667,.863,.196]],[1,[.992,.906,.145]]],
    plasma:[[0,[.051,.031,.529]],[.13,[.329,.008,.639]],[.25,[.545,.039,.647]],[.38,[.725,.196,.537]],[.5,[.859,.361,.408]],[.63,[.957,.533,.286]],[.75,[.996,.737,.169]],[1,[.941,.976,.129]]],
    inferno:[[0,[0,0,.016]],[.13,[.078,.043,.208]],[.25,[.227,.047,.376]],[.38,[.396,.082,.431]],[.5,[.584,.153,.376]],[.63,[.784,.259,.235]],[.75,[.929,.475,.106]],[.88,[.984,.749,.141]],[1,[.988,1,.643]]],
    magma:[[0,[0,0,.016]],[.13,[.071,.051,.196]],[.25,[.2,.063,.408]],[.38,[.384,.09,.502]],[.5,[.584,.161,.475]],[.63,[.765,.282,.408]],[.75,[.902,.478,.349]],[.88,[.976,.729,.455]],[1,[.988,.992,.749]]],
    cividis:[[0,[0,.125,.302]],[.25,[.239,.263,.42]],[.5,[.486,.482,.471]],[.75,[.761,.702,.392]],[1,[.992,.918,.271]]],
    coolwarm:[[0,[.231,.298,.753]],[.25,[.486,.624,.902]],[.5,[.867,.867,.867]],[.75,[.855,.51,.416]],[1,[.706,.016,.149]]],
    hot:[[0,[.043,0,0]],[.33,[.902,0,0]],[.66,[1,.824,0]],[1,[1,1,1]]],
    YlOrRd:[[0,[1,1,.8]],[.25,[.996,.8,.361]],[.5,[.992,.553,.235]],[.75,[.89,.102,.11]],[1,[.502,0,.149]]],
    YlGnBu:[[0,[1,1,.851]],[.25,[.631,.855,.706]],[.5,[.255,.714,.769]],[.75,[.133,.369,.659]],[1,[.031,.114,.345]]],
    Greens:[[0,[.969,.988,.961]],[.25,[.729,.894,.702]],[.5,[.455,.769,.463]],[.75,[.137,.545,.271]],[1,[0,.267,.106]]],
    Blues:[[0,[.969,.984,1]],[.25,[.62,.792,.882]],[.5,[.259,.573,.776]],[.75,[.031,.318,.612]],[1,[.031,.188,.42]]],
    Reds:[[0,[1,.961,.941]],[.25,[.988,.682,.569]],[.5,[.984,.416,.29]],[.75,[.796,.094,.114]],[1,[.404,0,.051]]],
    Greys:[[0,[1,1,1]],[.25,[.8,.8,.8]],[.5,[.588,.588,.588]],[.75,[.322,.322,.322]],[1,[0,0,0]]],
    turbo:[[0,[.188,.071,.231]],[.13,[.271,.459,.706]],[.25,[.125,.725,.686]],[.38,[.427,.91,.384]],[.5,[.796,.922,.102]],[.63,[.992,.714,.047]],[.75,[.973,.424,.094]],[.88,[.78,.129,.149]],[1,[.478,.016,.012]]],
    Spectral:[[0,[.62,.004,.259]],[.25,[.957,.427,.263]],[.5,[1,1,.749]],[.75,[.4,.761,.647]],[1,[.369,.31,.635]]],
    RdBu:[[0,[.404,0,.122]],[.25,[.792,.306,.247]],[.5,[.969,.969,.969]],[.75,[.294,.58,.769]],[1,[.02,.188,.38]]],
    jet:[[0,[0,0,.502]],[.12,[0,0,1]],[.37,[0,1,1]],[.5,[0,1,0]],[.63,[1,1,0]],[.87,[1,0,0]],[1,[.502,0,0]]],
    rainbow:[[0,[.502,0,1]],[.25,[0,.502,1]],[.5,[0,1,0]],[.75,[1,1,0]],[1,[1,0,0]]]
  }};
  let currentColormap = "viridis";
  function cmapRGB(t, cmapName) {{
    const stops = COLORMAPS_GL[cmapName || currentColormap] || COLORMAPS_GL.viridis;
    t = clamp01(t);
    for (let i = 0; i < stops.length - 1; i++) {{
      const a = stops[i], b = stops[i+1];
      if (t >= a[0] && t <= b[0]) {{
        const u = (t - a[0]) / (b[0] - a[0] || 1);
        return [a[1][0]+u*(b[1][0]-a[1][0]), a[1][1]+u*(b[1][1]-a[1][1]), a[1][2]+u*(b[1][2]-a[1][2])];
      }}
    }}
    return stops[stops.length-1][1];
  }}

  // Parse CSS color to RGB (0-1)
  function parseColor(colorStr) {{
    if (!colorStr) return [0.6, 0.6, 0.6];
    
    // Handle rgb(r,g,b)
    const rgbMatch = colorStr.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (rgbMatch) {{
      return [parseInt(rgbMatch[1])/255, parseInt(rgbMatch[2])/255, parseInt(rgbMatch[3])/255];
    }}
    
    // Handle rgba(r,g,b,a)
    const rgbaMatch = colorStr.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
    if (rgbaMatch) {{
      return [parseInt(rgbaMatch[1])/255, parseInt(rgbaMatch[2])/255, parseInt(rgbaMatch[3])/255];
    }}
    
    // Handle hex
    if (colorStr.startsWith('#')) {{
      const hex = colorStr.slice(1);
      if (hex.length === 3) {{
        return [parseInt(hex[0]+hex[0],16)/255, parseInt(hex[1]+hex[1],16)/255, parseInt(hex[2]+hex[2],16)/255];
      }} else if (hex.length >= 6) {{
        return [parseInt(hex.slice(0,2),16)/255, parseInt(hex.slice(2,4),16)/255, parseInt(hex.slice(4,6),16)/255];
      }}
    }}
    
    // Handle hsl
    const hslMatch = colorStr.match(/hsl\(([0-9.]+),\s*([0-9.]+)%,\s*([0-9.]+)%\)/);
    if (hslMatch) {{
      const h = parseFloat(hslMatch[1]) / 360;
      const s = parseFloat(hslMatch[2]) / 100;
      const l = parseFloat(hslMatch[3]) / 100;
      return hslToRgb(h, s, l);
    }}
    
    return [0.6, 0.6, 0.6];  // Default grey
  }}
  
  function hslToRgb(h, s, l) {{
    let r, g, b;
    if (s === 0) {{
      r = g = b = l;
    }} else {{
      const hue2rgb = (p, q, t) => {{
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      }};
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }}
    return [r, g, b];
  }}

  function fallbackCatColorRGB(k) {{
    const h = (k * 137.508) % 360;
    return hslToRgb(h / 360, 0.55, 0.55);
  }}
  
  // For backward compatibility (used by minimap)
  function fallbackCatColor(k) {{
    const h = (k * 137.508) % 360;
    return `hsl(${{h}}, 55%, 55%)`;
  }}

  function resizeCanvas() {{
    const rect = panel.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);
    // Resize label overlay too
    if (labelOverlay) {{
      labelOverlay.width = canvas.width;
      labelOverlay.height = canvas.height;
    }}
    draw();
  }}
  
  // Draw sample labels on 2D overlay (only when in layout mode)
  function drawSampleLabels() {{
    if (!labelCtx || !labelOverlay) return;
    labelCtx.clearRect(0, 0, labelOverlay.width, labelOverlay.height);
    
    if (!showSampleLabels) return;
    if (!SAMPLE_META || !SAMPLE_META.names) return;
    
    // Choose which label positions to use
    let labels, positions;
    if (currentEmbedding === "layout" && layoutHasData && layoutSampleLabels.length > 0) {{
      labels = layoutSampleLabels;
      positions = layoutLabelPositions;
    }} else if (currentEmbedding === "spatial" && SAMPLE_META.names) {{
      labels = SAMPLE_META.names;
      positions = SAMPLE_META.names.map((_, i) => [SAMPLE_META.cx[i], SAMPLE_META.cy[i]]);
    }} else {{
      return;
    }}
    
    const embedMeta = METADATA[currentEmbedding];
    if (!embedMeta) return;
    
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    const dpr = window.devicePixelRatio || 1;
    
    const minX = embedMeta.minX, maxX = embedMeta.maxX;
    const minY = embedMeta.minY, maxY = embedMeta.maxY;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const pad = 12;
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    const scale = baseScale * zoom;
    const offX = (W - spanX * scale) / 2 + panX;
    const offY = (H - spanY * scale) / 2 + panY;
    
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    const cx = W / 2;
    const cy = H / 2;
    
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    labelCtx.font = "bold 11px ui-monospace, monospace";
    labelCtx.textAlign = "center";
    labelCtx.textBaseline = "middle";
    
    for (let i = 0; i < labels.length; i++) {{
      const lbl = labels[i];
      const [dx, dy] = positions[i];
      
      let px = offX + (dx - minX) * scale;
      let py = offY + (dy - minY) * scale;
      
      const rx = cos * (px - cx) - sin * (py - cy) + cx;
      const ry = sin * (px - cx) + cos * (py - cy) + cy;
      
      // Skip if off-screen
      if (rx < -50 || rx > W + 50 || ry < -20 || ry > H + 20) continue;
      
      const tw = labelCtx.measureText(lbl).width + 8;
      
      labelCtx.fillStyle = "rgba(0,0,0,0.55)";
      labelCtx.beginPath();
      labelCtx.roundRect(rx - tw/2, ry - 8, tw, 16, 4);
      labelCtx.fill();
      
      labelCtx.fillStyle = "#fff";
      labelCtx.fillText(lbl, rx, ry);
    }}
    
    // Draw group labels (larger, above each group) - mid grey text
    if (currentEmbedding === "layout" && layoutGroupLabels.length > 0) {{
      labelCtx.font = "bold 13px system-ui, sans-serif";
      labelCtx.textAlign = "center";
      labelCtx.textBaseline = "bottom";
      
      for (const grp of layoutGroupLabels) {{
        let px = offX + (grp.x - minX) * scale;
        let py = offY + (grp.y - minY) * scale;
        
        const rx = cos * (px - cx) - sin * (py - cy) + cx;
        const ry = sin * (px - cx) + cos * (py - cy) + cy;
        
        if (rx < -100 || rx > W + 100 || ry < -50 || ry > H + 50) continue;
        
        labelCtx.fillStyle = "rgba(100,100,100,0.9)";
        labelCtx.fillText(grp.name, rx, ry);
      }}
    }}
    
    // Draw column labels (along top of first group) - mid grey, smaller
    if (currentEmbedding === "layout" && layoutColLabels.length > 0) {{
      labelCtx.font = "11px system-ui, sans-serif";
      labelCtx.textAlign = "center";
      labelCtx.textBaseline = "bottom";
      
      for (const col of layoutColLabels) {{
        let px = offX + (col.x - minX) * scale;
        let py = offY + (col.y - minY) * scale;
        
        const rx = cos * (px - cx) - sin * (py - cy) + cx;
        const ry = sin * (px - cx) + cos * (py - cy) + cy;
        
        if (rx < -50 || rx > W + 50 || ry < -30 || ry > H + 30) continue;
        
        labelCtx.fillStyle = "rgba(120,120,120,0.85)";
        labelCtx.fillText(col.name, rx, ry);
      }}
    }}
    
    // Draw row labels (along left of first group) - mid grey, smaller
    if (currentEmbedding === "layout" && layoutRowLabels.length > 0) {{
      labelCtx.font = "11px system-ui, sans-serif";
      labelCtx.textAlign = "right";
      labelCtx.textBaseline = "middle";
      
      for (const row of layoutRowLabels) {{
        let px = offX + (row.x - minX) * scale;
        let py = offY + (row.y - minY) * scale;
        
        const rx = cos * (px - cx) - sin * (py - cy) + cx;
        const ry = sin * (px - cx) + cos * (py - cy) + cy;
        
        if (rx < -50 || rx > W + 50 || ry < -30 || ry > H + 30) continue;
        
        labelCtx.fillStyle = "rgba(120,120,120,0.85)";
        labelCtx.fillText(row.name, rx, ry);
      }}
    }}
    
    labelCtx.restore();
  }}
  
  // ============================================================
  // JS-SIDE LAYOUT COMPUTATION (instant, no Python round-trip)
  // ============================================================
  // Cache for on-demand sample metadata columns
  const sampleMetaCache = {{}};  // column -> cached meta
  
  function doLayoutCompute(params) {{
    const success = computeLayoutJS(params);
    if (!success) return;
    
    const iframeEl = document.getElementById(iframeId);
    if (iframeEl && iframeEl.contentWindow) {{
      iframeEl.contentWindow.postMessage({{
        type: "layout_applied",
        sample_labels: layoutSampleLabels,
      }}, "*");
    }}
    
    if (currentEmbedding === "layout" && !isAnimating) {{
      markGPUDirty(); draw();
    }} else {{
      animationSourceEmbedding = currentEmbedding;
      animationTargetEmbedding = "layout";
      currentEmbedding = "layout";
      isAnimating = true;
      animationStartTime = performance.now();
      zoom = 1.0; panX = 0; panY = 0; rotation = 0;
      requestAnimationFrame(animateEmbeddingTransition);
    }}
    setLabel("Layout");
  }}
  
  function computeLayoutJS(params) {{
    if (!SAMPLE_META || !SAMPLE_META.names || SAMPLE_META.names.length === 0) {{
      console.warn("[Layout] No sample metadata available");
      return false;
    }}
    
    const names = SAMPLE_META.names;
    const nSamp = names.length;
    const cxArr = SAMPLE_META.cx;
    const cyArr = SAMPLE_META.cy;
    const wArr = SAMPLE_META.w;
    const hArr = SAMPLE_META.h;
    
    const groupCol = params.group_by || "";
    const groupNCols = params.group_ncols || 2;
    const colsCol = params.columns || "";      // Determines columns within group
    const rowsCol = params.rows || "";         // Determines row order within column
    const gap = params.gap != null ? params.gap : 2.0;
    const transpose = !!params.transpose;
    
    // Use median sample bbox for cell size
    const sortedW = Array.from(wArr).sort((a,b) => a-b);
    const sortedH = Array.from(hArr).sort((a,b) => a-b);
    const medW = sortedW[Math.floor(nSamp/2)] || 1;
    const medH = sortedH[Math.floor(nSamp/2)] || 1;
    // Gap: fraction of median size added between samples (equal X and Y)
    const gapMult = 1 + gap * 0.25;
    const cellW = medW * gapMult;
    const cellH = medH * gapMult;
    
    // Group assignment
    const groupMeta = (groupCol && sampleMetaCache[groupCol]) ? sampleMetaCache[groupCol] : null;
    const groupCodes = groupMeta ? groupMeta.codes : null;
    const groupCats = groupMeta ? groupMeta.cats : ["all"];
    const nGroups = groupCats.length;
    
    const groups = new Array(nGroups);
    for (let g = 0; g < nGroups; g++) groups[g] = [];
    for (let si = 0; si < nSamp; si++) {{
      const gi = groupCodes ? groupCodes[si] : 0;
      groups[gi].push(si);
    }}
    
    // Column/Row metadata
    const colsMeta = (colsCol && sampleMetaCache[colsCol]) ? sampleMetaCache[colsCol] : null;
    const rowsMeta = (rowsCol && sampleMetaCache[rowsCol]) ? sampleMetaCache[rowsCol] : null;
    
    const targetX = new Float32Array(nSamp);
    const targetY = new Float32Array(nSamp);
    layoutSampleLabels = [];
    layoutLabelPositions = [];
    layoutGroupLabels = [];
    layoutColLabels = [];
    layoutRowLabels = [];
    layoutAxisInfo = {{ columns: colsCol || "(auto)", rows: rowsCol || "(name)" }};
    
    // Auto-square: find most square grid for n items
    function autoSquare(n) {{
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      return [cols, rows];
    }}
    
    // Group gap: slightly larger than cell gap for visual separation
    const groupGapX = cellW * 1.2;
    const groupGapY = cellH * 1.2;
    
    // Track max block size for group positioning
    let maxBlockW = 0, maxBlockH = 0;
    
    // Pre-compute each group's grid
    const groupGrids = [];
    
    for (let gi = 0; gi < nGroups; gi++) {{
      const members = groups[gi];
      
      if (colsMeta) {{
        // 2D GRID: columns = unique colsMeta values, rows = rowsMeta order
        const colMap = new Map();
        for (const si of members) {{
          const colKey = colsMeta.codes[si];
          if (!colMap.has(colKey)) colMap.set(colKey, []);
          colMap.get(colKey).push(si);
        }}
        
        const colKeys = Array.from(colMap.keys()).sort((a, b) => a - b);
        
        // Sort rows within each column
        for (const key of colKeys) {{
          colMap.get(key).sort((a, b) => {{
            if (rowsMeta) {{
              const ra = rowsMeta.codes[a], rb = rowsMeta.codes[b];
              if (ra !== rb) return ra - rb;
            }}
            return names[a] < names[b] ? -1 : names[a] > names[b] ? 1 : 0;
          }});
        }}
        
        let maxRows = 0;
        for (const key of colKeys) {{
          if (colMap.get(key).length > maxRows) maxRows = colMap.get(key).length;
        }}
        
        // With transpose: swap cols/rows
        const nC = transpose ? maxRows : colKeys.length;
        const nR = transpose ? colKeys.length : maxRows;
        
        groupGrids.push({{ type: "2d", colKeys, colMap, nC, nR, transpose: transpose }});
        if (nC * cellW > maxBlockW) maxBlockW = nC * cellW;
        if (nR * cellH > maxBlockH) maxBlockH = nR * cellH;
      }} else {{
        // AUTO-SQUARE GRID
        const [autoCols, autoRows] = autoSquare(members.length);
        const nC = transpose ? autoRows : autoCols;
        const nR = transpose ? autoCols : autoRows;
        
        // Sort by rows metadata or name
        members.sort((a, b) => {{
          if (rowsMeta) {{
            const ra = rowsMeta.codes[a], rb = rowsMeta.codes[b];
            if (ra !== rb) return ra - rb;
          }}
          return names[a] < names[b] ? -1 : names[a] > names[b] ? 1 : 0;
        }});
        
        groupGrids.push({{ type: "flat", members: [...members], nC, nR }});
        if (nC * cellW > maxBlockW) maxBlockW = nC * cellW;
        if (nR * cellH > maxBlockH) maxBlockH = nR * cellH;
      }}
    }}
    
    // Place groups in grid
    for (let gi = 0; gi < nGroups; gi++) {{
      const gRow = Math.floor(gi / groupNCols);
      const gCol = gi % groupNCols;
      const gx0 = gCol * (maxBlockW + groupGapX);
      const gy0 = gRow * (maxBlockH + groupGapY);
      
      // Store group label position (centered above the group)
      layoutGroupLabels.push({{
        name: groupCats[gi],
        x: gx0 + maxBlockW / 2,
        y: gy0 - cellH * 0.8  // Further above the group
      }});
      
      const grid = groupGrids[gi];
      
      if (grid.type === "2d") {{
        const {{ colKeys, colMap }} = grid;
        const isT = grid.transpose;
        
        // Get max rows across all columns for proper positioning
        let maxRowsInGroup = 0;
        for (const key of colKeys) {{
          if (colMap.get(key).length > maxRowsInGroup) maxRowsInGroup = colMap.get(key).length;
        }}
        
        // Collect unique row values if rowsMeta exists
        let sortedRowKeys = [];
        if (rowsMeta) {{
          const rowKeysSet = new Set();
          for (const key of colKeys) {{
            colMap.get(key).forEach(si => rowKeysSet.add(rowsMeta.codes[si]));
          }}
          sortedRowKeys = Array.from(rowKeysSet).sort((a, b) => a - b);
          console.log("[Layout] rowsMeta found, sortedRowKeys:", sortedRowKeys, "cats:", rowsMeta.cats);
        }} else {{
          console.log("[Layout] rowsMeta is null, rowsCol:", rowsCol, "cache keys:", Object.keys(sampleMetaCache));
        }}
        
        // Add axis labels based on transpose state
        if (isT) {{
          // TRANSPOSED: what was columns (colsMeta) is now on left as rows
          //             what was rows (rowsMeta) is now on top as columns
          
          // Row headers (left side) - use colsMeta values
          if (colsMeta) {{
            colKeys.forEach((ck, ri) => {{
              const rowName = colsMeta.cats[ck] || ("" + ck);
              layoutRowLabels.push({{ 
                name: rowName, 
                x: gx0 - cellW * 0.2, 
                y: gy0 + ri * cellH + cellH / 2 
              }});
            }});
          }}
          
          // Column headers (top) - use rowsMeta values
          if (rowsMeta && sortedRowKeys.length > 0) {{
            sortedRowKeys.forEach((rk, ci) => {{
              const colName = rowsMeta.cats[rk] || ("" + rk);
              layoutColLabels.push({{ 
                name: colName, 
                x: gx0 + ci * cellW + cellW / 2, 
                y: gy0 - cellH * 0.2
              }});
            }});
          }}
        }} else {{
          // NORMAL: colsMeta on top as columns, rowsMeta on left as rows
          
          // Column headers (top) - use colsMeta values  
          if (colsMeta) {{
            colKeys.forEach((ck, ci) => {{
              const colName = colsMeta.cats[ck] || ("" + ck);
              layoutColLabels.push({{ 
                name: colName, 
                x: gx0 + ci * cellW + cellW / 2, 
                y: gy0 - cellH * 0.2
              }});
            }});
          }}
          
          // Row headers (left side) - use rowsMeta values
          if (rowsMeta && sortedRowKeys.length > 0) {{
            sortedRowKeys.forEach((rk, ri) => {{
              const rowName = rowsMeta.cats[rk] || ("" + rk);
              layoutRowLabels.push({{ 
                name: rowName, 
                x: gx0 - cellW * 0.2, 
                y: gy0 + ri * cellH + cellH / 2 
              }});
            }});
          }}
        }}
        
        for (let ci = 0; ci < colKeys.length; ci++) {{
          const colSamples = colMap.get(colKeys[ci]);
          for (let ri = 0; ri < colSamples.length; ri++) {{
            const si = colSamples[ri];
            const gc = isT ? ri : ci;
            const gr = isT ? ci : ri;
            targetX[si] = gx0 + gc * cellW + cellW / 2;
            targetY[si] = gy0 + gr * cellH + cellH / 2;
            layoutSampleLabels.push(names[si]);
            layoutLabelPositions.push([targetX[si], targetY[si]]);
          }}
        }}
      }} else {{
        const {{ members, nC }} = grid;
        for (let mi = 0; mi < members.length; mi++) {{
          const si = members[mi];
          const col = mi % nC;
          const row = Math.floor(mi / nC);
          targetX[si] = gx0 + col * cellW + cellW / 2;
          targetY[si] = gy0 + row * cellH + cellH / 2;
          layoutSampleLabels.push(names[si]);
          layoutLabelPositions.push([targetX[si], targetY[si]]);
        }}
      }}
    }}
    
    // Apply offsets to all loaded cells
    for (let i = 0; i < loadedCount; i++) {{
      const sid = cellSampleId[i];
      posLayout[i * 2] = posSpatial[i * 2] + (targetX[sid] - cxArr[sid]);
      posLayout[i * 2 + 1] = posSpatial[i * 2 + 1] + (targetY[sid] - cyArr[sid]);
    }}
    
    // Bounds
    let minX = Infinity, maxX2 = -Infinity, minY = Infinity, maxY2 = -Infinity;
    for (let i = 0; i < loadedCount; i++) {{
      const x = posLayout[i * 2], y = posLayout[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX2) maxX2 = x;
      if (y < minY) minY = y;
      if (y > maxY2) maxY2 = y;
    }}
    METADATA["layout"] = {{ minX, maxX: maxX2, minY, maxY: maxY2, count: loadedCount }};
    
    layoutHasData = true;
    layoutParams = params;
    console.log(`[Layout] ${{nSamp}} samples, ${{nGroups}} groups (${{groupCats.join(",")}}) → instant`);
    return true;
  }}

  // Track which embedding the GPU buffers were built for
  let gpuEmbedding = null;
  let gpuLoadedCount = 0;  // Track how many points are in GPU buffers

  // Build GPU buffers from struct-of-arrays for current embedding
  // FAST: No object iteration, just array slicing and buffer upload
  function updateGPUBuffers() {{
    // Rebuild if data changed OR if embedding changed
    if (!gpuDataDirty && gpuEmbedding === currentEmbedding && gpuLoadedCount === loadedCount) return;
    
    if (loadedCount === 0) {{
      gpuPointCount = 0;
      gpuDataDirty = false;
      gpuEmbedding = currentEmbedding;
      gpuLoadedCount = 0;
      return;
    }}
    
    // Select which position array to use based on current embedding
    let posArray;
    if (currentEmbedding === "spatial") {{
      posArray = posSpatial;
    }} else if (currentEmbedding === "umap") {{
      posArray = posUmap;
    }} else if (currentEmbedding === "pca") {{
      posArray = posPca;
    }} else if (currentEmbedding === "layout") {{
      posArray = posLayout;
    }} else {{
      posArray = posSpatial;
    }}
    
    // FAST: Just slice the preallocated arrays (no iteration needed for positions!)
    const positions = posArray.subarray(0, loadedCount * 2);
    
    // Get plot state for toggles, colormap, opacity
    const _ps2 = window["_plotState_" + iframeId];
    const enabledArr = _ps2 && _ps2.obs && _ps2.obs.enabled;
    const activeCmap = (_ps2 && _ps2.gex && _ps2.gex.colormap) || currentColormap;
    const obsAlpha = (_ps2 && _ps2.obs && _ps2.obs.opacity != null) ? _ps2.obs.opacity : 1.0;
    
    // Build selection index set for fast lookup
    const selIndices = _ps2 && _ps2.selectionIndices;
    const selectionSet = selIndices && selIndices.length > 0 ? new Set(selIndices) : null;
    const hasActiveSelection = selectionSet !== null;
    
    // LAYERED: GEX base + obs overlay
    const colors = new Float32Array(loadedCount * 3);
    for (let i = 0; i < loadedCount; i++) {{
      let r = 0.6, g = 0.6, b = 0.6;
      const obsVal = obsValues[i];
      const gexVal = gexValues[i];
      
      // Layer 1: GEX colormap — when gene is active, ALL cells get colored
      // gexVal 0 = low end of colormap (dark), 1-255 = expression range
      if (currentGexGene) {{
        const t = clamp01(gexVal / 255.0);
        const c = cmapRGB(t, activeCmap);
        r = c[0]; g = c[1]; b = c[2];
      }}
      
      // Layer 2: obs overlay (alpha-blend on top)
      if (obsVal > 0 && currentObsColumn) {{
        const ci = obsVal - 1;
        if (enabledArr && enabledArr[ci] === false) {{
          r *= 0.15; g *= 0.15; b *= 0.15;
        }} else {{
          let or2, og, ob;
          if (currentPalette && currentPalette[ci]) {{
            const c = parseColor(currentPalette[ci]); or2=c[0]; og=c[1]; ob=c[2];
          }} else {{
            const c = fallbackCatColorRGB(ci); or2=c[0]; og=c[1]; ob=c[2];
          }}
          r = r*(1-obsAlpha) + or2*obsAlpha;
          g = g*(1-obsAlpha) + og*obsAlpha;
          b = b*(1-obsAlpha) + ob*obsAlpha;
        }}
      }}
      
      // Layer 3: Selection highlighting - dim unselected cells to 20%
      if (hasActiveSelection && !selectionSet.has(i)) {{
        r *= 0.2;
        g *= 0.2;
        b *= 0.2;
      }}
      
      colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
    }}
    
    // Upload to GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    
    gpuPointCount = loadedCount;
    gpuDataDirty = false;
    gpuEmbedding = currentEmbedding;
    gpuLoadedCount = loadedCount;
    
    console.log(`[WebGL] Uploaded ${{loadedCount.toLocaleString()}} points for ${{currentEmbedding}}`);
  }}

  // ----------------------------
  // EMBEDDING ANIMATION FUNCTIONS (need WebGL context)
  // ----------------------------
  
  // Animate embedding transition (called via requestAnimationFrame)
  function animateEmbeddingTransition(timestamp) {{
    if (!isAnimating) return;
    
    const elapsed = timestamp - animationStartTime;
    const t = Math.min(1.0, elapsed / animationDuration);
    const easedT = easeInOutQuad(t);
    
    // Get source and target position arrays
    const srcPositions = getEmbeddingPositions(animationSourceEmbedding);
    const tgtPositions = getEmbeddingPositions(animationTargetEmbedding);
    
    // Create interpolated positions
    const interpolated = new Float32Array(loadedCount * 2);
    for (let i = 0; i < loadedCount * 2; i++) {{
      interpolated[i] = lerp(srcPositions[i], tgtPositions[i], easedT);
    }}
    
    // Upload interpolated positions to GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, interpolated, gl.DYNAMIC_DRAW);
    
    // Draw the frame
    drawAnimationFrame(easedT);
    
    if (t < 1.0) {{
      // Continue animation
      requestAnimationFrame(animateEmbeddingTransition);
    }} else {{
      // Animation complete
      isAnimating = false;
      currentEmbedding = animationTargetEmbedding;
      animationSourceEmbedding = null;
      animationTargetEmbedding = null;
      markGPUDirty();  // Force rebuild with final positions
      draw();
      console.log(`[Embedding] Transition complete: now on ${{currentEmbedding}}`);
    }}
  }}
  
  // Draw a single animation frame with interpolated bounds
  function drawAnimationFrame(t) {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    if (loadedCount === 0) return;
    
    // Interpolate bounds between source and target embeddings
    const srcMeta = METADATA[animationSourceEmbedding];
    const tgtMeta = METADATA[animationTargetEmbedding];
    if (!srcMeta || !tgtMeta) return;
    
    const minX = lerp(srcMeta.minX, tgtMeta.minX, t);
    const maxX = lerp(srcMeta.maxX, tgtMeta.maxX, t);
    const minY = lerp(srcMeta.minY, tgtMeta.minY, t);
    const maxY = lerp(srcMeta.maxY, tgtMeta.maxY, t);
    
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const pad = 12;
    
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    const scale = baseScale * zoom;
    const offX = (W - spanX * scale) / 2 + panX;
    const offY = (H - spanY * scale) / 2 + panY;
    
    // Build transformation matrix
    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvas.width / dpr;
    const canvasH = canvas.height / dpr;
    
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    
    const a = scale * cos;
    const b = scale * sin;
    const c = -scale * sin;
    const d = scale * cos;
    
    const tx_pre = -minX * scale + offX;
    const ty_pre = -minY * scale + offY;
    
    const tx = cos * (tx_pre - cx) - sin * (ty_pre - cy) + cx;
    const ty = sin * (tx_pre - cx) + cos * (ty_pre - cy) + cy;
    
    const sx = 2.0 / canvasW;
    const sy = -2.0 / canvasH;
    const ox = -1.0;
    const oy = 1.0;
    
    const m00 = a * sx;
    const m01 = c * sx;
    const m02 = tx * sx + ox;
    const m10 = b * sy;
    const m11 = d * sy;
    const m12 = ty * sy + oy;
    
    const matrix = new Float32Array([
      m00, m10, 0,
      m01, m11, 0,
      m02, m12, 1
    ]);
    
    gl.useProgram(program);
    gl.uniformMatrix3fv(u_matrix, false, matrix);
    
    const baseSize = Math.max(1.5, Math.min(6, 3000 / Math.sqrt(loadedCount)));
    const zoomAdjustedSize = baseSize * Math.pow(zoom, 0.5);
    const finalSize = Math.max(1.0, Math.min(zoomAdjustedSize, 15)) * dpr;
    gl.uniform1f(u_pointSize, finalSize);
    const _ps = window["_plotState_" + iframeId];
    const _gexOp = (_ps && _ps.gex && _ps.gex.opacity != null) ? _ps.gex.opacity : 0.85;
    gl.uniform1f(u_opacity, currentGexGene ? _gexOp : 0.85);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    gl.drawArrays(gl.POINTS, 0, loadedCount);
    
    drawMinimap();
  }}

  function draw() {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    // Clear with transparent background
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Check if we have metadata
    const embedMeta = METADATA[currentEmbedding];
    if (!embedMeta || !METADATA.streaming) {{
      // Can't show text with WebGL easily, just leave blank
      drawMinimap();
      return;
    }}

    const {{minX, maxX, minY, maxY}} = embedMeta;
    const pad = 12;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    const scale = baseScale * zoom;
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (W - usedW) / 2 + panX;
    const offY = (H - usedH) / 2 + panY;
    
    // Check if we have cells loaded (chunk 0 should load instantly)
    if (loadedCount === 0) {{
      drawMinimap();
      return;
    }}
    
    // Update GPU buffers if needed (handles embedding switch too)
    updateGPUBuffers();
    
    if (gpuPointCount === 0) {{
      drawMinimap();
      return;
    }}
    
    // Build transformation matrix (data coords -> clip coords)
    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvas.width / dpr;
    const canvasH = canvas.height / dpr;
    
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    
    // Center of canvas in pixels
    const cx = canvasW / 2;
    const cy = canvasH / 2;
    
    // Transform: data -> pixels -> rotated pixels -> clip space
    // For a point (x, y) in data coords:
    // px = offX + (x - minX) * scale
    // py = offY + (y - minY) * scale
    // Then rotate around center, then convert to clip space
    
    // We'll build this as a 3x3 matrix for 2D homogeneous coordinates
    // Final transform: clip = M * dataPoint
    
    // Step 1: data to pixels (before rotation)
    // px = scale * x - scale * minX + offX
    // py = scale * y - scale * minY + offY
    
    // Step 2: rotate around (cx, cy)
    // px' = cos*(px-cx) - sin*(py-cy) + cx
    // py' = sin*(px-cx) + cos*(py-cy) + cy
    
    // Step 3: pixels to clip space
    // clipX = (px' / canvasW) * 2 - 1
    // clipY = 1 - (py' / canvasH) * 2  (flip Y)
    
    // Combining all these into one matrix...
    const sx = scale;
    const sy = scale;
    const tx = -minX * scale + offX;
    const ty = -minY * scale + offY;
    
    // After data->pixel transform, we have:
    // px = sx * x + tx
    // py = sy * y + ty
    
    // Rotate around (cx, cy):
    // px' = cos*(px - cx) - sin*(py - cy) + cx
    // py' = sin*(px - cx) + cos*(py - cy) + cy
    //
    // px' = cos*px - cos*cx - sin*py + sin*cy + cx
    // py' = sin*px - sin*cx + cos*py - cos*cy + cy
    //
    // Substituting px, py:
    // px' = cos*(sx*x + tx) - sin*(sy*y + ty) + cx*(1-cos) + cy*sin
    // py' = sin*(sx*x + tx) + cos*(sy*y + ty) - cx*sin + cy*(1-cos)
    //
    // px' = cos*sx*x - sin*sy*y + (cos*tx - sin*ty + cx*(1-cos) + cy*sin)
    // py' = sin*sx*x + cos*sy*y + (sin*tx + cos*ty - cx*sin + cy*(1-cos))
    
    const rotTx = cos*tx - sin*ty + cx*(1-cos) + cy*sin;
    const rotTy = sin*tx + cos*ty - cx*sin + cy*(1-cos);
    
    // Now convert to clip space:
    // clipX = px' * (2/canvasW) - 1
    // clipY = 1 - py' * (2/canvasH) = -py' * (2/canvasH) + 1
    
    // Final matrix (row-major for WebGL uniform):
    // | a  b  c |   | x |   | clipX |
    // | d  e  f | * | y | = | clipY |
    // | 0  0  1 |   | 1 |   |   1   |
    
    const toClipX = 2 / canvasW;
    const toClipY = -2 / canvasH;
    
    const a = cos * sx * toClipX;
    const b = -sin * sy * toClipX;
    const c = rotTx * toClipX - 1;
    
    const d = sin * sx * toClipY;
    const e = cos * sy * toClipY;
    const f = rotTy * toClipY + 1;
    
    // WebGL uses column-major, so transpose
    const matrix = new Float32Array([
      a, d, 0,
      b, e, 0,
      c, f, 1
    ]);
    
    // Set up WebGL state
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Set uniforms
    gl.uniformMatrix3fv(u_matrix, false, matrix);
    
    const state = window["_plotState_" + iframeId] || {{}};
    const pointSize = (state.pointSize || 1.1) * (window.devicePixelRatio || 1) * 2;
    gl.uniform1f(u_pointSize, pointSize);
    gl.uniform1f(u_opacity, 0.85);
    
    // Bind position buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
    
    // Bind color buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    
    // Draw!
    gl.drawArrays(gl.POINTS, 0, gpuPointCount);
    
    drawSampleLabels();
    drawMinimap();
    
    // Draw selection outline on top (if any)
    drawSelectionHandles();
  }}
  
  // Helper: programmatically trigger a button click with data (from parent context)
  function sendButtonClick(buttonId, data) {{
    const requestData = {{
      ...data,
      activeColumn: currentObsColumn,
      activeGene: currentGexGene
    }};
    
    console.log(`[Parent] Sending button click: ${{buttonId}}`, requestData);
    
    window["_requests_" + iframeId].push({{
      type: 'button_click',
      iframeId: iframeId,
      buttonId: buttonId,
      data: requestData,
      timestamp: Date.now()
    }});
  }}
  
  // ----------------------------
  // Minimap drawing
  // ----------------------------
  function drawMinimap() {{
    const mmW = minimap.width;
    const mmH = minimap.height;
    
    minimapCtx.fillStyle = "#000";
    minimapCtx.fillRect(0, 0, mmW, mmH);
    
    // Get bounds from metadata
    const embedMeta = METADATA[currentEmbedding];
    if (!embedMeta) return;
    
    const minX = embedMeta.minX;
    const maxX = embedMeta.maxX;
    const minY = embedMeta.minY;
    const maxY = embedMeta.maxY;
    
    const pad = 5;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const scale = Math.min((mmW - 2*pad) / spanX, (mmH - 2*pad) / spanY);
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (mmW - usedW) / 2;
    const offY = (mmH - usedH) / 2;
    
    // Draw minimap sample points as small dots
    minimapCtx.fillStyle = "rgba(200, 200, 200, 0.8)";
    
    // For layout mode, draw from actual layout positions (sample centroids)
    if (currentEmbedding === "layout" && layoutHasData && layoutLabelPositions.length > 0) {{
      for (let i = 0; i < layoutLabelPositions.length; i++) {{
        const [x, y] = layoutLabelPositions[i];
        const px = offX + (x - minX) * scale;
        const py = offY + (y - minY) * scale;
        minimapCtx.beginPath();
        minimapCtx.arc(px, py, 2, 0, Math.PI * 2);
        minimapCtx.fill();
      }}
    }} else if (MINIMAP_POINTS && MINIMAP_POINTS.length > 0) {{
      // For other embeddings, use the precomputed minimap sample
      for (let i = 0; i < MINIMAP_POINTS.length; i++) {{
        const [x, y] = MINIMAP_POINTS[i];
        const px = offX + (x - minX) * scale;
        const py = offY + (y - minY) * scale;
        minimapCtx.fillRect(px - 0.5, py - 0.5, 1, 1);
      }}
    }}
    
    // Draw viewport indicator on minimap
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    const baseScale = Math.min((W - 24) / spanX, (H - 24) / spanY);
    const viewScale = baseScale * zoom;
    const viewW = spanX * viewScale;
    const viewH = spanY * viewScale;
    const viewOffX = (W - viewW) / 2 + panX;
    const viewOffY = (H - viewH) / 2 + panY;
    
    // Calculate viewport bounds in data space (unrotated)
    const dataMinX = minX - viewOffX / viewScale;
    const dataMaxX = minX + (W - viewOffX) / viewScale;
    const dataMinY = minY - viewOffY / viewScale;
    const dataMaxY = minY + (H - viewOffY) / viewScale;
    
    // Calculate 4 corners of viewport
    const dataCenterX = (dataMinX + dataMaxX) / 2;
    const dataCenterY = (dataMinY + dataMaxY) / 2;
    
    const corners = [
      [dataMinX, dataMinY],  // top-left
      [dataMaxX, dataMinY],  // top-right
      [dataMaxX, dataMaxY],  // bottom-right
      [dataMinX, dataMaxY]   // bottom-left
    ];
    
    // If rotation is active, rotate corners
    if (rotation !== 0) {{
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      
      for (let i = 0; i < corners.length; i++) {{
        const dx = corners[i][0] - dataCenterX;
        const dy = corners[i][1] - dataCenterY;
        corners[i][0] = dataCenterX + (dx * cos - dy * sin);
        corners[i][1] = dataCenterY + (dx * sin + dy * cos);
      }}
    }}
    
    // Convert corners to minimap pixels
    const mmCorners = corners.map(([x, y]) => [
      offX + (x - minX) * scale,
      offY + (y - minY) * scale
    ]);
    
    // Draw the viewport polygon (shows rotation!)
    minimapCtx.strokeStyle = "rgba(255, 50, 50, 0.9)";  // Bright red
    minimapCtx.lineWidth = 1.5;
    minimapCtx.beginPath();
    minimapCtx.moveTo(mmCorners[0][0], mmCorners[0][1]);
    for (let i = 1; i < mmCorners.length; i++) {{
      minimapCtx.lineTo(mmCorners[i][0], mmCorners[i][1]);
    }}
    minimapCtx.closePath();
    minimapCtx.stroke();
  }}

  // ----------------------------
  // Initialize: Load chunk 0 and start rendering
  // Wait for pako to load if using compression
  // ----------------------------
  function startLoading() {{
    loadChunk0();
    markGPUDirty();
    resizeCanvas();
  }}
  
  if (USE_COMPRESSION && typeof pako === 'undefined') {{
    // Wait for pako to load (check every 50ms, max 2 seconds)
    let pakoWaitCount = 0;
    const pakoWait = setInterval(() => {{
      pakoWaitCount++;
      if (typeof pako !== 'undefined') {{
        clearInterval(pakoWait);
        startLoading();
      }} else if (pakoWaitCount > 40) {{
        clearInterval(pakoWait);
        console.error("[Compression] pako failed to load, trying without compression");
        // Try anyway - decodeBinaryCoords will handle uncompressed fallback
        startLoading();
      }}
    }}, 50);
  }} else {{
    startLoading();
  }}
  
  window.addEventListener("resize", resizeCanvas);

  // ----------------------------
  // Pan/Zoom/Rotation controls
  // ----------------------------
  
  // Wheel zoom (WebGL handles all points efficiently - no viewport loading needed!)
  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const delta = e.deltaY * 0.0005;
    const zoomFactor = delta > 0 ? 0.975 : 1.025;
    const oldZoom = zoom;
    zoom *= zoomFactor;
    zoom = Math.max(0.01, zoom);  // Only minimum zoom, no maximum! Zoom in forever!
    
    const zoomChange = zoom / oldZoom;
    panX = mouseX - (mouseX - panX) * zoomChange;
    panY = mouseY - (mouseY - panY) * zoomChange;
    
    draw();  // WebGL renders instantly regardless of point count
  }});
  
  // Double-click to reset view (just reset pan/zoom/rotation - keep all loaded cells!)
  canvas.addEventListener("dblclick", () => {{
    zoom = 1.0;
    lastZoomLevel = 1.0;
    panX = 0;
    panY = 0;
    rotation = 0;
    rotationMode = false;
    draw();
  }});
  
  // Auto-focus panel on mouse enter to capture keyboard events
  panel.addEventListener("mouseenter", () => {{
    panel.focus();
  }});
  
  // Keyboard controls - only work when panel has focus
  panel.addEventListener("keydown", (e) => {{
    // Arrow keys for size (when not in input)
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {{
      e.preventDefault();
      e.stopPropagation();
      const state = window["_plotState_" + iframeId];
      if (!state) return;
      
      const step = 0.1;
      const min = 0.5;
      const max = 8;
      
      if (e.key === "ArrowLeft") {{
        state.pointSize = Math.max(min, state.pointSize - step);
      }} else {{
        state.pointSize = Math.min(max, state.pointSize + step);
      }}
      
      draw();
    }}
    
    // R key for rotation
    if (e.key === "r" || e.key === "R") {{
      e.preventDefault();
      e.stopPropagation();
      const now = Date.now();
      if (now - lastRotationKeyPress < 500) {{
        // Double-tap: reset rotation
        rotation = 0;
        rotationMode = false;
      }} else {{
        // Single tap: toggle rotation mode
        rotationMode = !rotationMode;
        canvas.style.cursor = rotationMode ? "crosshair" : "default";
      }}
      lastRotationKeyPress = now;
      draw();
    }}
  }});
  
  // Rotation with mouse when in rotation mode
  canvas.addEventListener("mousemove", (e) => {{
    if (rotationMode && !isDragging && !window["_isDrawing_" + iframeId]) {{
      const rect = canvas.getBoundingClientRect();
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const angle = Math.atan2(e.clientY - rect.top - centerY, e.clientX - rect.left - centerX);
      rotation = angle;
      draw();
      return;
    }}
  }});

  // ----------------------------
  // Selection drawing AND panning
  // ----------------------------
  // Selection state
  window["_selectionMode_" + iframeId] = null; // "drawing", "moving", "resizing"
  let selectionMode = null; // Local reference for convenience
  let selectionHandle = null; // which handle is being dragged for resize
  
  // Helper: check if point is near another point
  function nearPoint(x, y, px, py, threshold = 8) {{
    return Math.abs(x - px) < threshold && Math.abs(y - py) < threshold;
  }}
  
  // Helper: check if point is inside selection for moving
  function isInsideSelection(x, y) {{
    const path = window["_selectionPath_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    if (!path || path.length < 2) return false;
    
    if (tool === "rectangle") {{
      const [x1, y1] = path[0];
      const [x2, y2] = path[1];
      const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
      const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
      return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }} else if (tool === "circle") {{
      const [cx, cy] = path[0];
      const [ex, ey] = path[1];
      const rx = Math.abs(ex - cx);
      const ry = path.length > 2 ? Math.abs(path[2][1] - cy) : rx;
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      return (dx * dx + dy * dy) <= 1;
    }} else if (tool === "lasso" || tool === "polygon") {{
      // Point in polygon test
      let inside = false;
      for (let i = 0, j = path.length - 1; i < path.length; j = i++) {{
        const [xi, yi] = path[i];
        const [xj, yj] = path[j];
        if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {{
          inside = !inside;
        }}
      }}
      return inside;
    }}
    return false;
  }}
  
  canvas.addEventListener("mousedown", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Exit rotation mode on click (locks rotation in place)
    if (rotationMode) {{
      rotationMode = false;
      canvas.style.cursor = "default";
      return;
    }}
    
    // Check for transform handles on ANY existing selection (even without tool selected)
    const existingPath = window["_selectionPath_" + iframeId];
    const isDrawing = window["_isDrawing_" + iframeId];
    
    if (!isDrawing && existingPath && existingPath.length >= 2) {{
      const handleHit = checkTransformHandleHit(x, y);
      if (handleHit) {{
        e.preventDefault();
        return;  // Handle interaction started
      }}
      
      // Check if clicking inside existing selection to move it
      if (isInsideSelection(x, y)) {{
        selectionMode = "moving";
        window["_selectionMode_" + iframeId] = "moving";
        window["_moveStart_" + iframeId] = [x, y];
        window["_justTransformed_" + iframeId] = true;  // Prevent click from firing
        canvas.style.cursor = "move";
        return;
      }}
    }}
    
    // Polygon uses click events ONLY for drawing - don't interfere with mousedown
    if (tool === "polygon") {{
      return;
    }}
    
    if (tool) {{
      // Start new selection (this clears the existing one)
      selectionMode = "drawing";
      window["_selectionMode_" + iframeId] = "drawing";
      window["_isDrawing_" + iframeId] = true;
      window["_selectionPathData_" + iframeId] = null;  // Clear data path
      
      // Tell iframe to deactivate current selection so we create a new one
      iframe.contentWindow.postMessage({{
        type: "clear_active_selection"
      }}, "*");
      
      if (tool === "circle") {{
        // Circle: first point is center
        window["_selectionPath_" + iframeId] = [[x, y], [x, y]];
        window["_selectionStart_" + iframeId] = [x, y];
      }} else if (tool === "rectangle") {{
        // Rectangle: first point is center
        window["_selectionPath_" + iframeId] = [[x, y], [x, y]];
        window["_selectionStart_" + iframeId] = [x, y];
      }} else if (tool === "lasso") {{
        // Lasso: mousedown starts, mousemove adds points
        window["_selectionPath_" + iframeId] = [[x, y]];
      }}
    }} else if (!rotationMode) {{
      // Panning mode
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing";
    }}
  }});
  
  // Check if clicking on a transform handle
  function checkTransformHandleHit(x, y) {{
    const handles = window["_selectionHandles_" + iframeId];
    const rotHandle = window["_rotationHandle_" + iframeId];
    const bounds = window["_selectionBounds_" + iframeId];
    const path = window["_selectionPath_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    
    console.log("[HandleHit] Checking at", x, y, "bounds:", bounds, "handles:", handles ? handles.length : "null", "rotHandle:", rotHandle);
    
    if (!bounds) {{
      console.log("[HandleHit] No bounds - returning null");
      return null;
    }}
    
    // Check rotation handle first
    if (rotHandle) {{
      const dist = Math.sqrt((x - rotHandle.x) ** 2 + (y - rotHandle.y) ** 2);
      console.log("[HandleHit] Rot handle dist:", dist, "threshold:", rotHandle.r);
      if (dist <= rotHandle.r) {{
        selectionMode = "rotating";
        window["_selectionMode_" + iframeId] = "rotating";
        window["_transformStart_" + iframeId] = {{ x, y, bounds: {{...bounds}} }};
        window["_justTransformed_" + iframeId] = true;  // Prevent click from firing
        canvas.style.cursor = "grab";
        console.log("[HandleHit] HIT rotation handle!");
        return "rotate";
      }}
    }}
    
    // Check polygon vertex handles
    if (tool === "polygon" && path && path.length > 2) {{
      for (let i = 0; i < path.length; i++) {{
        const [px, py] = path[i];
        if (Math.abs(x - px) < 8 && Math.abs(y - py) < 8) {{
          selectionMode = "vertex";
          window["_selectionMode_" + iframeId] = "vertex";
          window["_dragVertex_" + iframeId] = i;
          window["_transformStart_" + iframeId] = {{ x, y }};
          window["_justTransformed_" + iframeId] = true;  // Prevent click from firing
          canvas.style.cursor = "move";
          return "vertex";
        }}
      }}
    }}
    
    // Check resize handles
    if (handles) {{
      for (let i = 0; i < handles.length; i++) {{
        const h = handles[i];
        if (Math.abs(x - h.x) < 6 && Math.abs(y - h.y) < 6) {{
          selectionMode = "resizing";
          window["_selectionMode_" + iframeId] = "resizing";
          window["_resizeHandle_" + iframeId] = i;
          window["_transformStart_" + iframeId] = {{ x, y, bounds: {{...bounds}}, path: path.map(p => [...p]) }};
          window["_justTransformed_" + iframeId] = true;  // Prevent click from firing
          canvas.style.cursor = h.cursor;
          console.log("[HandleHit] HIT resize handle", i);
          return "resize";
        }}
      }}
    }}
    
    console.log("[HandleHit] No hit");
    return null;
  }}
  
  canvas.addEventListener("mousemove", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (selectionMode === "moving") {{
      // Move selection
      const [startX, startY] = window["_moveStart_" + iframeId];
      const dx = x - startX;
      const dy = y - startY;
      const path = window["_selectionPath_" + iframeId];
      
      if (path) {{
        console.log("[Transform] Moving by", dx, dy);
        window["_didDrag_" + iframeId] = true;  // Mark that actual dragging happened
        for (let i = 0; i < path.length; i++) {{
          path[i][0] += dx;
          path[i][1] += dy;
        }}
        window["_moveStart_" + iframeId] = [x, y];
        drawSelectionOutline();
      }}
    }} else if (selectionMode === "rotating") {{
      // Rotate selection around center
      const start = window["_transformStart_" + iframeId];
      const bounds = start.bounds;
      const path = window["_selectionPath_" + iframeId];
      
      // Calculate angle from center to current mouse
      const angle1 = Math.atan2(start.y - bounds.cy, start.x - bounds.cx);
      const angle2 = Math.atan2(y - bounds.cy, x - bounds.cx);
      const deltaAngle = angle2 - angle1;
      
      // Rotate all points around center
      const cos = Math.cos(deltaAngle);
      const sin = Math.sin(deltaAngle);
      
      if (path) {{
        console.log("[Transform] Rotating by", deltaAngle * 180 / Math.PI, "degrees");
        window["_didDrag_" + iframeId] = true;  // Mark that actual dragging happened
        for (let i = 0; i < path.length; i++) {{
          const px = path[i][0] - bounds.cx;
          const py = path[i][1] - bounds.cy;
          path[i][0] = bounds.cx + (px * cos - py * sin);
          path[i][1] = bounds.cy + (px * sin + py * cos);
        }}
        window["_transformStart_" + iframeId] = {{ x, y, bounds: {{...bounds}} }};
        drawSelectionOutline();
      }}
    }} else if (selectionMode === "resizing") {{
      // Resize selection from handle
      const handleIdx = window["_resizeHandle_" + iframeId];
      const start = window["_transformStart_" + iframeId];
      const path = window["_selectionPath_" + iframeId];
      const origBounds = start.bounds;
      const origPath = start.path;
      
      console.log("[Transform] Resizing handle", handleIdx, "path:", path ? path.length : "null", "origPath:", origPath ? origPath.length : "null");
      
      if (!path || !origBounds) return;
      
      window["_didDrag_" + iframeId] = true;  // Mark that actual dragging happened
      
      // Calculate scale factors based on which handle
      let scaleX = 1, scaleY = 1;
      let anchorX = origBounds.cx, anchorY = origBounds.cy;
      
      // Handle indices: 0=TL, 1=TR, 2=BR, 3=BL, 4=T, 5=R, 6=B, 7=L
      if (handleIdx === 0) {{ // top-left
        scaleX = (origBounds.maxX - x) / origBounds.w;
        scaleY = (origBounds.maxY - y) / origBounds.h;
        anchorX = origBounds.maxX; anchorY = origBounds.maxY;
      }} else if (handleIdx === 1) {{ // top-right
        scaleX = (x - origBounds.minX) / origBounds.w;
        scaleY = (origBounds.maxY - y) / origBounds.h;
        anchorX = origBounds.minX; anchorY = origBounds.maxY;
      }} else if (handleIdx === 2) {{ // bottom-right
        scaleX = (x - origBounds.minX) / origBounds.w;
        scaleY = (y - origBounds.minY) / origBounds.h;
        anchorX = origBounds.minX; anchorY = origBounds.minY;
      }} else if (handleIdx === 3) {{ // bottom-left
        scaleX = (origBounds.maxX - x) / origBounds.w;
        scaleY = (y - origBounds.minY) / origBounds.h;
        anchorX = origBounds.maxX; anchorY = origBounds.minY;
      }} else if (handleIdx === 4) {{ // top
        scaleY = (origBounds.maxY - y) / origBounds.h;
        anchorY = origBounds.maxY;
      }} else if (handleIdx === 5) {{ // right
        scaleX = (x - origBounds.minX) / origBounds.w;
        anchorX = origBounds.minX;
      }} else if (handleIdx === 6) {{ // bottom
        scaleY = (y - origBounds.minY) / origBounds.h;
        anchorY = origBounds.minY;
      }} else if (handleIdx === 7) {{ // left
        scaleX = (origBounds.maxX - x) / origBounds.w;
        anchorX = origBounds.maxX;
      }}
      
      // Ensure minimum scale
      scaleX = Math.max(0.1, scaleX);
      scaleY = Math.max(0.1, scaleY);
      
      // Apply scale to all points
      for (let i = 0; i < path.length; i++) {{
        path[i][0] = anchorX + (origPath[i][0] - anchorX) * scaleX;
        path[i][1] = anchorY + (origPath[i][1] - anchorY) * scaleY;
      }}
      
      drawSelectionOutline();
    }} else if (selectionMode === "vertex") {{
      // Move single polygon vertex
      const vertIdx = window["_dragVertex_" + iframeId];
      const path = window["_selectionPath_" + iframeId];
      
      if (path && vertIdx !== undefined && vertIdx < path.length) {{
        window["_didDrag_" + iframeId] = true;  // Mark that actual dragging happened
        path[vertIdx][0] = x;
        path[vertIdx][1] = y;
        drawSelectionOutline();
      }}
    }} else if (tool && window["_isDrawing_" + iframeId]) {{
      // Selection drawing
      if (tool === "lasso") {{
        window["_selectionPath_" + iframeId].push([x, y]);
      }} else if (tool === "polygon") {{
        // For polygon, just update preview cursor position (don't modify path)
        window["_polygonPreview_" + iframeId] = [x, y];
        drawSelectionOutline();
      }} else if (tool === "rectangle") {{
        // Rectangle expands from center
        const [cx, cy] = window["_selectionStart_" + iframeId];
        const halfW = Math.abs(x - cx);
        const halfH = Math.abs(y - cy);
        // Store as two corners
        window["_selectionPath_" + iframeId] = [
          [cx - halfW, cy - halfH],  // top-left
          [cx + halfW, cy + halfH]   // bottom-right
        ];
      }} else if (tool === "circle") {{
        // Circle/ellipse expands from center
        const [cx, cy] = window["_selectionStart_" + iframeId];
        // If shift is held, make it a circle; otherwise allow ellipse
        if (e.shiftKey) {{
          const radius = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
          window["_selectionPath_" + iframeId] = [[cx, cy], [cx + radius, cy]];
        }} else {{
          // Ellipse: store center, edge X, edge Y
          window["_selectionPath_" + iframeId] = [[cx, cy], [x, cy], [cx, y]];
        }}
      }}
      drawSelectionOutline();
    }} else if (isDragging) {{
      // Panning - rotation-aware
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      
      // Rotate the pan delta by the inverse of current rotation
      const cos = Math.cos(-rotation);
      const sin = Math.sin(-rotation);
      const rotatedDx = dx * cos - dy * sin;
      const rotatedDy = dx * sin + dy * cos;
      
      panX += rotatedDx;
      panY += rotatedDy;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      draw();
    }} else {{
      // Update cursor based on hover state (handles, inside, outside)
      // This works even without a tool selected for editing existing selections
      const isDrawing = window["_isDrawing_" + iframeId];
      const existingPath = window["_selectionPath_" + iframeId];
      
      if (!isDrawing && existingPath && existingPath.length >= 2) {{
        // Check handle hover for cursor
        const handles = window["_selectionHandles_" + iframeId];
        const rotHandle = window["_rotationHandle_" + iframeId];
        const selTool = window["_selectionTool_" + iframeId];
        
        // Rotation handle
        if (rotHandle) {{
          const dist = Math.sqrt((x - rotHandle.x) ** 2 + (y - rotHandle.y) ** 2);
          if (dist <= rotHandle.r) {{
            canvas.style.cursor = "grab";
            return;
          }}
        }}
        
        // Polygon vertices
        if (selTool === "polygon" && existingPath.length > 2) {{
          for (let i = 0; i < existingPath.length; i++) {{
            const [px, py] = existingPath[i];
            if (Math.abs(x - px) < 8 && Math.abs(y - py) < 8) {{
              canvas.style.cursor = "move";
              return;
            }}
          }}
        }}
        
        // Resize handles
        if (handles) {{
          for (const h of handles) {{
            if (Math.abs(x - h.x) < 6 && Math.abs(y - h.y) < 6) {{
              canvas.style.cursor = h.cursor;
              return;
            }}
          }}
        }}
        
        // Inside selection
        if (isInsideSelection(x, y)) {{
          canvas.style.cursor = "move";
          return;
        }}
      }}
      
      canvas.style.cursor = tool ? "crosshair" : "default";
    }}
  }});
  
  canvas.addEventListener("mouseup", () => {{
    const tool = window["_selectionTool_" + iframeId];
    
    console.log("[MouseUp] selectionMode:", selectionMode, "tool:", tool, "didDrag:", window["_didDrag_" + iframeId]);
    
    // Complete any transform operations - but only if we actually dragged
    if (selectionMode === "moving" || selectionMode === "rotating" || 
        selectionMode === "resizing" || selectionMode === "vertex") {{
      
      const didDrag = window["_didDrag_" + iframeId];
      console.log("[MouseUp] In transform mode, didDrag:", didDrag);
      
      if (didDrag) {{
        console.log("[MouseUp] Actually dragged - completing selection");
        // Update the data path from the transformed canvas path
        const canvasPath = window["_selectionPath_" + iframeId];
        if (canvasPath && canvasPath.length > 0) {{
          window["_selectionPathData_" + iframeId] = pathCanvasToData(canvasPath);
        }}
        
        // Recompute selection with new shape
        completeSelection();
      }} else {{
        console.log("[MouseUp] No drag - NOT completing");
      }}
      
      selectionMode = null;
      window["_selectionMode_" + iframeId] = null;
      window["_didDrag_" + iframeId] = false;
      canvas.style.cursor = tool ? "crosshair" : "default";
      return;
    }}
    
    // For polygon, don't do anything on mouseup - polygon uses click events only
    if (tool === "polygon") {{
      console.log("[MouseUp] Polygon tool - returning early");
      return;
    }}
    
    if (tool && window["_isDrawing_" + iframeId]) {{
      console.log("[MouseUp] Was drawing - completing");
      window["_isDrawing_" + iframeId] = false;
      selectionMode = null;
      window["_selectionMode_" + iframeId] = null;
      completeSelection();
    }}
    
    // End panning (no viewport refresh needed - all data is loaded!)
    if (isDragging) {{
      isDragging = false;
      canvas.style.cursor = "default";
    }}
  }});
  
  canvas.addEventListener("mouseleave", () => {{
    if (isDragging) {{
      isDragging = false;
      canvas.style.cursor = "default";
    }}
  }});
  
  // Polygon: click adds point, click near first point closes
  canvas.addEventListener("click", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    if (tool !== "polygon") return;
    
    // If we just finished a transform, don't start a new polygon
    const justTransformed = window["_justTransformed_" + iframeId];
    if (justTransformed) {{
      window["_justTransformed_" + iframeId] = false;
      return;
    }}
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const path = window["_selectionPath_" + iframeId];
    const isDrawing = window["_isDrawing_" + iframeId];
    
    console.log("[Polygon] Click - isDrawing:", isDrawing, "path:", path ? path.length : "null");
    
    // If we're currently drawing a polygon
    if (isDrawing && path && path.length > 0) {{
      // Check if clicking near first point to close polygon
      const [firstX, firstY] = path[0];
      const distToFirst = Math.sqrt((x - firstX) ** 2 + (y - firstY) ** 2);
      
      if (path.length >= 3 && distToFirst < 15) {{
        // Close polygon - clicked near first point
        console.log("[Polygon] Closing polygon");
        window["_isDrawing_" + iframeId] = false;
        window["_polygonPreview_" + iframeId] = null;
        completeSelection();
        return;
      }}
      
      // Add new vertex
      console.log("[Polygon] Adding vertex at", x, y);
      path.push([x, y]);
      drawSelectionOutline();
      return;
    }}
    
    // Not currently drawing - start new polygon ONLY if no active polygon
    if (!isDrawing) {{
      console.log("[Polygon] Starting new polygon at", x, y);
      window["_isDrawing_" + iframeId] = true;
      window["_selectionPath_" + iframeId] = [[x, y]];
      window["_selectionPathData_" + iframeId] = null;  // Clear data path
      window["_polygonPreview_" + iframeId] = [x, y];
      
      // Tell iframe to deactivate current selection so we create a new one
      iframe.contentWindow.postMessage({{
        type: "clear_active_selection"
      }}, "*");
      
      drawSelectionOutline();
    }}
  }});
  
  canvas.addEventListener("dblclick", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    if (tool !== "polygon") return;
    
    e.preventDefault();
    const pathCheck = window["_selectionPath_" + iframeId];
    if (pathCheck && pathCheck.length >= 3) {{
      window["_isDrawing_" + iframeId] = false;
      window["_polygonPreview_" + iframeId] = null;
      completeSelection();
    }}
  }});
  
  // Draw selection handles on label overlay (called from draw())
  function drawSelectionHandles() {{
    const dataPath = window["_selectionPathData_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    const isDrawing = window["_isDrawing_" + iframeId];
    
    // If we're actively drawing OR actively transforming, use the canvas path directly
    // Only convert from data coords when we're in a "stable" state (not manipulating)
    // Use window variable to check selection mode (avoids reference before declaration)
    const currentSelMode = window["_selectionMode_" + iframeId];
    const isTransforming = currentSelMode === "moving" || currentSelMode === "rotating" || 
                           currentSelMode === "resizing" || currentSelMode === "vertex";
    
    let path;
    if (isDrawing || isTransforming) {{
      // Use canvas path directly during active manipulation
      path = window["_selectionPath_" + iframeId];
    }} else if (dataPath && dataPath.length > 0) {{
      // Convert stored data coords to current canvas coords (stable state)
      path = pathDataToCanvas(dataPath);
      window["_selectionPath_" + iframeId] = path;  // Update canvas path for hit testing
    }} else {{
      path = window["_selectionPath_" + iframeId];
    }}
    
    if (!path || path.length === 0) return;
    if (!labelCtx || !labelOverlay) return;
    
    const dpr = window.devicePixelRatio || 1;
    
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    
    // Calculate bounding box for transform handles
    let bounds = null;
    if (!isDrawing && path.length >= 2) {{
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      if (tool === "circle") {{
        const [cx, cy] = path[0];
        const [ex, ey] = path[1];
        const rx = Math.abs(ex - cx);
        const ry = path.length > 2 ? Math.abs(path[2][1] - cy) : rx;
        minX = cx - rx; maxX = cx + rx;
        minY = cy - ry; maxY = cy + ry;
      }} else {{
        path.forEach(([px, py]) => {{
          if (px < minX) minX = px;
          if (px > maxX) maxX = px;
          if (py < minY) minY = py;
          if (py > maxY) maxY = py;
        }});
      }}
      bounds = {{ minX, maxX, minY, maxY, cx: (minX+maxX)/2, cy: (minY+maxY)/2, w: maxX-minX, h: maxY-minY }};
      window["_selectionBounds_" + iframeId] = bounds;
    }}
    
    // Draw the shape outline
    const drawOutline = (style, width, dash) => {{
      labelCtx.strokeStyle = style;
      labelCtx.lineWidth = width;
      labelCtx.setLineDash(dash);
      
      if (tool === "lasso" && path.length > 1) {{
        labelCtx.beginPath();
        labelCtx.moveTo(path[0][0], path[0][1]);
        for (let i = 1; i < path.length; i++) {{
          labelCtx.lineTo(path[i][0], path[i][1]);
        }}
        if (!isDrawing) labelCtx.closePath();
        labelCtx.stroke();
      }} else if (tool === "polygon" && path.length > 0) {{
        labelCtx.beginPath();
        labelCtx.moveTo(path[0][0], path[0][1]);
        for (let i = 1; i < path.length; i++) {{
          labelCtx.lineTo(path[i][0], path[i][1]);
        }}
        
        const preview = window["_polygonPreview_" + iframeId];
        if (isDrawing && preview) {{
          labelCtx.lineTo(preview[0], preview[1]);
        }}
        
        if (!isDrawing && path.length > 2) {{
          labelCtx.closePath();
        }}
        labelCtx.stroke();
        
        // Draw close target while drawing
        if (isDrawing) {{
          path.forEach(([px, py], idx) => {{
            labelCtx.beginPath();
            if (idx === 0 && path.length >= 3) {{
              labelCtx.fillStyle = "rgba(34, 197, 94, 0.9)";
              labelCtx.arc(px, py, 8, 0, Math.PI * 2);
            }} else {{
              labelCtx.fillStyle = "white";
              labelCtx.arc(px, py, 4, 0, Math.PI * 2);
            }}
            labelCtx.fill();
            labelCtx.strokeStyle = "rgba(0,0,0,0.5)";
            labelCtx.lineWidth = 1;
            labelCtx.setLineDash([]);
            labelCtx.stroke();
          }});
        }}
      }} else if (tool === "rectangle" && path.length >= 2) {{
        const [x1, y1] = path[0];
        const [x2, y2] = path[1];
        labelCtx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
      }} else if (tool === "circle" && path.length >= 2) {{
        const [cx, cy] = path[0];
        const [ex, ey] = path[1];
        const rx = Math.abs(ex - cx);
        const ry = path.length > 2 ? Math.abs(path[2][1] - cy) : rx;
        labelCtx.beginPath();
        labelCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        labelCtx.stroke();
      }}
    }};
    
    // Draw dark outline first, then white dashed
    drawOutline("rgba(0, 0, 0, 0.5)", 4, []);
    drawOutline("rgba(255, 255, 255, 0.9)", 2, [5, 5]);
    
    // Draw transform handles when selection is complete
    if (!isDrawing && bounds) {{
      labelCtx.setLineDash([]);
      const {{ minX, maxX, minY, maxY, cx, cy }} = bounds;
      const handleSize = 8;
      
      const handles = [
        {{ x: minX, y: minY, cursor: "nw-resize", type: "corner" }},
        {{ x: maxX, y: minY, cursor: "ne-resize", type: "corner" }},
        {{ x: maxX, y: maxY, cursor: "se-resize", type: "corner" }},
        {{ x: minX, y: maxY, cursor: "sw-resize", type: "corner" }},
        {{ x: cx, y: minY, cursor: "n-resize", type: "edge" }},
        {{ x: maxX, y: cy, cursor: "e-resize", type: "edge" }},
        {{ x: cx, y: maxY, cursor: "s-resize", type: "edge" }},
        {{ x: minX, y: cy, cursor: "w-resize", type: "edge" }},
      ];
      
      window["_selectionHandles_" + iframeId] = handles;
      
      // Bounding box
      labelCtx.strokeStyle = "rgba(59, 130, 246, 0.8)";
      labelCtx.lineWidth = 1;
      labelCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
      
      // Rotation handle
      const rotHandleY = minY - 25;
      labelCtx.beginPath();
      labelCtx.moveTo(cx, minY);
      labelCtx.lineTo(cx, rotHandleY);
      labelCtx.stroke();
      
      labelCtx.beginPath();
      labelCtx.arc(cx, rotHandleY - 8, 8, 0, Math.PI * 2);
      labelCtx.fillStyle = "white";
      labelCtx.fill();
      labelCtx.strokeStyle = "rgba(59, 130, 246, 0.8)";
      labelCtx.lineWidth = 2;
      labelCtx.stroke();
      
      // Rotation icon
      labelCtx.beginPath();
      labelCtx.arc(cx, rotHandleY - 8, 4, -Math.PI * 0.7, Math.PI * 0.5);
      labelCtx.strokeStyle = "rgba(59, 130, 246, 0.9)";
      labelCtx.lineWidth = 1.5;
      labelCtx.stroke();
      labelCtx.beginPath();
      labelCtx.moveTo(cx + 3, rotHandleY - 4);
      labelCtx.lineTo(cx + 5, rotHandleY - 7);
      labelCtx.lineTo(cx + 1, rotHandleY - 6);
      labelCtx.fillStyle = "rgba(59, 130, 246, 0.9)";
      labelCtx.fill();
      
      window["_rotationHandle_" + iframeId] = {{ x: cx, y: rotHandleY - 8, r: 10 }};
      
      // Resize handles
      handles.forEach(h => {{
        labelCtx.fillStyle = "white";
        labelCtx.fillRect(h.x - handleSize/2, h.y - handleSize/2, handleSize, handleSize);
        labelCtx.strokeStyle = "rgba(59, 130, 246, 0.8)";
        labelCtx.lineWidth = 1.5;
        labelCtx.strokeRect(h.x - handleSize/2, h.y - handleSize/2, handleSize, handleSize);
      }});
      
      // Polygon vertex handles
      if (tool === "polygon" && path.length > 2) {{
        path.forEach(([px, py], idx) => {{
          labelCtx.beginPath();
          labelCtx.arc(px, py, 5, 0, Math.PI * 2);
          labelCtx.fillStyle = "rgba(34, 197, 94, 0.9)";
          labelCtx.fill();
          labelCtx.strokeStyle = "white";
          labelCtx.lineWidth = 2;
          labelCtx.stroke();
        }});
      }}
    }}
    
    labelCtx.restore();
  }}
  
  // Draw selection outline (calls draw to refresh everything)
  function drawSelectionOutline() {{
    draw();
  }}
  
  // Convert canvas coordinates to data coordinates
  function canvasToData(canvasX, canvasY) {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    const meta = METADATA[currentEmbedding] || METADATA.spatial || {{}};
    const minX = meta.minX ?? 0;
    const maxX = meta.maxX ?? 1;
    const minY = meta.minY ?? 0;
    const maxY = meta.maxY ?? 1;
    
    const pad = 12;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    const scale = baseScale * zoom;
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (W - usedW) / 2 + panX;
    const offY = (H - usedH) / 2 + panY;
    
    // Reverse rotation first
    const centerX = W / 2;
    const centerY = H / 2;
    const cos = Math.cos(-rotation);
    const sin = Math.sin(-rotation);
    const dx = canvasX - centerX;
    const dy = canvasY - centerY;
    const unrotatedX = centerX + (dx * cos - dy * sin);
    const unrotatedY = centerY + (dx * sin + dy * cos);
    
    // Then reverse the pan/scale/offset
    const dataX = minX + (unrotatedX - offX) / scale;
    const dataY = minY + (unrotatedY - offY) / scale;
    
    return [dataX, dataY];
  }}
  
  // Convert data coordinates to canvas coordinates
  function dataToCanvas(dataX, dataY) {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    const meta = METADATA[currentEmbedding] || METADATA.spatial || {{}};
    const minX = meta.minX ?? 0;
    const maxX = meta.maxX ?? 1;
    const minY = meta.minY ?? 0;
    const maxY = meta.maxY ?? 1;
    
    const pad = 12;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    const scale = baseScale * zoom;
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (W - usedW) / 2 + panX;
    const offY = (H - usedH) / 2 + panY;
    
    // Apply scale and offset
    let px = offX + (dataX - minX) * scale;
    let py = offY + (dataY - minY) * scale;
    
    // Apply rotation around canvas center
    const centerX = W / 2;
    const centerY = H / 2;
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    const dx = px - centerX;
    const dy = py - centerY;
    const canvasX = centerX + (dx * cos - dy * sin);
    const canvasY = centerY + (dx * sin + dy * cos);
    
    return [canvasX, canvasY];
  }}
  
  // Convert a path from canvas coords to data coords
  function pathCanvasToData(canvasPath) {{
    return canvasPath.map(([cx, cy]) => canvasToData(cx, cy));
  }}
  
  // Convert a path from data coords to canvas coords
  function pathDataToCanvas(dataPath) {{
    return dataPath.map(([dx, dy]) => dataToCanvas(dx, dy));
  }}
  
  // Complete selection and detect points inside
  function completeSelection() {{
    const path = window["_selectionPath_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    if (!path || path.length === 0) return;
    
    console.log("[Selection] Completing with tool:", tool, "path length:", path.length);
    
    // Get current plot state to check categorical mask
    const state = window["_plotState_" + iframeId] || {{ obs: {{}} }};
    
    // Get current embedding's position array
    const positions = getEmbeddingPositions(currentEmbedding);
    const numPoints = loadedCount;
    
    if (numPoints === 0) {{
      console.log("[Selection] No points loaded yet");
      return;
    }}
    
    // Convert selection path from canvas coords to data coords
    const selectedIndices = [];
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    // Get bounds from METADATA for current embedding
    const meta = METADATA[currentEmbedding] || METADATA.spatial || {{}};
    const minX = meta.minX ?? 0;
    const maxX = meta.maxX ?? 1;
    const minY = meta.minY ?? 0;
    const maxY = meta.maxY ?? 1;
    
    const pad = 12;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
    
    // Apply zoom, pan, rotation (same as draw function)
    const scale = baseScale * zoom;
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (W - usedW) / 2 + panX;
    const offY = (H - usedH) / 2 + panY;
    
    // Check each point
    for (let i = 0; i < numPoints; i++) {{
      // IMPORTANT: Only select from visible (unmasked) points
      let passesMask = true;
      
      // Check categorical masking from obsValues (1-indexed category) and enabled array
      if (currentObsColumn && state.obs && state.obs.enabled) {{
        const categoryIdx = obsValues[i];  // 0 = no category, 1+ = category index
        if (categoryIdx > 0) {{
          const catIndex = categoryIdx - 1;  // Convert to 0-indexed
          if (state.obs.enabled[catIndex] === false) {{
            passesMask = false;
          }}
        }}
      }}
      
      // Skip masked points
      if (!passesMask) continue;
      
      // Get point coordinates from typed array
      const x = positions[i * 2];
      const y = positions[i * 2 + 1];
      
      // Transform to canvas coordinates
      let px = offX + (x - minX) * scale;
      let py = offY + (y - minY) * scale;
      
      // Apply rotation transform to point (same as draw function)
      // Rotate around canvas center
      const centerX = W / 2;
      const centerY = H / 2;
      const dx = px - centerX;
      const dy = py - centerY;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      px = centerX + (dx * cos - dy * sin);
      py = centerY + (dx * sin + dy * cos);
      
      let inside = false;
      
      if (tool === "lasso" || tool === "polygon") {{
        inside = pointInPolygon(px, py, path);
      }} else if (tool === "rectangle") {{
        const [x1, y1] = path[0];
        const [x2, y2] = path[1];
        const minRX = Math.min(x1, x2), maxRX = Math.max(x1, x2);
        const minRY = Math.min(y1, y2), maxRY = Math.max(y1, y2);
        inside = px >= minRX && px <= maxRX && py >= minRY && py <= maxRY;
      }} else if (tool === "circle") {{
        const [cx, cy] = path[0];
        const [ex, ey] = path[1];
        const rx = Math.abs(ex - cx);
        const ry = path.length > 2 ? Math.abs(path[2][1] - cy) : rx;
        // Ellipse equation: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 <= 1
        if (rx > 0 && ry > 0) {{
          const ddx = (px - cx) / rx;
          const ddy = (py - cy) / ry;
          inside = (ddx * ddx + ddy * ddy) <= 1;
        }}
      }}
      
      if (inside) {{
        selectedIndices.push(i);
      }}
    }}
    
    console.log("[Selection] Found", selectedIndices.length, "points inside selection");
    
    // Convert path to DATA coordinates for storage (so it survives pan/zoom/rotate)
    const dataPath = pathCanvasToData(path);
    
    // Send selection to iframe with path in DATA coords
    iframe.contentWindow.postMessage({{
      type: "selection_completed",
      indices: selectedIndices,
      path: dataPath,  // Store in data coordinates!
      tool: tool
    }}, "*");
    
    // Keep the path for editing (don't clear it)
    // Just mark as not drawing anymore
    window["_isDrawing_" + iframeId] = false;
    drawSelectionOutline();  // Redraw with handles
  }}
  
  // Point in polygon test (ray casting)
  function pointInPolygon(x, y, polygon) {{
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {{
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      
      const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }}
    return inside;
  }}

}})();
</script>
"""
    display(HTML(container_html))

    # ----------------------------
    # Poll buttons: one hidden ipywidgets.Button per iframe button id
    # ----------------------------
    for bid in button_ids:
        label = f"_poll_{iframe_id}__{bid}"
        poll_btn = widgets.Button(
            description=label,
            tooltip=label,
            layout=widgets.Layout(width="0px", height="0px", visibility="hidden", display="none"),
        )

        def make_handler(button_id: str):
            def handler(_b):
                try:
                    bridge_value = data_bridges[button_id].value
                    request_data: Dict[str, Any] = {}

                    if bridge_value:
                        try:
                            request_data = json.loads(bridge_value)
                        except json.JSONDecodeError as e:
                            if debug:
                                with output:
                                    print(f"[Python] JSON decode error: {e}, value: {bridge_value}")

                    if debug:
                        with output:
                            print(f"[Python] Handling {button_id!r}")
                            print(f"[Python] Request data: {request_data}")

                    cb = button_callbacks[button_id]
                    result = cb(request_data, **callback_args)

                    serialized = _serialize_result(result)
                    json_str = json.dumps(serialized)
                    size = len(json_str.encode("utf-8"))
                    if size > max_result_size:
                        serialized = {"type": "error", "message": f"Result too large: {size:,} bytes"}

                    with output:
                        js_code = f"""
                        (function() {{
                          const iframeId = {json.dumps(iframe_id)};
                          const sendFn = window["sendToIframe_" + iframeId];
                          const updFn  = window["updatePlot_" + iframeId];
                          if (sendFn) sendFn({json.dumps(serialized)});
                          if (updFn)  updFn({json.dumps(serialized)});
                        }})();
                        """
                        display(Javascript(js_code))

                    data_bridges[button_id].value = ""

                except Exception as e:
                    if debug:
                        with output:
                            print(f"[Python] Error in {button_id}:")
                            traceback.print_exc()

                    error_result = {
                        "type": "error",
                        "message": str(e),
                        "traceback": traceback.format_exc() if debug else None,
                    }

                    with output:
                        js_code = f"""
                        (function() {{
                          const iframeId = {json.dumps(iframe_id)};
                          const sendFn = window["sendToIframe_" + iframeId];
                          const updFn  = window["updatePlot_" + iframeId];
                          if (sendFn) sendFn({json.dumps(error_result)});
                          if (updFn)  updFn({json.dumps(error_result)});
                        }})();
                        """
                        display(Javascript(js_code))

            return handler

        poll_btn.on_click(make_handler(bid))
        display(poll_btn)

    # ----------------------------
    # Dispatcher: moves queued iframe clicks into the correct hidden widget bridge + triggers poll button
    # ----------------------------
    dispatch_script = f"""
<script>
(function() {{
  const iframeId = {json.dumps(iframe_id)};
  let busy = false;

  function findAndClick(label) {{
    const btns = Array.from(document.querySelectorAll("button"));
    for (const b of btns) {{
      const txt = (b.textContent || "");
      const title = (b.getAttribute("title") || "");
      const aria = (b.getAttribute("aria-label") || "");
      if (txt.includes(label) || title.includes(label) || aria.includes(label)) {{
        b.click();
        return true;
      }}
    }}
    return false;
  }}

  function fireWidgetEvents(inputEl) {{
    inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
    inputEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }}

  function findAndSetInput(description, value) {{
    const parts = description.split('_');
    const buttonId = parts[parts.length - 1];
    const iframeId2 = parts.slice(2, parts.length - 1).join('_'); // iframe_<hex>

    // Strategy 1: scope to (iframeId, buttonId) to avoid stale widgets
    const scopedSelector = '.data-bridge-' + iframeId2 + '.data-bridge-' + buttonId + ' input';
    const scoped = document.querySelector(scopedSelector);
    if (scoped) {{
      scoped.value = value;
      fireWidgetEvents(scoped);
      return true;
    }}

    // Strategy 2: match by widget label text containing the full description
    const allContainers = Array.from(document.querySelectorAll('.widget-text, .jupyter-widgets'));
    for (const container of allContainers) {{
      const label = container.querySelector('.widget-label');
      if (label && label.textContent && label.textContent.includes(description)) {{
        const input = container.querySelector('input');
        if (input) {{
          input.value = value;
          fireWidgetEvents(input);
          return true;
        }}
      }}
    }}
    return false;
  }}

  function dispatch() {{
    if (busy) return;

    const q = window["_requests_" + iframeId] || [];
    if (q.length === 0) return;

    busy = true;
    const req = q.shift();

    const pollLabel = "_poll_" + iframeId + "__" + req.buttonId;
    const dataLabel = "_data_" + iframeId + "_" + req.buttonId;

    findAndSetInput(dataLabel, JSON.stringify(req.data));

    setTimeout(() => {{
      findAndClick(pollLabel);
      setTimeout(() => {{ busy = false; }}, 100);
    }}, 150);
  }}

  setInterval(dispatch, 150);
}})();
</script>
"""
    display(HTML(dispatch_script))

    output.layout.visibility = "hidden"
    output.layout.height = "0px"
    display(output)

    return iframe_id

# PART 2! BASE HTML:
# html_template 2:
html_template = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    :root{
      --accent: rgba(141,236,245,.9);
      --accentSolid: rgba(141,236,245,1);
      --accentRGB: rgb(141,236,245);
    }

    html, body {
      height:100%;
      margin:0;
      padding:0;
      background: transparent;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }

    /* Panel styling (copied/trimmed from _spatial_scatter.py) */
    .panel{
      width:100%;
      height:100%;
      overflow:auto;
      border:1px solid rgba(0,0,0,.15);
      border-radius:10px;
      padding:16px;
      background:rgba(255,255,255,.98);
      box-sizing:border-box;
    }

    .control-group{ margin-bottom:20px; }
    .control-label{
      font-size:13px;
      font-weight:700;
      color:#444;
      margin-bottom:8px;
      display:block;
    }

    .embedding-selector{
      display:inline-flex;
      align-items:center;
      gap:0;
      font-size:13px;
      background:rgba(0,0,0,.05);
      border-radius:20px;
      padding:4px;
    }
    .embedding-btn{
      background:none;
      border:none;
      color:#888;
      cursor:pointer;
      padding:6px 16px;
      transition:all .2s;
      font-size:13px;
      font-weight:600;
      border-radius:16px;
    }
    .embedding-btn:hover{ color:#555; }
    .embedding-btn.active{
      background:var(--accent);
      color:#000;
      box-shadow:0 2px 8px rgba(0,0,0,.2);
    }
    .embedding-btn:disabled{ opacity:.3; cursor:not-allowed; }
    .separator{ color:#ccc; padding:0 4px; }

    input[type="text"]{
      width:100%;
      padding:8px 12px;
      border:1px solid rgba(0,0,0,.2);
      border-radius:8px;
      font-size:13px;
      background:white;
      box-sizing:border-box;
      font-family: ui-monospace, Menlo, Monaco, monospace;
    }
    input[type="text"]:focus{
      outline:none;
      border-color:var(--accent);
    }

    .btn-row{ display:flex; gap:8px; margin-top:8px; }
    .btn-primary{
      flex:1;
      padding:8px 16px;
      border:none;
      background:var(--accent);
      color:#000;
      border-radius:8px;
      cursor:pointer;
      font-size:13px;
      font-weight:800;
      transition:all .2s;
    }
    .btn-primary:hover{
      background:var(--accentSolid);
      box-shadow:0 2px 8px rgba(0,0,0,.2);
    }
    .btn-secondary{
      flex:1;
      padding:8px 16px;
      border:1px solid rgba(0,0,0,.25);
      background:rgba(255,255,255,.7);
      color:#222;
      border-radius:8px;
      cursor:pointer;
      font-size:13px;
      font-weight:800;
      transition:all .2s;
    }
    .btn-secondary:hover{
      background:rgba(255,255,255,.9);
      box-shadow:0 2px 8px rgba(0,0,0,.12);
    }

    .opacity-row{ display:flex; align-items:center; gap:10px; }
    .opacity-val{
      min-width:44px;
      text-align:right;
      font-size:12px;
      color:#666;
      font-variant-numeric:tabular-nums;
      font-family: ui-monospace, Menlo, Monaco, monospace;
    }

    /* Slider styling (filled left only) */
    input[type="range"]{
      width:100%;
      appearance:none;
      -webkit-appearance:none;
      background:#ffffff;
      border:none;
      outline:none;
      --p: 100%;
    }
    input[type="range"]::-webkit-slider-runnable-track{
      height:6px;
      border-radius:999px;
      background:
        linear-gradient(
          to right,
          var(--accentRGB) 0%,
          var(--accentRGB) var(--p),
          #ffffff var(--p),
          #ffffff 100%
        );
    }
    input[type="range"]::-webkit-slider-thumb{
      -webkit-appearance:none;
      appearance:none;
      width:14px;
      height:14px;
      border-radius:50%;
      background: var(--accentRGB);
      border: 2px solid #ffffff;
      margin-top:-4px;
    }

    /* Legend (toggle list) */
    .legend{
      display:none;
      margin-top:10px;
      border:1px solid rgba(0,0,0,.10);
      background: rgba(255,255,255,.92);
      border-radius:10px;
      overflow:hidden;
    }
    .legend.visible{ display:block; }

    .legend-header{
      padding:10px 10px 8px 10px;
      border-bottom:1px solid rgba(0,0,0,.08);
    }
    .legend-title{
      font-size:12px;
      font-weight:900;
      color:#444;
    }
    .legend-actions{
      margin-top:8px;
      display:flex;
      gap:8px;
    }
    .legend-btn{
      flex:1;
      padding:6px 10px;
      border-radius:8px;
      border:1px solid rgba(0,0,0,.12);
      background: rgba(255,255,255,.85);
      cursor:pointer;
      font-size:12px;
      font-weight:900;
      color:#333;
      transition: all .15s ease;
    }
    .legend-btn:hover{
      background: rgba(255,255,255,1);
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
    }
    .legend-body{
      max-height:220px;
      overflow:auto;
      padding:6px 10px;
    }
    .legend-row{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      padding:6px 2px;
      border-bottom:1px solid rgba(0,0,0,.06);
      cursor:pointer;
      user-select:none;
    }
    .legend-row:last-child{ border-bottom:none; }
    .legend-row:hover{ background:rgba(0,0,0,.03); }

    .legend-label{
      font-size:12px;
      color:#333;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      max-width:200px;
    }
    .legend-dot{
      width:16px;height:16px;border-radius:999px;border:2px solid rgba(0,0,0,.12);
      flex:0 0 auto;cursor:pointer;position:relative;transition:transform .1s ease;
    }
    .legend-dot:hover{ transform:scale(1.3); }
    .legend-dot input[type="color"]{
      position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;cursor:pointer;border:0;padding:0;
    }
    .legend-row.off .legend-label{ opacity:.35; text-decoration:line-through; }
    .legend-row.off .legend-dot{ opacity:.15; }
    .cmap-dropdown{
      display:none;max-height:240px;overflow-y:auto;border:1px solid rgba(0,0,0,.1);
      border-radius:8px;margin-top:4px;background:#fff;box-shadow:0 4px 16px rgba(0,0,0,.08);
    }
    .cmap-dropdown-search{
      position:sticky;top:0;background:#fff;padding:6px 8px;border-bottom:1px solid rgba(0,0,0,.06);z-index:2;
    }
    .cmap-dropdown-search input{
      width:100%;box-sizing:border-box;font-size:11px;padding:5px 8px;
      background:rgba(0,0,0,.03);border:1px solid rgba(0,0,0,.08);border-radius:6px;color:#333;outline:none;
    }
    .cmap-dropdown-search input:focus{ border-color:rgba(59,130,246,.4); }
    .cmap-row{
      display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;transition:background .1s;
    }
    .cmap-row:hover{ background:rgba(59,130,246,.06); }
    .cmap-row.active{ background:rgba(59,130,246,.12); }
    .cmap-row-name{ font-size:11px;color:#666;min-width:54px;flex-shrink:0; }
    .cmap-row-bar{ flex:1;height:16px;border-radius:4px;display:block; }

    /* Gene chips */
    #geneChips{
      padding:8px;
      background:rgba(0,0,0,.02);
      border-radius:8px;
      border:1px solid rgba(0,0,0,.08);
      margin-top:10px;
    }
    #geneChipContainer{
      display:flex;
      flex-wrap:wrap;
      gap:6px;
    }
    #geneChipContainer.drag-over{
      border-color:rgba(59,130,246,.5);
      background:rgba(59,130,246,.05);
    }
    .gene-chip{
      display:inline-flex;align-items:center;gap:6px;padding:5px 10px;
      background:rgba(0,0,0,.04);border:1px solid rgba(0,0,0,.12);
      border-radius:12px;font-size:11px;font-weight:800;color:#888;cursor:grab;transition:all .15s;
    }
    .gene-chip:hover{ background:rgba(0,0,0,.08);border-color:rgba(0,0,0,.2);color:#555; }
    .gene-chip.active{
      background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.6);
      color:#2563eb;box-shadow:0 0 0 2px rgba(59,130,246,.12);
    }
    .gene-chip.active:hover{ background:rgba(59,130,246,.22);border-color:rgba(59,130,246,.8); }
    .gene-chip.dragging{ opacity:0.5; cursor:grabbing; }
    .gene-chip-remove{
      width:14px;
      height:14px;
      display:flex;
      align-items:center;
      justify-content:center;
      background:rgba(0,0,0,.15);
      border-radius:50%;
      font-size:10px;
      line-height:1;
      transition:all .15s;
    }
    .gene-chip-remove:hover{
      background:rgba(255,0,0,.6);
      color:white;
    }
    
    /* Gene group folders */
    .gene-group-folder{
      margin-bottom:8px;
      background:rgba(147,51,234,0.03);
      border:1px solid rgba(147,51,234,0.15);
      border-radius:8px;
      overflow:hidden;
    }
    .gene-group-folder.drag-over{
      border-color:rgba(147,51,234,0.5);
      background:rgba(147,51,234,0.08);
    }
    .gene-group-header{
      display:flex;
      align-items:center;
      gap:6px;
      padding:6px 10px;
      background:rgba(147,51,234,0.05);
      cursor:pointer;
      font-size:11px;
      font-weight:600;
      color:#6b21a8;
    }
    .gene-group-header:hover{ background:rgba(147,51,234,0.1); }
    .gene-group-header.active{
      background:rgba(147,51,234,0.2);
      color:#581c87;
    }
    .gene-group-arrow{
      font-size:8px;
      transition:transform 0.2s;
    }
    .gene-group-arrow.expanded{ transform:rotate(90deg); }
    .gene-group-content{
      display:none;
      padding:6px;
      min-height:28px;
      border-top:1px solid rgba(147,51,234,0.1);
    }
    .gene-group-content.expanded{ display:flex; flex-wrap:wrap; gap:4px; }
    .gene-group-name{
      flex:1;
      background:transparent;
      border:none;
      font-size:11px;
      font-weight:600;
      color:#6b21a8;
      cursor:text;
      padding:0;
    }
    .gene-group-name:focus{ outline:none; background:rgba(255,255,255,0.5); border-radius:2px; }
    .gene-group-count{ font-size:9px; color:#9333ea; font-weight:400; }
    .gene-group-remove{
      width:16px;height:16px;display:flex;align-items:center;justify-content:center;
      background:rgba(147,51,234,0.1);border-radius:50%;font-size:10px;color:#9333ea;
      cursor:pointer;transition:all 0.15s;
    }
    .gene-group-remove:hover{ background:rgba(255,0,0,0.6); color:white; }
    
    /* Selection group folders (orange/amber theme) */
    .sel-group-folder{
      margin-bottom:8px;
      background:rgba(245,158,11,0.03);
      border:1px solid rgba(245,158,11,0.15);
      border-radius:8px;
      overflow:hidden;
    }
    .sel-group-folder.drag-over{
      border-color:rgba(245,158,11,0.5);
      background:rgba(245,158,11,0.08);
    }
    .sel-group-header{
      display:flex;
      align-items:center;
      gap:6px;
      padding:6px 10px;
      background:rgba(245,158,11,0.05);
      cursor:pointer;
      font-size:11px;
      font-weight:600;
      color:#b45309;
    }
    .sel-group-header:hover{ background:rgba(245,158,11,0.1); }
    .sel-group-header.active{
      background:rgba(245,158,11,0.2);
      color:#92400e;
    }
    .sel-group-arrow{
      font-size:8px;
      transition:transform 0.2s;
    }
    .sel-group-arrow.expanded{ transform:rotate(90deg); }
    .sel-group-content{
      display:none;
      padding:6px;
      min-height:28px;
      border-top:1px solid rgba(245,158,11,0.1);
    }
    .sel-group-content.expanded{ display:flex; flex-wrap:wrap; gap:4px; }
    .sel-group-name{
      flex:1;
      background:transparent;
      border:none;
      font-size:11px;
      font-weight:600;
      color:#b45309;
      cursor:text;
      padding:0;
    }
    .sel-group-name:focus{ outline:none; background:rgba(255,255,255,0.5); border-radius:2px; }
    .sel-group-count{ font-size:9px; color:#d97706; font-weight:400; }
    .sel-group-remove{
      width:16px;height:16px;display:flex;align-items:center;justify-content:center;
      background:rgba(245,158,11,0.1);border-radius:50%;font-size:10px;color:#d97706;
      cursor:pointer;transition:all 0.15s;
    }
    .sel-group-remove:hover{ background:rgba(255,0,0,0.6); color:white; }
    
    /* Selection chips */
    .sel-chip{
      display:inline-flex;align-items:center;gap:6px;padding:5px 10px;
      background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
      border-radius:12px;font-size:11px;font-weight:600;color:#b45309;cursor:grab;transition:all .15s;
    }
    .sel-chip:hover{ background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.4); }
    .sel-chip.active{
      background:rgba(245,158,11,0.25);border-color:rgba(245,158,11,0.6);
      box-shadow:0 0 0 2px rgba(245,158,11,0.15);
    }
    .sel-chip.dragging{ opacity:0.5; cursor:grabbing; }
    .sel-chip-name{
      background:transparent;border:none;font-size:11px;font-weight:600;color:inherit;
      cursor:text;padding:0;max-width:80px;
    }
    .sel-chip-name:focus{ outline:none; }
    .sel-chip-count{ font-size:9px; color:#d97706; }
    .sel-chip-remove{
      width:14px;height:14px;display:flex;align-items:center;justify-content:center;
      background:rgba(245,158,11,0.2);border-radius:50%;font-size:10px;
      transition:all .15s;
    }
    .sel-chip-remove:hover{ background:rgba(255,0,0,0.6); color:white; }
    #selectionChipContainer.drag-over{
      border-color:rgba(245,158,11,.5);
      background:rgba(245,158,11,.05);
    }

    /* Selection tool buttons */
    .tool-btn{
      flex:1;
      padding:8px 10px;
      border-radius:8px;
      border:1px solid rgba(0,0,0,.12);
      background: rgba(255,255,255,.85);
      cursor:pointer;
      display:flex;
      align-items:center;
      justify-content:center;
      transition: all .15s ease;
      font-size:11px;
      font-weight:700;
      color:#333;
    }
    .tool-btn:hover{
      background: rgba(255,255,255,1);
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
    }
    .tool-btn.active{
      background: rgba(141,236,245,.25);
      border-color: rgba(141,236,245,.6);
      box-shadow: 0 2px 10px rgba(0,0,0,.10);
    }

    /* Collapsible section headers */
    .section-header{
      display:flex;
      align-items:center;
      gap:6px;
      cursor:pointer;
      user-select:none;
      margin-bottom:10px;
    }
    .section-arrow{
      font-size:10px;
      transition: transform .15s ease;
      color:#666;
    }
    .section-arrow.expanded{
      transform: rotate(90deg);
    }
    .section-content{
      display:none;
    }
    .section-content.expanded{
      display:block;
    }

    /* Selection folder */
    .selection-folder{
      margin-bottom:8px;
    }
    .folder-header{
      display:flex;
      align-items:center;
      gap:6px;
      padding:8px 10px;
      border-radius:8px;
      border:1px solid rgba(0,0,0,.12);
      background: rgba(240,240,240,.85);
      cursor:pointer;
      user-select:none;
      transition: all .15s ease;
    }
    .folder-header:hover{
      background: rgba(240,240,240,1);
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }
    .folder-header.active{
      background: rgba(141,236,245,.15);
      border-color: rgba(141,236,245,.5);
    }
    .folder-arrow{
      font-size:10px;
      transition: transform .15s ease;
      flex:0 0 auto;
    }
    .folder-arrow.expanded{
      transform: rotate(90deg);
    }
    .folder-name{
      flex:1;
      font-size:12px;
      font-weight:700;
      color:#333;
    }
    .folder-children{
      margin-left:20px;
      margin-top:4px;
      display:none;
    }
    .folder-children.expanded{
      display:block;
    }
    .gene-chip.multi-selected{
      border:2px solid rgba(141,236,245,.8);
      background: rgba(141,236,245,.1);
    }

    .hint{
      font-size:11px;
      color:#666;
      margin-top:8px;
      font-style:italic;
    }
    /* Hide number input spinners */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button { -webkit-appearance:none; margin:0; }
    input[type="number"] { -moz-appearance:textfield; }
  </style>
</head>

<body>
  <div class="panel">
    <!-- Embedding -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('embedding')">
        <span class="section-arrow expanded" id="embedding-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Embedding</label>
      </div>
      <div class="section-content expanded" id="embedding-content">
        <div class="embedding-selector">
          <button class="embedding-btn active" id="spatialBtn">Spatial</button>
          <span class="separator">|</span>
          <button class="embedding-btn" id="umapBtn">UMAP</button>
          <span class="separator">|</span>
          <button class="embedding-btn" id="pcaBtn">PCA</button>
        </div>
      </div>
    </div>

    <!-- Point size -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('size')">
        <span class="section-arrow expanded" id="size-arrow">&#9654;</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Size</label>
      </div>
      <div class="section-content expanded" id="size-content">
        <div class="opacity-row">
          <input type="range" id="sizeSlider" min="0.5" max="8" step="0.1" value="1.1">
          <div class="opacity-val" id="sizeVal">1.1</div>
        </div>
      </div>
    </div>

    <!-- LAYOUT (grid arrangement) -->
    <div class="control-group" id="layoutSection">
      <div class="section-header" onclick="toggleSection('layout')">
        <span class="section-arrow" id="layout-arrow">&#9654;</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Layout</label>
      </div>
      <div class="section-content" id="layout-content">
        <!-- Group by -->
        <div style="margin-bottom:6px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:11px;color:#888;min-width:58px;">Group by</span>
            <input type="text" id="layoutGroupBy" list="layoutGroupByList" placeholder="obs column..." 
                   style="flex:1;font-size:11px;padding:4px 6px;border:1px solid rgba(0,0,0,.12);border-radius:6px;background:#fff;">
            <datalist id="layoutGroupByList"></datalist>
            <span style="font-size:10px;color:#888;">Grid</span>
            <input type="number" id="layoutGroupCols" min="1" max="20" value="2" 
                   style="width:36px;font-size:11px;padding:4px;border:1px solid rgba(0,0,0,.12);border-radius:6px;text-align:center;"
                   title="Number of columns in group grid">
          </div>
        </div>
        
        <!-- Columns (determines columns within each group) -->
        <div style="margin-bottom:6px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:11px;color:#888;min-width:58px;">Columns</span>
            <input type="text" id="layoutSortBy" list="layoutSortByList" placeholder="leave blank for auto..."
                   style="flex:1;font-size:11px;padding:4px 6px;border:1px solid rgba(0,0,0,.12);border-radius:6px;background:#fff;"
                   title="Each unique value becomes a column. Leave blank for auto-square grid.">
            <datalist id="layoutSortByList"></datalist>
          </div>
        </div>
        
        <!-- Rows (determines row ordering within columns) -->
        <div style="margin-bottom:6px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:11px;color:#888;min-width:58px;">Rows</span>
            <input type="text" id="layoutOrderBy" list="layoutOrderByList" placeholder="optional row order..."
                   style="flex:1;font-size:11px;padding:4px 6px;border:1px solid rgba(0,0,0,.12);border-radius:6px;background:#fff;"
                   title="Controls vertical ordering within each column">
            <datalist id="layoutOrderByList"></datalist>
          </div>
        </div>
        
        <!-- Gap + Transpose row -->
        <div style="margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:11px;color:#888;min-width:58px;">Gap</span>
            <input type="range" id="layoutGap" min="0" max="3" step="0.05" value="2.00" style="flex:1;">
            <span style="font-size:10px;color:#888;min-width:24px;" id="layoutGapVal">2.00</span>
            <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:3px;cursor:pointer;margin-left:4px;">
              <input type="checkbox" id="layoutTranspose" style="margin:0;width:14px;height:14px;">
              <span>T</span>
            </label>
          </div>
        </div>
        
        <!-- Action buttons -->
        <div style="display:flex;gap:6px;margin-bottom:6px;">
          <button class="btn-primary" id="computeLayoutBtn" style="flex:1;font-size:11px;padding:6px 8px;">Compute</button>
          <button class="btn-secondary" id="saveLayoutBtn" style="flex:1;font-size:11px;padding:6px 8px;">Save</button>
        </div>
        
        <!-- Label toggle -->
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
          <label style="font-size:11px;color:#888;display:flex;align-items:center;gap:4px;cursor:pointer;">
            <input type="checkbox" id="layoutLabelToggle" checked style="margin:0;">
            Show sample labels
          </label>
        </div>
        
        <!-- Saved layouts list -->
        <div id="layoutSavedList" style="border-top:1px solid rgba(0,0,0,0.08);padding-top:6px;">
          <div style="font-size:10px;color:#999;margin-bottom:4px;">Layouts</div>
          <div id="layoutSavedItems"></div>
        </div>
        
        <!-- Hidden buttons for Python callbacks -->
        <button id="deleteLayoutBtn" style="display:none;"></button>
        <button id="loadLayoutBtn" style="display:none;"></button>
        <button id="sampleMetaBtn" style="display:none;"></button>
        <button id="obsmBtn" style="display:none;"></button>
        <button id="geneGroupBtn" style="display:none;"></button>
      </div>
    </div>

    <!-- GEX (base layer) -->
    <div class="control-group" id="gexSection">
      <div class="section-header" onclick="toggleSection('gex')">
        <span class="section-arrow" id="gex-arrow">&#9654;</span>
        <label class="control-label" style="margin:0;cursor:pointer;">GEX (gene)</label>
      </div>
      <div class="section-content" id="gex-content">
        <input type="text" id="gexInput" data-for="geneBtn" data-key="gene" placeholder="Enter gene name" list="geneList" />
        <datalist id="geneList"></datalist>
        <div style="margin-top:10px">
          <div class="opacity-row">
            <span style="font-size:11px;color:#888;min-width:48px;">Opacity</span>
            <input type="range" id="gexOpacity" min="0" max="1" step="0.05" value="1.0">
            <div class="opacity-val" id="gexOpacityVal">1.00</div>
          </div>
        </div>
        <!-- Colormap: bar with min/max, no enclosing box -->
        <div style="margin-top:8px;">
          <div style="font-size:11px;color:#888;margin-bottom:4px;">Colormap</div>
          <div id="cmapSelected" style="cursor:pointer;display:flex;align-items:center;gap:8px;">
            <span id="cmapName" style="font-size:11px;color:#666;min-width:54px;flex-shrink:0;">viridis</span>
            <div style="flex:1;position:relative;">
              <div style="display:flex;justify-content:space-between;font-size:9px;color:#999;margin-bottom:2px;">
                <span id="cmapMinLabel">0</span><span id="cmapMaxLabel">max</span>
              </div>
              <canvas id="cmapPreview" style="width:100%;height:16px;border-radius:4px;display:block;cursor:pointer;" height="16"></canvas>
            </div>
            <span style="font-size:9px;color:#999;flex-shrink:0;">&#9660;</span>
          </div>
          <div class="cmap-dropdown" id="cmapDropdown">
            <div class="cmap-dropdown-search"><input type="text" id="cmapSearch" placeholder="Search colormaps..."></div>
            <div id="cmapList"></div>
          </div>
        </div>
      <div class="btn-row">
        <button class="btn-primary" id="geneBtn">Add Gene</button>
        <button class="btn-secondary" id="clearGexBtn">Clear</button>
        <button class="btn-secondary" id="createGeneGroupBtn" style="font-size:10px;">+ Group</button>
      </div>
      
      <!-- Gene Groups (collapsible folders with drag-drop) -->
      <div id="geneGroupsSection" style="margin-top:8px;">
        <div id="geneGroupContainer"></div>
      </div>
      
      <!-- Ungrouped genes (drag source/target) -->
      <div id="geneChips" style="margin-top:8px;">
        <div style="font-size:11px;font-weight:800;color:#666;margin-bottom:6px;">Genes <span style="font-weight:400;color:#999;">(drag to group)</span>:</div>
        <div id="geneChipContainer" style="display:flex;flex-wrap:wrap;gap:4px;min-height:24px;padding:4px;border:1px dashed transparent;border-radius:4px;transition:border-color 0.2s;"></div>
      </div>
      </div>
    </div>

    <!-- Color by obs (overlay) -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('colorby')">
        <span class="section-arrow" id="colorby-arrow">&#9654;</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Color By (obs column)</label>
      </div>
      <div class="section-content" id="colorby-content">
        <input type="text" id="obsInput" data-for="obsBtn" data-key="column" placeholder="Enter obs column name" list="obsList" />
        <datalist id="obsList"></datalist>
      <div style="margin-top:10px">
        <div class="opacity-row">
          <span style="font-size:11px;color:#888;min-width:48px;">Opacity</span>
          <input type="range" id="obsOpacity" min="0" max="1" step="0.05" value="1.0">
          <div class="opacity-val" id="obsOpacityVal">1.00</div>
        </div>
      </div>
      <div class="btn-row">
        <button class="btn-primary" id="obsBtn">Apply</button>
        <button class="btn-secondary" id="clearObsBtn">Clear</button>
        <button class="btn-secondary" id="saveColorsBtn" title="Save colors to adata.uns">Save colors</button>
      </div>
      <div id="legend" class="legend"></div>
      </div>
    </div>

    <!-- Selection Tools -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('selection')">
        <span class="section-arrow" id="selection-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Selection</label>
      </div>
      <div class="section-content" id="selection-content">
        <div class="selection-tools" style="display:flex; gap:6px; margin-top:10px;">
          <button class="tool-btn" id="lassoBtn" data-tool="lasso" title="Lasso tool">Lasso</button>
          <button class="tool-btn" id="polygonBtn" data-tool="polygon" title="Polygon tool">Poly</button>
          <button class="tool-btn" id="rectangleBtn" data-tool="rectangle" title="Square/Rectangle tool">Square</button>
          <button class="tool-btn" id="circleBtn" data-tool="circle" title="Circle/Ellipse tool">Circle</button>
        </div>

        <div class="btn-row" style="margin-top:10px;">
          <button class="btn-secondary" id="createSelGroupBtn" style="font-size:10px;">+ Group</button>
          <button class="btn-secondary" id="clearSelectionBtn">Clear All</button>
        </div>

        <!-- Selection Groups (folders with drag-drop, each has save button) -->
        <div id="selectionGroupsSection" style="margin-top:8px;">
          <div id="selectionGroupContainer"></div>
        </div>

        <!-- Ungrouped selections (drag source/target) -->
        <div id="selectionChips" style="margin-top:8px;">
          <div style="font-size:11px;font-weight:800;color:#666;margin-bottom:6px;">Selections <span style="font-weight:400;color:#999;">(drag into group to save)</span>:</div>
          <div id="selectionChipContainer" style="display:flex;flex-wrap:wrap;gap:4px;min-height:24px;padding:4px;border:1px dashed transparent;border-radius:4px;transition:border-color 0.2s;"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // ----------------------------
    // Toggle section collapse/expand
    // ----------------------------
    function toggleSection(sectionId) {
      const content = document.getElementById(sectionId + '-content');
      const arrow = document.getElementById(sectionId + '-arrow');
      content.classList.toggle('expanded');
      arrow.classList.toggle('expanded');
    }

    // ----------------------------
    // Local UI state (iframe-only)
    // ----------------------------
    const STATE = {
      obs: {
        column: null,
        mode: null,              // "categorical"|"continuous"
        values: null,            // per-point codes/values
        categories: null,
        colors: null,            // palette array (optional)
        enabled: null,           // boolean per category (categorical)
        opacity: 1.0
      },
      gex: {
        genes: {},               // gene -> values[]
        order: [],               // list of added genes
        active: null,
        opacity: 1.0
      },
      selection: {
        tool: null,              // "lasso"|"polygon"|"rectangle"|"circle"|null
        selections: {},          // name -> {indices: [...], tool: "..."}
        order: [],               // list of selection names AND folder names
        active: null,            // active selection/folder name (can be array for multi-select)
        counter: 1,              // for auto-naming: Selection 1, Selection 2, etc.
        folderCounter: 1,        // for auto-naming: Group 1, Group 2, etc.
        folders: {},             // folderName -> {selections: [name1, name2], expanded: true/false}
        multiSelect: []          // array of currently selected items for grouping
      },
      pointSize: 1.1
    };

    // ----------------------------
    // Autocomplete from INITIAL_DATA
    // ----------------------------
    function fillDatalist(id, values) {
      const dl = document.getElementById(id);
      dl.innerHTML = "";
      (values || []).forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        dl.appendChild(opt);
      });
    }

    if (window.INITIAL_DATA) {
      fillDatalist("obsList", window.INITIAL_DATA.obs_columns || []);
      fillDatalist("geneList", window.INITIAL_DATA.var_names || []);
    }

    // ----------------------------
    // Sliders → update labels + inform parent plot
    // ----------------------------
    function setRangeFill(rangeEl) {
      const min = parseFloat(rangeEl.min);
      const max = parseFloat(rangeEl.max);
      const v = parseFloat(rangeEl.value);
      const p = ((v - min) / (max - min)) * 100;
      rangeEl.style.setProperty("--p", `${p}%`);
    }

    function postUIState() {
      // Calculate selection indices and path
      let selectionIndices = null;
      let selectionPath = null;
      let selectionTool = null;
      
      // Check if a selection group is active
      if (STATE.selection.activeGroup) {
        const group = STATE.selection.groups[STATE.selection.activeGroup];
        if (group) {
          const combinedIndices = new Set();
          group.selections.forEach(selName => {
            const indices = STATE.selection.selections[selName]?.indices || [];
            indices.forEach(idx => combinedIndices.add(idx));
          });
          selectionIndices = Array.from(combinedIndices);
        }
      } else if (STATE.selection.active) {
        // Single selection active
        const sel = STATE.selection.selections[STATE.selection.active];
        selectionIndices = sel?.indices || null;
        selectionPath = sel?.path || null;
        selectionTool = sel?.tool || null;
      }
      
      // Tell the parent plot what to render (minimal payload)
      window.parent.postMessage({
        type: "ui_state",
        obs: {
          column: STATE.obs.column,
          mode: STATE.obs.mode,
          values: STATE.obs.values,
          categories: STATE.obs.categories,
          colors: STATE.obs.colors,
          enabled: STATE.obs.enabled,
          opacity: STATE.obs.opacity
        },
        gex: {
          active: STATE.gex.active,
          opacity: STATE.gex.opacity,
          colormap: STATE.gex.colormap || "viridis"
        },
        selection: {
          active: STATE.selection.active || STATE.selection.activeGroup,
          indices: selectionIndices,
          path: selectionPath,
          tool: selectionTool
        },
        pointSize: STATE.pointSize
      }, "*");
    }

    // Size
    const sizeSlider = document.getElementById("sizeSlider");
    const sizeVal = document.getElementById("sizeVal");
    setRangeFill(sizeSlider);
    sizeSlider.addEventListener("input", () => {
      setRangeFill(sizeSlider);
      const v = parseFloat(sizeSlider.value);
      STATE.pointSize = v;
      sizeVal.textContent = v.toFixed(1);
      postUIState();
    });

    // ----------------------------
    // LAYOUT section logic
    // ----------------------------
    (function initLayout() {
      const existingLayouts = (window.INITIAL_DATA && window.INITIAL_DATA.existing_layouts) || [];
      const sampleId = (window.INITIAL_DATA && window.INITIAL_DATA.sample_id) || null;
      const obsCols = (window.INITIAL_DATA && window.INITIAL_DATA.obs_columns) || [];
      
      // Populate datalists with all obs columns (same as color by)
      const groupList = document.getElementById("layoutGroupByList");
      const sortList = document.getElementById("layoutSortByList");
      const orderList = document.getElementById("layoutOrderByList");
      
      [groupList, sortList, orderList].forEach(dl => {
        if (!dl) return;
        obsCols.forEach(col => {
          const opt = document.createElement("option");
          opt.value = col;
          dl.appendChild(opt);
        });
      });
      
      // Text inputs
      const groupInput = document.getElementById("layoutGroupBy");
      const sortInput = document.getElementById("layoutSortBy");
      const orderInput = document.getElementById("layoutOrderBy");
      
      // Gap slider (single unified gap) — sends adjust_layout for live updates
      const gapSlider = document.getElementById("layoutGap");
      const gapVal = document.getElementById("layoutGapVal");
      const transposeEl = document.getElementById("layoutTranspose");
      
      if (gapSlider) {
        setRangeFill(gapSlider);
        gapSlider.addEventListener("input", () => { 
          setRangeFill(gapSlider); 
          gapVal.textContent = parseFloat(gapSlider.value).toFixed(2);
          // Live adjustment (only if layout has been computed)
          window.parent.postMessage({
            type: "adjust_layout",
            gap: parseFloat(gapSlider.value),
            transpose: transposeEl ? transposeEl.checked : false
          }, "*");
        });
      }
      
      // Transpose checkbox — live adjustment
      if (transposeEl) {
        transposeEl.addEventListener("change", () => {
          window.parent.postMessage({
            type: "adjust_layout",
            gap: gapSlider ? parseFloat(gapSlider.value) : 2.0,
            transpose: transposeEl.checked
          }, "*");
        });
      }
      
      // Override computeLayoutBtn click — gather params and send to parent
      const computeBtn = document.getElementById("computeLayoutBtn");
      if (computeBtn) {
        computeBtn.removeAttribute("data-for");
        computeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const transposeEl = document.getElementById("layoutTranspose");
          const groupByEl = document.getElementById("layoutGroupBy");
          const sortByEl = document.getElementById("layoutSortBy");
          const orderByEl = document.getElementById("layoutOrderBy");
          const gapEl = document.getElementById("layoutGap");
          const ncolsEl = document.getElementById("layoutGroupCols");
          
          const params = {
            type: "compute_layout",
            sample_id: sampleId,
            group_by: groupByEl ? groupByEl.value.trim() : "",
            group_ncols: ncolsEl ? parseInt(ncolsEl.value) || 2 : 2,
            columns: sortByEl ? sortByEl.value.trim() : "",
            rows: orderByEl ? orderByEl.value.trim() : "",
            gap: gapEl ? parseFloat(gapEl.value) : 2.0,
            transpose: transposeEl ? transposeEl.checked : false,
          };
          console.log("[Layout] Compute params:", params);
          window.parent.postMessage(params, "*");
        });
      }
      
      // Save layout button — saves to JS memory (instant, no Python)
      const saveBtn = document.getElementById("saveLayoutBtn");
      if (saveBtn) {
        saveBtn.removeAttribute("data-for");
        saveBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const count = savedLayouts.length + 1;
          const defaultName = "Layout_" + count;
          const name = prompt("Layout name:", defaultName);
          if (!name || !name.trim()) return;
          window.parent.postMessage({
            type: "save_layout_request",
            name: name.trim()
          }, "*");
        });
      }
      
      // Label toggle
      const labelToggle = document.getElementById("layoutLabelToggle");
      if (labelToggle) {
        labelToggle.addEventListener("change", () => {
          window.parent.postMessage({
            type: "toggle_sample_labels",
            show: labelToggle.checked
          }, "*");
        });
      }
      
      // Saved layouts list + active tracking
      let savedLayouts = [...existingLayouts];
      let activeLayoutName = null;
      
      function renderSavedLayouts() {
        const container = document.getElementById("layoutSavedItems");
        if (!container) return;
        if (savedLayouts.length === 0) {
          container.innerHTML = '<div style="font-size:10px;color:#bbb;padding:2px 0;">No saved layouts</div>';
          return;
        }
        let html = '';
        savedLayouts.forEach(name => {
          const isActive = (name === activeLayoutName);
          html += `<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px;padding:4px 8px;
                    background:${isActive ? 'rgba(141,236,245,0.15)' : 'rgba(0,0,0,.03)'};
                    border:1px solid ${isActive ? 'rgba(141,236,245,0.5)' : 'rgba(0,0,0,.08)'};
                    border-radius:6px;cursor:pointer;transition:all 0.15s;"
                    class="saved-layout-row" data-name="${name}">
            <span style="flex:1;font-size:11px;font-weight:${isActive ? '700' : '500'};
                  color:${isActive ? '#333' : '#555'};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                  title="Click to switch">${name}</span>
            <button style="background:none;border:1px solid rgba(0,0,0,.1);border-radius:4px;cursor:pointer;
                    font-size:9px;color:#555;padding:1px 4px;line-height:1.2;white-space:nowrap;"
                    class="layout-obsm-btn" data-name="${name}" title="Save to adata.obsm">Save to obsm</button>
            <button style="background:none;border:none;cursor:pointer;font-size:13px;color:#c00;padding:0 2px;line-height:1;"
                    class="layout-delete-btn" data-name="${name}" title="Delete">&times;</button>
          </div>`;
        });
        container.innerHTML = html;
        
        // Click row = switch to this layout
        container.querySelectorAll(".saved-layout-row").forEach(row => {
          row.addEventListener("click", (e) => {
            if (e.target.classList.contains("layout-delete-btn")) return;
            if (e.target.classList.contains("layout-obsm-btn")) return;
            const n = row.dataset.name;
            activeLayoutName = n;
            document.querySelectorAll(".embedding-btn").forEach(b => b.classList.remove("active"));
            renderSavedLayouts();
            window.parent.postMessage({ type: "switch_to_saved_layout", name: n }, "*");
          });
        });
        
        // Save to obsm button
        container.querySelectorAll(".layout-obsm-btn").forEach(btn => {
          btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const n = btn.dataset.name;
            btn.textContent = "...";
            btn.disabled = true;
            window.parent.postMessage({ type: "save_to_obsm", name: n }, "*");
          });
        });
        
        // Delete button
        container.querySelectorAll(".layout-delete-btn").forEach(btn => {
          btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const n = btn.dataset.name;
            if (!confirm("Delete layout '" + n + "'?")) return;
            window.parent.postMessage({ type: "delete_saved_layout", name: n }, "*");
          });
        });
      }
      
      renderSavedLayouts();
      
      // Listen for layout events from parent
      window.addEventListener("message", (event) => {
        if (event.data && event.data.type === "layout_saved") {
          const n = event.data.name;
          if (!savedLayouts.includes(n)) savedLayouts.push(n);
          activeLayoutName = n;
          renderSavedLayouts();
        }
        if (event.data && event.data.type === "layout_deleted") {
          savedLayouts = savedLayouts.filter(x => x !== event.data.name);
          if (activeLayoutName === event.data.name) activeLayoutName = null;
          renderSavedLayouts();
        }
        if (event.data && event.data.type === "layout_applied") {
          document.querySelectorAll(".embedding-btn").forEach(b => b.classList.remove("active"));
          if (event.data.layout_name) {
            activeLayoutName = event.data.layout_name;
          }
          renderSavedLayouts();
        }
        // obsm save confirmation
        if (event.data && event.data.type === "layout_obsm_saved") {
          const btns = document.querySelectorAll(".layout-obsm-btn");
          btns.forEach(b => { b.textContent = "obsm"; b.disabled = false; });
        }
      });
      
      // When user clicks Spatial/UMAP/PCA, deactivate layout
      document.querySelectorAll(".embedding-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          activeLayoutName = null;
          renderSavedLayouts();
        });
      });
      
      // Store iframe ID for message passing
      window._iframeId = window._iframeId || null;
    })();

    // Obs opacity
    const obsOpacity = document.getElementById("obsOpacity");
    const obsOpacityVal = document.getElementById("obsOpacityVal");
    setRangeFill(obsOpacity);
    obsOpacity.addEventListener("input", () => {
      setRangeFill(obsOpacity);
      const v = parseFloat(obsOpacity.value);
      STATE.obs.opacity = v;
      obsOpacityVal.textContent = v.toFixed(2);
      postUIState();
    });

    // GEX opacity
    const gexOpacity = document.getElementById("gexOpacity");
    const gexOpacityVal = document.getElementById("gexOpacityVal");
    setRangeFill(gexOpacity);
    gexOpacity.addEventListener("input", () => {
      setRangeFill(gexOpacity);
      const v = parseFloat(gexOpacity.value);
      STATE.gex.opacity = v;
      gexOpacityVal.textContent = v.toFixed(2);
      postUIState();
    });

    // ----------------------------
    // Colormap Picker
    // ----------------------------
    const COLORMAPS = {
      viridis:[[0,[68,1,84]],[.13,[71,44,122]],[.25,[59,81,139]],[.38,[44,113,142]],[.5,[33,144,141]],[.63,[39,173,129]],[.75,[92,200,99]],[.88,[170,220,50]],[1,[253,231,37]]],
      plasma:[[0,[13,8,135]],[.5,[219,92,104]],[1,[240,249,33]]],
      inferno:[[0,[0,0,4]],[.5,[149,39,96]],[1,[252,255,164]]],
      magma:[[0,[0,0,4]],[.5,[149,41,121]],[1,[252,253,191]]],
      cividis:[[0,[0,32,77]],[.5,[124,123,120]],[1,[253,234,69]]],
      coolwarm:[[0,[59,76,192]],[.5,[221,221,221]],[1,[180,4,38]]],
      hot:[[0,[11,0,0]],[.33,[230,0,0]],[.66,[255,210,0]],[1,[255,255,255]]],
      YlOrRd:[[0,[255,255,204]],[.5,[253,141,60]],[1,[128,0,38]]],
      YlGnBu:[[0,[255,255,217]],[.5,[65,182,196]],[1,[8,29,88]]],
      Greens:[[0,[247,252,245]],[.5,[116,196,118]],[1,[0,68,27]]],
      Blues:[[0,[247,251,255]],[.5,[66,146,198]],[1,[8,48,107]]],
      Reds:[[0,[255,245,240]],[.5,[251,106,74]],[1,[103,0,13]]],
      Greys:[[0,[255,255,255]],[.5,[150,150,150]],[1,[0,0,0]]],
      turbo:[[0,[48,18,59]],[.25,[32,185,175]],[.5,[203,235,26]],[.75,[248,108,24]],[1,[122,4,3]]],
      Spectral:[[0,[158,1,66]],[.5,[255,255,191]],[1,[94,79,162]]],
      RdBu:[[0,[103,0,31]],[.5,[247,247,247]],[1,[5,48,97]]],
      jet:[[0,[0,0,128]],[.25,[0,255,255]],[.5,[0,255,0]],[.75,[255,255,0]],[1,[128,0,0]]],
      rainbow:[[0,[128,0,255]],[.25,[0,128,255]],[.5,[0,255,0]],[.75,[255,255,0]],[1,[255,0,0]]]
    };
    STATE.gex.colormap = "viridis";
    STATE.gex.vmax = 0;

    function cmapInterp(stops, t) {
      t = Math.max(0, Math.min(1, t));
      for (let i = 0; i < stops.length-1; i++) {
        const a = stops[i], b = stops[i+1];
        if (t >= a[0] && t <= b[0]) {
          const u = (t-a[0])/(b[0]-a[0]||1);
          return [Math.round(a[1][0]+u*(b[1][0]-a[1][0])), Math.round(a[1][1]+u*(b[1][1]-a[1][1])), Math.round(a[1][2]+u*(b[1][2]-a[1][2]))];
        }
      }
      return stops[stops.length-1][1];
    }
    function drawCmapBar(canvas, name) {
      if (!canvas || !canvas.parentElement) return;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext("2d");
      const stops = COLORMAPS[name] || COLORMAPS.viridis;
      for (let x = 0; x < w; x++) {
        const c = cmapInterp(stops, x/(w-1||1));
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
        ctx.fillRect(x, 0, 1, h);
      }
    }
    function updateCmapLabels() {
      const minEl = document.getElementById("cmapMinLabel");
      const maxEl = document.getElementById("cmapMaxLabel");
      if (minEl) minEl.textContent = "0";
      if (maxEl) maxEl.textContent = STATE.gex.vmax ? STATE.gex.vmax.toFixed(2) : "max";
    }
    requestAnimationFrame(() => { drawCmapBar(document.getElementById("cmapPreview"), "viridis"); updateCmapLabels(); });

    document.getElementById("cmapSelected").addEventListener("click", () => {
      const dd = document.getElementById("cmapDropdown");
      const vis = dd.style.display === "block";
      dd.style.display = vis ? "none" : "block";
      if (!vis) { renderCmapList(""); const s = document.getElementById("cmapSearch"); s.value = ""; s.focus(); }
    });
    document.addEventListener("click", (e) => {
      const dd = document.getElementById("cmapDropdown"), sel = document.getElementById("cmapSelected");
      if (dd && sel && !dd.contains(e.target) && !sel.contains(e.target)) dd.style.display = "none";
    });
    document.getElementById("cmapSearch").addEventListener("input", (e) => renderCmapList(e.target.value.toLowerCase()));
    document.getElementById("cmapSearch").addEventListener("click", (e) => e.stopPropagation());

    function renderCmapList(filter) {
      const list = document.getElementById("cmapList"); list.innerHTML = "";
      Object.keys(COLORMAPS).forEach(name => {
        if (filter && !name.toLowerCase().includes(filter)) return;
        const row = document.createElement("div");
        row.className = "cmap-row" + (name === STATE.gex.colormap ? " active" : "");
        const label = document.createElement("span"); label.className = "cmap-row-name"; label.textContent = name;
        const bar = document.createElement("canvas"); bar.className = "cmap-row-bar"; bar.height = 16;
        row.appendChild(label); row.appendChild(bar); list.appendChild(row);
        requestAnimationFrame(() => drawCmapBar(bar, name));
        row.addEventListener("click", (e) => {
          e.stopPropagation();
          STATE.gex.colormap = name;
          document.getElementById("cmapName").textContent = name;
          drawCmapBar(document.getElementById("cmapPreview"), name);
          document.getElementById("cmapDropdown").style.display = "none";
          postUIState();
        });
      });
    }

    window.addEventListener("message", (event) => {
      if (event.data && event.data.type === "gex_loaded") {
        const gene = event.data.gene;
        const vmax = event.data.vmax || 0;
        if (!gene) return;
        if (!STATE.gex.order.includes(gene)) STATE.gex.order.push(gene);
        STATE.gex.genes[gene] = true;
        STATE.gex.active = gene;
        STATE.gex.vmax = vmax;
        updateCmapLabels();
        const inp = document.getElementById("gexInput");
        if (inp) inp.value = "";
        renderGeneChips();
        // Hide GEX loading overlay
        window.parent.postMessage({ type: "hide_gex_loading" }, "*");
      }
      if (event.data && event.data.type === "gex_cache_miss") {
        const gene = event.data.gene;
        if (!gene) return;
        const inp = document.getElementById("gexInput");
        if (inp) inp.value = gene;
        // Show GEX loading overlay
        window.parent.postMessage({ type: "show_gex_loading" }, "*");
        document.getElementById("geneBtn").click();
      }
    });

    // ----------------------------
    // CSS-to-hex helper
    // ----------------------------
    function cssToHex(color) {
      if (!color) return "#999999";
      if (color.startsWith("#")) {
        if (color.length === 4) return "#"+color[1]+color[1]+color[2]+color[2]+color[3]+color[3];
        return color.slice(0,7);
      }
      const tmp = document.createElement("div"); tmp.style.color = color;
      document.body.appendChild(tmp);
      const comp = getComputedStyle(tmp).color; document.body.removeChild(tmp);
      const m = comp.match(/(\d+)/g);
      if (m && m.length >= 3) return "#"+((1<<24)+(+m[0]<<16)+(+m[1]<<8)+ +m[2]).toString(16).slice(1);
      return "#999999";
    }

    // ----------------------------
    // Legend rendering + toggles + color picker
    // ----------------------------
    function renderLegend() {
      const legend = document.getElementById("legend");

      if (!STATE.obs || STATE.obs.mode !== "categorical" || !STATE.obs.categories) {
        legend.classList.remove("visible");
        legend.innerHTML = "";
        return;
      }

      legend.classList.add("visible");

      const cats = STATE.obs.categories || [];
      const pal = STATE.obs.colors || [];
      if (!STATE.obs.enabled || STATE.obs.enabled.length !== cats.length) {
        STATE.obs.enabled = cats.map(() => true);
      }

      const header = document.createElement("div");
      header.className = "legend-header";

      const title = document.createElement("div");
      title.className = "legend-title";
      title.textContent = `${STATE.obs.column} (click to toggle)`;
      header.appendChild(title);

      const actions = document.createElement("div");
      actions.className = "legend-actions";

      const allOff = document.createElement("button");
      allOff.className = "legend-btn";
      allOff.textContent = "All off";
      allOff.onclick = () => {
        STATE.obs.enabled = cats.map(() => false);
        renderLegend();
        postUIState();
      };

      const allOn = document.createElement("button");
      allOn.className = "legend-btn";
      allOn.textContent = "All on";
      allOn.onclick = () => {
        STATE.obs.enabled = cats.map(() => true);
        renderLegend();
        postUIState();
      };

      actions.appendChild(allOff);
      actions.appendChild(allOn);
      header.appendChild(actions);

      const body = document.createElement("div");
      body.className = "legend-body";

      cats.forEach((c, i) => {
        const row = document.createElement("div");
        row.className = "legend-row" + (STATE.obs.enabled[i] ? "" : " off");
        row.onclick = () => { STATE.obs.enabled[i] = !STATE.obs.enabled[i]; renderLegend(); postUIState(); };
        const label = document.createElement("div"); label.className = "legend-label"; label.textContent = String(c);
        const curColor = pal[i] ? pal[i] : "hsl(" + ((i * 137.508) % 360) + ",55%,55%)";
        const dot = document.createElement("div"); dot.className = "legend-dot"; dot.style.background = curColor;
        const ci = document.createElement("input"); ci.type = "color"; ci.value = cssToHex(curColor);
        ci.addEventListener("click", (e) => e.stopPropagation());
        ci.addEventListener("input", (e) => {
          e.stopPropagation(); dot.style.background = e.target.value;
          if (!STATE.obs.colors) STATE.obs.colors = cats.map((_,j) => pal[j] || "hsl("+((j*137.508)%360)+",55%,55%)");
          STATE.obs.colors[i] = e.target.value; postUIState();
        });
        dot.appendChild(ci);
        row.appendChild(label); row.appendChild(dot); body.appendChild(row);
      });

      legend.innerHTML = "";
      legend.appendChild(header);
      legend.appendChild(body);
    }

    // ----------------------------
    // Gene chips and groups with drag-drop
    // ----------------------------
    // Initialize gene group state
    if (!STATE.gex.groups) STATE.gex.groups = {}; // { groupName: { genes: [], expanded: true } }
    if (!STATE.gex.groupOrder) STATE.gex.groupOrder = [];
    
    let draggedGene = null;
    let draggedFromGroup = null; // null = ungrouped, or group name
    
    function getUngroupedGenes() {
      // Genes not in any group
      const grouped = new Set();
      Object.values(STATE.gex.groups).forEach(g => g.genes.forEach(gene => grouped.add(gene)));
      return STATE.gex.order.filter(g => !grouped.has(g));
    }
    
    function createGeneChip(gene, fromGroup = null) {
      const chip = document.createElement("div");
      const isActive = STATE.gex.active === gene;
      chip.className = "gene-chip" + (isActive ? " active" : "");
      chip.draggable = true;
      chip.dataset.gene = gene;
      chip.dataset.fromGroup = fromGroup || "";
      
      chip.ondragstart = (e) => {
        draggedGene = gene;
        draggedFromGroup = fromGroup;
        chip.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      };
      chip.ondragend = () => {
        chip.classList.remove("dragging");
        draggedGene = null;
        draggedFromGroup = null;
      };
      
      chip.onclick = (e) => {
        if (e.target.classList.contains("gene-chip-remove")) return;
        if (STATE.gex.active === gene) {
          STATE.gex.active = null;
          STATE.gex.activeGroup = null;
          renderGeneChips();
          postUIState();
        } else {
          STATE.gex.active = gene;
          STATE.gex.activeGroup = null;
          renderGeneChips();
          window.parent.postMessage({ type: "switch_gene", gene: gene }, "*");
        }
      };
      
      const txt = document.createElement("span");
      txt.textContent = gene;
      chip.appendChild(txt);
      
      const rm = document.createElement("div");
      rm.className = "gene-chip-remove";
      rm.textContent = "×";
      rm.onclick = (e) => {
        e.stopPropagation();
        // Remove from group if in one
        if (fromGroup && STATE.gex.groups[fromGroup]) {
          STATE.gex.groups[fromGroup].genes = STATE.gex.groups[fromGroup].genes.filter(g => g !== gene);
        }
        // Remove from order and genes
        delete STATE.gex.genes[gene];
        STATE.gex.order = STATE.gex.order.filter(g => g !== gene);
        if (STATE.gex.active === gene) STATE.gex.active = null;
        renderGeneChips();
        postUIState();
      };
      chip.appendChild(rm);
      
      return chip;
    }
    
    function renderGeneGroup(groupName, container) {
      const group = STATE.gex.groups[groupName];
      if (!group) return;
      
      const folder = document.createElement("div");
      folder.className = "gene-group-folder";
      folder.dataset.group = groupName;
      
      // Drop zone for this group
      folder.ondragover = (e) => { e.preventDefault(); folder.classList.add("drag-over"); };
      folder.ondragleave = () => folder.classList.remove("drag-over");
      folder.ondrop = (e) => {
        e.preventDefault();
        folder.classList.remove("drag-over");
        if (draggedGene && draggedFromGroup !== groupName) {
          // Remove from old group
          if (draggedFromGroup && STATE.gex.groups[draggedFromGroup]) {
            STATE.gex.groups[draggedFromGroup].genes = STATE.gex.groups[draggedFromGroup].genes.filter(g => g !== draggedGene);
          }
          // Add to this group
          if (!group.genes.includes(draggedGene)) {
            group.genes.push(draggedGene);
          }
          renderGeneChips();
          postUIState();
        }
      };
      
      // Header
      const header = document.createElement("div");
      const isActive = STATE.gex.activeGroup === groupName;
      header.className = "gene-group-header" + (isActive ? " active" : "");
      
      const arrow = document.createElement("span");
      arrow.className = "gene-group-arrow" + (group.expanded !== false ? " expanded" : "");
      arrow.textContent = "▶";
      arrow.onclick = (e) => {
        e.stopPropagation();
        group.expanded = !group.expanded;
        renderGeneChips();
      };
      header.appendChild(arrow);
      
      const nameInput = document.createElement("input");
      nameInput.className = "gene-group-name";
      nameInput.value = groupName;
      nameInput.onclick = (e) => e.stopPropagation();
      nameInput.onblur = () => {
        const newName = nameInput.value.trim();
        if (newName && newName !== groupName && !STATE.gex.groups[newName]) {
          STATE.gex.groups[newName] = STATE.gex.groups[groupName];
          delete STATE.gex.groups[groupName];
          STATE.gex.groupOrder = STATE.gex.groupOrder.map(n => n === groupName ? newName : n);
          if (STATE.gex.activeGroup === groupName) STATE.gex.activeGroup = newName;
          renderGeneChips();
          postUIState();
        }
      };
      nameInput.onkeydown = (e) => { if (e.key === "Enter") nameInput.blur(); };
      header.appendChild(nameInput);
      
      const count = document.createElement("span");
      count.className = "gene-group-count";
      count.textContent = `(${group.genes.length})`;
      header.appendChild(count);
      
      // Click header to activate group
      header.onclick = () => {
        if (STATE.gex.activeGroup === groupName) {
          STATE.gex.activeGroup = null;
          renderGeneChips();
          postUIState();
        } else if (group.genes.length > 0) {
          STATE.gex.activeGroup = groupName;
          STATE.gex.active = null;
          renderGeneChips();
          window.parent.postMessage({ 
            type: "gene_group_expression", 
            groupName: groupName,
            genes: group.genes,
            method: "geometric_mean"
          }, "*");
        }
      };
      
      const rm = document.createElement("div");
      rm.className = "gene-group-remove";
      rm.textContent = "×";
      rm.title = "Delete group";
      rm.onclick = (e) => {
        e.stopPropagation();
        delete STATE.gex.groups[groupName];
        STATE.gex.groupOrder = STATE.gex.groupOrder.filter(n => n !== groupName);
        if (STATE.gex.activeGroup === groupName) STATE.gex.activeGroup = null;
        renderGeneChips();
        postUIState();
      };
      header.appendChild(rm);
      
      folder.appendChild(header);
      
      // Content (gene chips)
      const content = document.createElement("div");
      content.className = "gene-group-content" + (group.expanded !== false ? " expanded" : "");
      
      group.genes.forEach(gene => {
        if (STATE.gex.order.includes(gene)) {
          content.appendChild(createGeneChip(gene, groupName));
        }
      });
      
      // Empty state
      if (group.genes.length === 0) {
        const empty = document.createElement("div");
        empty.style.cssText = "font-size:10px;color:#999;padding:4px;";
        empty.textContent = "Drag genes here";
        content.appendChild(empty);
      }
      
      folder.appendChild(content);
      container.appendChild(folder);
    }
    
    function renderGeneChips() {
      const groupContainer = document.getElementById("geneGroupContainer");
      const chipContainer = document.getElementById("geneChipContainer");
      const chipsBox = document.getElementById("geneChips");
      
      // Render groups
      groupContainer.innerHTML = "";
      STATE.gex.groupOrder.forEach(name => renderGeneGroup(name, groupContainer));
      
      // Render ungrouped genes
      chipContainer.innerHTML = "";
      const ungrouped = getUngroupedGenes();
      
      if (ungrouped.length === 0 && STATE.gex.groupOrder.length === 0) {
        chipsBox.style.display = "none";
      } else {
        chipsBox.style.display = "block";
        ungrouped.forEach(gene => chipContainer.appendChild(createGeneChip(gene, null)));
      }
      
      // Make ungrouped area a drop target
      chipContainer.ondragover = (e) => { e.preventDefault(); chipContainer.classList.add("drag-over"); };
      chipContainer.ondragleave = () => chipContainer.classList.remove("drag-over");
      chipContainer.ondrop = (e) => {
        e.preventDefault();
        chipContainer.classList.remove("drag-over");
        if (draggedGene && draggedFromGroup) {
          // Remove from old group
          if (STATE.gex.groups[draggedFromGroup]) {
            STATE.gex.groups[draggedFromGroup].genes = STATE.gex.groups[draggedFromGroup].genes.filter(g => g !== draggedGene);
          }
          renderGeneChips();
          postUIState();
        }
      };
    }
    
    // Create group button
    const createGroupBtn = document.getElementById("createGeneGroupBtn");
    if (createGroupBtn) {
      createGroupBtn.onclick = () => {
        const name = prompt("Enter group name:", "Group " + (STATE.gex.groupOrder.length + 1));
        if (!name || !name.trim()) return;
        const groupName = name.trim();
        if (STATE.gex.groups[groupName]) {
          alert("Group already exists");
          return;
        }
        STATE.gex.groups[groupName] = { genes: [], expanded: true };
        STATE.gex.groupOrder.push(groupName);
        renderGeneChips();
        postUIState();
      };
    }

    // ----------------------------
    // Selection chips with drag-drop groups (like gene groups)
    // ----------------------------
    // Initialize selection group state
    if (!STATE.selection.groups) STATE.selection.groups = {}; // { groupName: { selections: [], expanded: true } }
    if (!STATE.selection.groupOrder) STATE.selection.groupOrder = [];
    
    let draggedSel = null;
    let draggedSelFromGroup = null;
    
    function getUngroupedSelections() {
      const grouped = new Set();
      Object.values(STATE.selection.groups).forEach(g => g.selections.forEach(s => grouped.add(s)));
      return STATE.selection.order.filter(s => !grouped.has(s) && STATE.selection.selections[s]);
    }
    
    function createSelectionChip(name, fromGroup = null) {
      const sel = STATE.selection.selections[name];
      if (!sel) return null;
      
      const chip = document.createElement("div");
      const isActive = STATE.selection.active === name;
      chip.className = "sel-chip" + (isActive ? " active" : "");
      chip.draggable = true;
      chip.dataset.selection = name;
      chip.dataset.fromGroup = fromGroup || "";
      
      chip.ondragstart = (e) => {
        draggedSel = name;
        draggedSelFromGroup = fromGroup;
        chip.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      };
      chip.ondragend = () => {
        chip.classList.remove("dragging");
        draggedSel = null;
        draggedSelFromGroup = null;
      };
      
      chip.onclick = (e) => {
        if (e.target.classList.contains("sel-chip-remove")) return;
        if (e.target.classList.contains("sel-chip-name")) return;
        if (STATE.selection.active === name) {
          STATE.selection.active = null;
        } else {
          STATE.selection.active = name;
          STATE.selection.activeGroup = null;
        }
        renderSelectionChips();
        postUIState();
      };
      
      // Editable name
      const nameInput = document.createElement("input");
      nameInput.className = "sel-chip-name";
      nameInput.value = name;
      nameInput.onclick = (e) => e.stopPropagation();
      nameInput.onblur = () => {
        const newName = nameInput.value.trim();
        if (newName && newName !== name && !STATE.selection.selections[newName]) {
          // Rename selection
          STATE.selection.selections[newName] = STATE.selection.selections[name];
          delete STATE.selection.selections[name];
          STATE.selection.order = STATE.selection.order.map(n => n === name ? newName : n);
          // Update in groups
          Object.values(STATE.selection.groups).forEach(g => {
            g.selections = g.selections.map(s => s === name ? newName : s);
          });
          if (STATE.selection.active === name) STATE.selection.active = newName;
          renderSelectionChips();
          postUIState();
        }
      };
      nameInput.onkeydown = (e) => { if (e.key === "Enter") nameInput.blur(); };
      chip.appendChild(nameInput);
      
      // Cell count
      const count = document.createElement("span");
      count.className = "sel-chip-count";
      count.textContent = `(${sel.indices ? sel.indices.length : 0})`;
      chip.appendChild(count);
      
      // Remove button
      const rm = document.createElement("div");
      rm.className = "sel-chip-remove";
      rm.textContent = "×";
      rm.onclick = (e) => {
        e.stopPropagation();
        if (fromGroup && STATE.selection.groups[fromGroup]) {
          STATE.selection.groups[fromGroup].selections = STATE.selection.groups[fromGroup].selections.filter(s => s !== name);
        }
        delete STATE.selection.selections[name];
        STATE.selection.order = STATE.selection.order.filter(s => s !== name);
        if (STATE.selection.active === name) STATE.selection.active = null;
        renderSelectionChips();
        postUIState();
      };
      chip.appendChild(rm);
      
      return chip;
    }
    
    function renderSelectionGroup(groupName, container) {
      const group = STATE.selection.groups[groupName];
      if (!group) return;
      
      const folder = document.createElement("div");
      folder.className = "sel-group-folder";
      folder.dataset.group = groupName;
      
      // Drop zone
      folder.ondragover = (e) => { e.preventDefault(); folder.classList.add("drag-over"); };
      folder.ondragleave = () => folder.classList.remove("drag-over");
      folder.ondrop = (e) => {
        e.preventDefault();
        folder.classList.remove("drag-over");
        if (draggedSel && draggedSelFromGroup !== groupName) {
          if (draggedSelFromGroup && STATE.selection.groups[draggedSelFromGroup]) {
            STATE.selection.groups[draggedSelFromGroup].selections = 
              STATE.selection.groups[draggedSelFromGroup].selections.filter(s => s !== draggedSel);
          }
          if (!group.selections.includes(draggedSel)) {
            group.selections.push(draggedSel);
          }
          renderSelectionChips();
          postUIState();
        }
      };
      
      // Header
      const header = document.createElement("div");
      const isActive = STATE.selection.activeGroup === groupName;
      header.className = "sel-group-header" + (isActive ? " active" : "");
      
      const arrow = document.createElement("span");
      arrow.className = "sel-group-arrow" + (group.expanded !== false ? " expanded" : "");
      arrow.textContent = "▶";
      arrow.onclick = (e) => {
        e.stopPropagation();
        group.expanded = !group.expanded;
        renderSelectionChips();
      };
      header.appendChild(arrow);
      
      const nameInput = document.createElement("input");
      nameInput.className = "sel-group-name";
      nameInput.value = groupName;
      nameInput.onclick = (e) => e.stopPropagation();
      nameInput.onblur = () => {
        const newName = nameInput.value.trim();
        if (newName && newName !== groupName && !STATE.selection.groups[newName]) {
          STATE.selection.groups[newName] = STATE.selection.groups[groupName];
          delete STATE.selection.groups[groupName];
          STATE.selection.groupOrder = STATE.selection.groupOrder.map(n => n === groupName ? newName : n);
          if (STATE.selection.activeGroup === groupName) STATE.selection.activeGroup = newName;
          renderSelectionChips();
          postUIState();
        }
      };
      nameInput.onkeydown = (e) => { if (e.key === "Enter") nameInput.blur(); };
      header.appendChild(nameInput);
      
      const count = document.createElement("span");
      count.className = "sel-group-count";
      count.textContent = `(${group.selections.length})`;
      header.appendChild(count);
      
      // Save button - saves group as obs column
      const saveBtn = document.createElement("button");
      saveBtn.textContent = "Save";
      saveBtn.title = "Save to adata.obs as column '" + groupName + "'";
      saveBtn.style.cssText = `
        background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.4);
        border-radius:4px;padding:2px 8px;font-size:9px;font-weight:700;
        color:#166534;cursor:pointer;margin-left:auto;
      `;
      saveBtn.onclick = (e) => {
        e.stopPropagation();
        // Build mapping: cell index -> selection name
        const columnData = {};
        group.selections.forEach(selName => {
          const sel = STATE.selection.selections[selName];
          if (sel && sel.indices) {
            sel.indices.forEach(idx => {
              columnData[idx] = selName;
            });
          }
        });
        
        if (Object.keys(columnData).length === 0) {
          alert("No cells in selections to save");
          return;
        }
        
        window.parent.postMessage({
          type: "save_obs_column",
          iframeId: window.frameElement?.id || "",
          columnName: groupName,
          columnData: columnData
        }, "*");
        
        alert(`Saved "${groupName}" to adata.obs!\n${Object.keys(columnData).length} cells labeled.`);
      };
      header.appendChild(saveBtn);
      
      // Click header to activate all selections in group
      header.onclick = () => {
        if (STATE.selection.activeGroup === groupName) {
          STATE.selection.activeGroup = null;
          STATE.selection.active = null;
        } else {
          STATE.selection.activeGroup = groupName;
          STATE.selection.active = null;
        }
        renderSelectionChips();
        postUIState();
      };
      
      const rm = document.createElement("div");
      rm.className = "sel-group-remove";
      rm.textContent = "×";
      rm.onclick = (e) => {
        e.stopPropagation();
        delete STATE.selection.groups[groupName];
        STATE.selection.groupOrder = STATE.selection.groupOrder.filter(n => n !== groupName);
        if (STATE.selection.activeGroup === groupName) STATE.selection.activeGroup = null;
        renderSelectionChips();
        postUIState();
      };
      header.appendChild(rm);
      folder.appendChild(header);
      
      // Content
      const content = document.createElement("div");
      content.className = "sel-group-content" + (group.expanded !== false ? " expanded" : "");
      
      group.selections.forEach(selName => {
        const chip = createSelectionChip(selName, groupName);
        if (chip) content.appendChild(chip);
      });
      
      if (group.selections.length === 0) {
        const empty = document.createElement("div");
        empty.style.cssText = "font-size:10px;color:#999;padding:4px;";
        empty.textContent = "Drag selections here";
        content.appendChild(empty);
      }
      
      folder.appendChild(content);
      container.appendChild(folder);
    }
    
    function renderSelectionChips() {
      const groupContainer = document.getElementById("selectionGroupContainer");
      const chipContainer = document.getElementById("selectionChipContainer");
      const chipsBox = document.getElementById("selectionChips");
      
      // Render groups
      if (groupContainer) {
        groupContainer.innerHTML = "";
        STATE.selection.groupOrder.forEach(name => renderSelectionGroup(name, groupContainer));
      }
      
      // Render ungrouped selections
      if (chipContainer) {
        chipContainer.innerHTML = "";
        const ungrouped = getUngroupedSelections();
        
        if (ungrouped.length === 0 && STATE.selection.groupOrder.length === 0) {
          if (chipsBox) chipsBox.style.display = "none";
        } else {
          if (chipsBox) chipsBox.style.display = "block";
          ungrouped.forEach(name => {
            const chip = createSelectionChip(name, null);
            if (chip) chipContainer.appendChild(chip);
          });
        }
        
        // Drop zone for ungrouped
        chipContainer.ondragover = (e) => { e.preventDefault(); chipContainer.classList.add("drag-over"); };
        chipContainer.ondragleave = () => chipContainer.classList.remove("drag-over");
        chipContainer.ondrop = (e) => {
          e.preventDefault();
          chipContainer.classList.remove("drag-over");
          if (draggedSel && draggedSelFromGroup) {
            if (STATE.selection.groups[draggedSelFromGroup]) {
              STATE.selection.groups[draggedSelFromGroup].selections = 
                STATE.selection.groups[draggedSelFromGroup].selections.filter(s => s !== draggedSel);
            }
            renderSelectionChips();
            postUIState();
          }
        };
      }
    }
    
    // Create selection group button
    const createSelGroupBtn = document.getElementById("createSelGroupBtn");
    if (createSelGroupBtn) {
      createSelGroupBtn.onclick = () => {
        const name = prompt("Enter group name:", "Group " + (STATE.selection.groupOrder.length + 1));
        if (!name || !name.trim()) return;
        const groupName = name.trim();
        if (STATE.selection.groups[groupName]) {
          alert("Group already exists");
          return;
        }
        STATE.selection.groups[groupName] = { selections: [], expanded: true };
        STATE.selection.groupOrder.push(groupName);
        renderSelectionChips();
        postUIState();
      };
    }

    // ----------------------------
    // Selection tool buttons
    // ----------------------------
    const toolButtons = document.querySelectorAll(".tool-btn");
    toolButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const tool = btn.dataset.tool;
        
        // Toggle: if clicking active tool, deactivate it
        if (STATE.selection.tool === tool) {
          STATE.selection.tool = null;
          toolButtons.forEach(b => b.classList.remove("active"));
          // Tell parent to disable drawing mode
          window.parent.postMessage({ type: "selection_tool", tool: null }, "*");
        } else {
          STATE.selection.tool = tool;
          toolButtons.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          // Tell parent to enable this drawing tool
          window.parent.postMessage({ type: "selection_tool", tool: tool }, "*");
        }
      });
    });

    // ----------------------------
    // Clear buttons
    // ----------------------------
    document.getElementById("clearObsBtn").addEventListener("click", () => {
      STATE.obs.column = null;
      STATE.obs.mode = null;
      STATE.obs.values = null;
      STATE.obs.categories = null;
      STATE.obs.colors = null;
      STATE.obs.enabled = null;
      renderLegend();
      postUIState();
    });

    document.getElementById("clearGexBtn").addEventListener("click", () => {
      STATE.gex.genes = {};
      STATE.gex.order = [];
      STATE.gex.active = null;
      renderGeneChips();
      postUIState();
    });

    // Clear all selections button
    const clearSelBtn = document.getElementById("clearSelectionBtn");
    if (clearSelBtn) {
      clearSelBtn.addEventListener("click", () => {
        STATE.selection.selections = {};
        STATE.selection.order = [];
        STATE.selection.active = null;
        STATE.selection.activeGroup = null;
        STATE.selection.counter = 1;
        STATE.selection.groups = {};
        STATE.selection.groupOrder = [];
        renderSelectionChips();
        postUIState();
      });
    }
    
    // Save colors button - save current color scheme to adata.uns
    const saveColorsBtn = document.getElementById("saveColorsBtn");
    if (saveColorsBtn) {
      saveColorsBtn.addEventListener("click", () => {
        if (!STATE.obs.column || !STATE.obs.colors || !STATE.obs.categories) {
          alert("No color scheme to save. Apply a categorical column first.");
          return;
        }
        // Send save request to Python
        window.parent.postMessage({
          type: "button_click",
          iframeId: window.frameElement ? window.frameElement.id : "",
          buttonId: "saveColorsBtn",
          data: {
            column: STATE.obs.column,
            colors: STATE.obs.colors,
            categories: STATE.obs.categories
          },
          timestamp: Date.now()
        }, "*");
        
        // Visual feedback
        saveColorsBtn.textContent = "✓ Saved!";
        saveColorsBtn.style.background = "#4CAF50";
        saveColorsBtn.style.color = "white";
        setTimeout(() => {
          saveColorsBtn.textContent = "Save colors";
          saveColorsBtn.style.background = "";
          saveColorsBtn.style.color = "";
        }, 2000);
      });
    }

    // Enter-to-apply
    document.getElementById("obsInput").addEventListener("keypress", (e) => {
      if (e.key === "Enter") document.getElementById("obsBtn").click();
    });
    document.getElementById("gexInput").addEventListener("keypress", (e) => {
      if (e.key === "Enter") document.getElementById("geneBtn").click();
    });
    
    // Show GEX loading overlay when clicking gene button
    document.getElementById("geneBtn").addEventListener("click", () => {
      const gene = document.getElementById("gexInput").value.trim();
      if (gene) {
        window.parent.postMessage({ type: "show_gex_loading" }, "*");
      }
    });

    // ----------------------------
    // Embedding button active class toggle
    // ----------------------------
    const embeddingButtons = document.querySelectorAll(".embedding-btn");
    embeddingButtons.forEach(btn => {
      const originalClickHandler = btn.onclick;
      btn.addEventListener("click", () => {
        // Remove active from all embedding buttons
        embeddingButtons.forEach(b => b.classList.remove("active"));
        // Add active to clicked button
        btn.classList.add("active");
      });
    });

    // ----------------------------
    // Receive pythonResponse (from your existing bridge)
    // ----------------------------
    window.addEventListener("pythonResponse", (event) => {
      const data = event.detail || {};
      if (!data.type) return;

      if (data.type === "obs_values") {
        STATE.obs.column = data.column || null;
        STATE.obs.mode = data.mode || null;
        STATE.obs.values = data.values || null;
        STATE.obs.categories = data.categories || null;
        STATE.obs.colors = data.colors || null;

        // default: all enabled when new obs loaded
        if (STATE.obs.mode === "categorical" && Array.isArray(STATE.obs.categories)) {
          STATE.obs.enabled = STATE.obs.categories.map(() => true);
        } else {
          STATE.obs.enabled = null;
        }

        renderLegend();
        postUIState();
      }

      if (data.type === "gex_values") {
        // Binary data is decoded by parent, which sends us "gex_loaded" instead
        // Nothing to do here — parent handles the typed array writes
        return;
      }
    });

    // Listen for selection completed from parent canvas
    window.addEventListener("message", (event) => {
      if (!event.data) return;
      
      // Handle clear active selection
      if (event.data.type === "clear_active_selection") {
        STATE.selection.active = null;
        STATE.selection.activeGroup = null;
        return;
      }
      
      if (event.data.type !== "selection_completed") return;
      
      const indices = event.data.indices || [];
      if (!indices.length) return;
      
      // Check if we're editing an existing selection (active selection exists and matches tool)
      let name;
      if (STATE.selection.active && STATE.selection.selections[STATE.selection.active]) {
        // Update existing selection
        name = STATE.selection.active;
        STATE.selection.selections[name] = {
          indices: indices,
          tool: event.data.tool || STATE.selection.selections[name].tool,
          path: event.data.path || null
        };
      } else {
        // Create new selection
        name = `Selection ${STATE.selection.counter}`;
        STATE.selection.counter++;
        
        STATE.selection.selections[name] = {
          indices: indices,
          tool: event.data.tool || STATE.selection.tool,
          path: event.data.path || null
        };
        
        // Add to order
        if (!STATE.selection.order.includes(name)) STATE.selection.order.push(name);
      }
      
      // Make active
      STATE.selection.active = name;
      
      // Keep tool active (don't deactivate after drawing)
      // User can click tool button again to deactivate
      
      renderSelectionChips();
      postUIState();
    });
  </script>
</body>
</html>
"""

def load_full_embedding(data, adata=None, __sample_idx=None, **kwargs):
    """
    Load full embedding data (for UMAP/PCA - no streaming, load all cells)
    """
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
    
    # Load ALL cells (no filtering)
    indices = np.arange(len(coords))
    
    print(f"[Full Embedding] Loading {embedding}: {len(indices):,} cells")
    
    # Pack coordinates as binary
    coords_binary = base64.b64encode(coords.astype(np.float32).tobytes()).decode('ascii')
    
    return {
        "type": "full_embedding",
        "embedding": embedding,
        "indices": indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(indices)
    }


def set_embedding_spatial(data, adata=None, __sample_idx=None, **kwargs):
    """Switch to spatial embedding - JavaScript will load chunk 0 instantly"""
    return {"type": "set_embedding", "embedding": "spatial"}


def set_embedding_umap(data, adata=None, __sample_idx=None, **kwargs):
    """Switch to UMAP embedding - JavaScript will load chunk 0 instantly, then load others"""
    if adata is None or "X_umap" not in adata.obsm:
        return {"type": "error", "message": "No UMAP embedding"}
    return {"type": "set_embedding", "embedding": "umap"}


def set_embedding_pca(data, adata=None, __sample_idx=None, **kwargs):
    """Switch to PCA embedding - JavaScript will load chunk 0 instantly, then load others"""
    if adata is None or "X_pca" not in adata.obsm:
        return {"type": "error", "message": "No PCA embedding"}
    return {"type": "set_embedding", "embedding": "pca"}


def get_viewport_cells(data, adata=None, __sample_idx=None, **kwargs):
    """
    CHUNKED STREAMING: Load cells in viewport from NON-CHUNK-0 cells.
    Chunk 0 is always pre-loaded, this adds detail from other chunks when zooming.
    Max 30K cells returned at once with Gaussian center-weighting.
    """
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    # Get viewport bounds and embedding type
    embedding = data.get("embedding", "spatial")
    view_minX = data.get("viewMinX")
    view_maxX = data.get("viewMaxX")
    view_minY = data.get("viewMinY")
    view_maxY = data.get("viewMaxY")
    
    # Validate viewport bounds
    if view_minX is None or view_maxX is None or view_minY is None or view_maxY is None:
        return {"type": "error", "message": "Invalid viewport bounds"}
    
    # Get active column/gene if any
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
    
    # Calculate distance from center for all points
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)
    
    # Select cells within circular radius
    in_circle = distances <= radius
    
    # CHUNK-AWARE: Only load from NON-CHUNK-0 cells (chunk 0 is pre-loaded)
    if "__chunk__" in adata.obs.columns:
        chunk_assignments = adata.obs["__chunk__"].values
        non_chunk0 = (chunk_assignments != 0)
        in_circle = in_circle & non_chunk0
    
    candidate_indices = np.where(in_circle)[0]
    
    if len(candidate_indices) == 0:
        # No new cells to add (chunk 0 covers this area)
        return {
            "type": "viewport_cells",
            "embedding": embedding,
            "indices": [],
            "coords_binary": "",
            "coords_count": 0,
            "message": "No additional cells in viewport (chunk 0 covers this area)"
        }
    
    # Calculate normalized distance from center (for probability weighting)
    normalized_dist = distances[candidate_indices] / radius
    
    # Gaussian falloff: 1.0 at center, smoothly to 0 at edges
    sigma = 0.5
    probabilities = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
    
    # Limit to 30K cells per request
    MAX_CELLS = 30_000
    if len(candidate_indices) > MAX_CELLS:
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum()
        
        visible_indices = np.random.choice(
            candidate_indices, 
            size=MAX_CELLS, 
            replace=False,
            p=probabilities
        )
        visible_indices.sort()
    else:
        visible_indices = candidate_indices
    
    print(f"[Streaming] Viewport {embedding}: {len(visible_indices):,} cells (circular, rotation-safe)")
    
    # Pack coordinates as binary
    visible_coords = coords[visible_indices]
    coords_binary = _pack_coords_binary(visible_coords)
    
    # Get chunk assignments for visible cells
    chunks = None
    if "__chunk__" in adata.obs.columns:
        chunks = adata.obs["__chunk__"].iloc[visible_indices].tolist()
    
    # Build response
    response = {
        "type": "viewport_cells",
        "embedding": embedding,
        "indices": visible_indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(visible_indices),
        "chunks": chunks  # Include chunk info for each cell
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
            # Categorical
            if pd.api.types.is_categorical_dtype(vals):
                categories = list(vals.cat.categories)
                codes = vals.cat.codes.to_numpy()
                codes = np.where(codes < 0, 0, codes)
                values_idx = codes.tolist()
            else:
                categories = list(pd.unique(vals))
                lut = {v: i for i, v in enumerate(categories)}
                values_idx = [lut[v] for v in vals]
            
            # Get palette
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

def get_obs_column(data, adata=None, __sample_idx=None, **kwargs):
    """Return category metadata AND compressed per-cell codes"""
    import base64
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
            codes[np.isfinite(v)] = np.clip(((v[np.isfinite(v)]-vmin)/(vmax-vmin)*254+1),1,255).astype(np.uint8)
        compressed = zlib.compress(codes.tobytes(), level=6)
        b64 = base64.b64encode(compressed).decode("ascii")
        return {"type":"obs_values","column":column,"mode":"continuous",
                "categories":None,"colors":None,"codes_b64":b64,"count":len(codes)}
    # Categorical → 1-indexed codes
    if pd.api.types.is_categorical_dtype(vals):
        categories = list(vals.cat.categories)
        raw_codes = vals.cat.codes.values[idx]
        codes = (raw_codes + 1).astype(np.int16)
        codes[raw_codes < 0] = 0
    else:
        categories = list(pd.unique(vals.dropna()))
        cat_to_idx = {c: i+1 for i, c in enumerate(categories)}
        codes = np.array([cat_to_idx.get(v, 0) for v in vals.values[idx]], dtype=np.int16)
    palette = None
    key = f"{column}_colors"
    if hasattr(adata, "uns") and key in adata.uns:
        try:
            pal = list(adata.uns[key])
            if len(pal) >= len(categories): palette = pal[:len(categories)]
        except Exception: pass
    codes_u8 = np.clip(codes, 0, 255).astype(np.uint8)
    compressed = zlib.compress(codes_u8.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    return {"type":"obs_values","column":column,"mode":"categorical",
            "categories":categories,"colors":palette,"codes_b64":b64,"count":len(codes_u8)}


def get_gene_expression(data, adata=None, __sample_idx=None, **kwargs):
    import base64
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
    # NaN/negative → 0
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
    vmax = float(v.max())
    # Quantize to uint8: 0=zero expression, 1-255=expression range
    quantized = np.zeros(len(v), dtype=np.uint8)
    if vmax > 0:
        quantized[v > 0] = np.clip((v[v > 0] / vmax * 254 + 1), 1, 255).astype(np.uint8)
    # Compress: zlib + base64 (~500KB for 3.5M cells vs 30MB raw JSON)
    compressed = zlib.compress(quantized.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    return {
        "type": "gex_values",
        "gene": gene,
        "count": len(quantized),
        "vmax": round(vmax, 4),
        "values_b64": b64,
    }


def get_gene_group_expression(data, adata=None, __sample_idx=None, **kwargs):
    """Compute geometric mean expression for a group of genes."""
    import base64
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    genes = data.get("genes", [])
    method = data.get("method", "geometric_mean")
    group_name = data.get("groupName", "group")
    
    if not genes or len(genes) == 0:
        return {"type": "error", "message": "No genes specified"}
    
    # Filter to valid genes
    valid_genes = [g for g in genes if g in adata.var_names]
    if not valid_genes:
        return {"type": "error", "message": f"None of the genes found: {genes}"}
    
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        print(f"[GeneGroup] Warning: genes not found: {missing}")
    
    idx = np.asarray(__sample_idx if __sample_idx is not None else np.arange(adata.n_obs), dtype=int)
    X = adata.X
    
    # Get expression for each gene
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
    
    # Stack: shape (n_genes, n_cells)
    stacked = np.stack(gene_values, axis=0)
    
    # Geometric mean: (product of values)^(1/n)
    # For numerical stability, use exp(mean(log(x+1))) - 1
    # Add small epsilon to avoid log(0)
    epsilon = 1e-9
    log_vals = np.log(stacked + epsilon)
    combined = np.exp(np.mean(log_vals, axis=0)) - epsilon
    combined = np.maximum(combined, 0.0)  # Ensure non-negative
    
    vmax = float(combined.max())
    
    # Quantize to uint8
    quantized = np.zeros(len(combined), dtype=np.uint8)
    if vmax > 0:
        quantized[combined > 0] = np.clip((combined[combined > 0] / vmax * 254 + 1), 1, 255).astype(np.uint8)
    
    # Compress
    compressed = zlib.compress(quantized.tobytes(), level=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    
    print(f"[GeneGroup] {group_name}: geometric_mean({valid_genes}) → vmax={vmax:.4f}")
    
    return {
        "type": "gex_values",
        "gene": group_name,  # Use group name as the "gene" identifier
        "genes_in_group": valid_genes,
        "method": "geometric_mean",
        "count": len(quantized),
        "vmax": round(vmax, 4),
        "values_b64": b64,
    }

def set_embedding_spatial(data, adata=None, __sample_idx=None, **kwargs):
    return {"type": "set_embedding", "embedding": "spatial"}

def clear_plot(data, adata=None, __sample_idx=None):
    return {"type": "clear_plot"}
def clear_obs(data, adata=None, __sample_idx=None, **kwargs):
    return {"type": "clear_obs"}
def clear_gex(data, adata=None, __sample_idx=None, **kwargs):
    return {"type": "clear_gex"}

def save_obs_column(data, adata=None, __sample_idx=None, **kwargs):
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


def save_color_scheme(data, adata=None, __sample_idx=None, **kwargs):
    """Save color scheme to adata.uns['{column}_colors']"""
    if adata is None:
        return {"type": "error", "message": "No adata provided"}
    
    column = data.get("column")
    colors = data.get("colors")
    categories = data.get("categories")
    
    if not column or not colors:
        return {"type": "error", "message": "Missing column or colors data"}
    
    # Create the uns key
    uns_key = f"{column}_colors"
    
    # Store colors as a list in adata.uns
    adata.uns[uns_key] = list(colors)
    
    print(f"✓ Saved color scheme to adata.uns['{uns_key}']")
    print(f"  {len(colors)} colors for {len(categories) if categories else '?'} categories")
    
    return {"type": "success", "message": f"Colors saved to adata.uns['{uns_key}']"}


def get_chunk_cells(data, adata=None, __sample_idx=None, **kwargs):
    """
    CHUNKED: Load cells from a specific chunk with ALL embeddings (spatial, UMAP, PCA).
    This enables instant switching between embeddings on the client side.
    """
    if adata is None:
        return {"type": "error", "message": "adata not provided"}
    
    if "__chunk__" not in adata.obs.columns:
        return {"type": "error", "message": "Chunk assignments not found - run initialization first"}
    
    chunk_id = data.get("chunk", 1)
    request_id = data.get("requestId", None)  # Echo this back for stale detection
    active_column = data.get("activeColumn", None)
    active_gene = data.get("activeGene", None)
    
    print(f"[Chunk] Python processing request for chunk {chunk_id}, reqId={request_id}")
    
    # Get cells in this chunk
    chunk_mask = (adata.obs["__chunk__"].values == chunk_id)
    chunk_indices = np.where(chunk_mask)[0]
    
    if len(chunk_indices) == 0:
        return {"type": "error", "message": f"Chunk {chunk_id} is empty"}
    
    # Use compression for smaller payloads
    USE_COMPRESSION = True
    
    # Get ALL embedding coordinates for this chunk (compressed)
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
    
    # Build response with ALL embeddings
    response = {
        "type": "chunk_data",
        "chunk": chunk_id,
        "requestId": request_id,  # Echo back for stale detection
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
            # Categorical
            if pd.api.types.is_categorical_dtype(vals):
                categories = list(vals.cat.categories)
                codes = vals.cat.codes.to_numpy()
                codes = np.where(codes < 0, 0, codes)
                values_idx = codes.tolist()
            else:
                categories = list(pd.unique(adata.obs[active_column]))
                lut = {v: i for i, v in enumerate(categories)}
                values_idx = [lut.get(v, 0) for v in vals]
            
            # Get palette
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


# ============================================================
# LAYOUT: Grid-based sample arrangement
# ============================================================

def get_sample_meta(data, adata=None, __sample_idx=None, __sample_id__=None):
    """Fetch per-sample metadata for a given obs column (on-demand).
    Returns integer codes + categories if column has 1 unique value per sample."""
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
    
    # Check if column is sample-level (1 unique value per sample)
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

def compute_layout(data, adata=None, __sample_idx=None, __sample_id__=None):
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


def save_layout(data, adata=None, __sample_idx=None, __sample_id__=None, **kwargs):
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
            # Old format: full cell coordinates
            import zlib
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
    import zlib
    raw = base64.b64decode(centroids_b64)
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    
    centroids = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    print(f"[Layout] Received {centroids.shape[0]} sample centroids for '{name}'")
    
    if centroids.shape[0] != len(sample_labels):
        return {"type": "error", "message": f"Centroid/label mismatch: {centroids.shape[0]} vs {len(sample_labels)}"}
    
    # Build sample name to centroid mapping
    sample_to_centroid = {label: centroids[i] for i, label in enumerate(sample_labels)}
    
    # Get sample IDs from adata - use passed parameter or try common names
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
    
    # Reconstruct full cell positions: cell_layout = cell_spatial + (layout_centroid - spatial_centroid)
    layout_coords = np.zeros_like(spatial)
    for i, (sid, sp) in enumerate(zip(sample_ids, spatial)):
        if sid in sample_to_centroid:
            layout_centroid = sample_to_centroid[sid]
            spatial_centroid = sample_spatial_centroids[sid]
            offset = layout_centroid - spatial_centroid
            layout_coords[i] = sp + offset
        else:
            # Sample not in layout, keep original position
            layout_coords[i] = sp
    
    key = name if name.startswith("X_") else f"X_{name}"
    adata.obsm[key] = layout_coords
    print(f"[Layout] Saved '{name}' to adata.obsm['{key}'] shape={layout_coords.shape}")
    return {"type": "layout_obsm_saved", "name": name, "key": key}




def delete_layout(data, adata=None, __sample_idx=None, **kwargs):
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


def load_layout(data, adata=None, __sample_idx=None, __sample_id__=None, **kwargs):
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

    
def create_adata_interface(adata, figsize=(900, 600), debug=False, html_template: str = "", sample_id: str = None, chunk_size: int = 500_000, max_result_size: int = 30_000_000):
    # Unpack figsize tuple
    width, height = figsize
    
    # Detect categorical/string columns for layout dropdowns
    cat_columns = []
    for col in adata.obs.columns:
        if col.startswith("__"): continue
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
    
    initial_data = {
        "obs_columns": list(adata.obs.columns),
        "var_names": list(adata.var_names[:1000]),
        "cat_columns": cat_columns,
        "sample_id": sample_id,
        "sample_info": sample_info,
        "existing_layouts": existing_layouts,
    }

    return link_buttons_to_python(
        html_template,
        button_callbacks={
            "obsBtn": get_obs_column,
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
            
            # Layout
            "computeLayoutBtn": lambda data, adata=None, __sample_idx=None: compute_layout(data, adata=adata, __sample_id__=sample_id),
            "saveLayoutBtn": lambda data, adata=None, __sample_idx=None: save_layout(data, adata=adata, __sample_id__=sample_id),
            "obsmBtn": lambda data, adata=None, __sample_idx=None: save_layout(data, adata=adata, __sample_id__=sample_id),
            "deleteLayoutBtn": delete_layout,
            "loadLayoutBtn": lambda data, adata=None, __sample_idx=None: load_layout(data, adata=adata, __sample_id__=sample_id),
            "sampleMetaBtn": lambda data, adata=None, __sample_idx=None: get_sample_meta(data, adata=adata, __sample_id__=sample_id),
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
        chunk_size=chunk_size,
        max_result_size=max_result_size,
    )