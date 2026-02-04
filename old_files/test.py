# PART 1: Main body:
# STREAMING VERSION - Load only visible cells (max 100K at a time)

def _pack_coords_binary(coords_array) -> str:
    """Pack Nx2 float32 coordinates as base64."""
    import base64
    import numpy as np
    return base64.b64encode(coords_array.astype(np.float32).tobytes()).decode('ascii')

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
    max_result_size: int = 10_000_000,
    initial_data: Optional[Dict] = None,
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
            layout=widgets.Layout(width="1px", height="1px", visibility="hidden"),
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
  log("Initial data loaded:", initialData);

  buttonIds.forEach((bid) => {{
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

  window.addEventListener('message', (event) => {{
    if (event.data && event.data.type === 'python_response') {{
      log("Received Python response:", event.data.data);

      if (pendingRequest) {{
        const button = document.getElementById(pendingRequest);
        if (button) button.disabled = false;
        pendingRequest = null;
      }}

      const customEvent = new CustomEvent('pythonResponse', {{
        detail: event.data.data
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
    # STREAMING: Send metadata only (bounds, counts) - NO CELLS YET!
    # ----------------------------
    adata = callback_args.get("adata", None)

    spatial = None
    if adata is not None:
        if "spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["spatial"])
        elif "X_spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["X_spatial"])

    if spatial is None or spatial.ndim != 2 or spatial.shape[1] < 2:
        sample_idx = np.array([], dtype=int)
        embeds_js = json.dumps({"streaming": True, "spatial": None, "umap": None, "pca": None})
    else:
        n = spatial.shape[0]
        sample_idx = np.arange(n, dtype=int)  # Track all indices
        
        # Calculate bounds only (don't send coordinates yet!)
        spatial_coords = spatial[:, :2]
        bounds = {
            "minX": float(spatial_coords[:, 0].min()),
            "maxX": float(spatial_coords[:, 0].max()),
            "minY": float(spatial_coords[:, 1].min()),
            "maxY": float(spatial_coords[:, 1].max()),
            "count": n
        }
        
        # Sample 100K cells for minimap (global positioning)
        minimap_sample_size = min(100_000, n)
        minimap_indices = np.random.choice(n, size=minimap_sample_size, replace=False)
        minimap_coords = spatial_coords[minimap_indices]
        # Pack as binary for efficiency
        minimap_binary = base64.b64encode(minimap_coords.astype(np.float32).tobytes()).decode('ascii')
        
        # Optional embedding bounds
        umap_bounds = None
        pca_bounds = None
        if "X_umap" in adata.obsm:
            um = np.asarray(adata.obsm["X_umap"])
            if um.ndim == 2 and um.shape[1] >= 2 and um.shape[0] == n:
                umap_bounds = {
                    "minX": float(um[:, 0].min()),
                    "maxX": float(um[:, 0].max()),
                    "minY": float(um[:, 1].min()),
                    "maxY": float(um[:, 1].max()),
                    "count": n
                }
        if "X_pca" in adata.obsm:
            pc = np.asarray(adata.obsm["X_pca"])
            if pc.ndim == 2 and pc.shape[1] >= 2 and pc.shape[0] == n:
                pca_bounds = {
                    "minX": float(pc[:, 0].min()),
                    "maxX": float(pc[:, 0].max()),
                    "minY": float(pc[:, 1].min()),
                    "maxY": float(pc[:, 1].max()),
                    "count": n
                }
        
        embeds_js = json.dumps({
            "streaming": True,
            "spatial": bounds,
            "umap": umap_bounds,
            "pca": pca_bounds,
            "minimap": {
                "coords_binary": minimap_binary,
                "count": minimap_sample_size
            }
        })
        
        print(f"✓ Streaming mode: {n:,} total cells (will load ~100K at a time)")

    callback_args = dict(callback_args)
    callback_args["__sample_idx"] = sample_idx.tolist()

    # ----------------------------
    # Parent container: 2 panels + plot canvas (transparent background)
    # ----------------------------
    container_html = f"""
<div style="display:flex; gap:10px; width:100%; height:{int(height)}px;">
  <!-- Left panel: controls iframe -->
  <div id="left_{iframe_id}" style="width: 420px; height:100%; flex: 0 0 420px;">
    <iframe
      id="{iframe_id}"
      style="width:100%; height:100%; border:1px solid #ccc; border-radius:6px; background:white;"
    ></iframe>
  </div>

  <!-- Right panel: plot area -->
  <div id="plot_panel_{iframe_id}"
       tabindex="0"
       style="width:{int(width)}px; height:100%; flex: 0 0 {int(width)}px; border:1px solid rgba(0,0,0,0.15); border-radius:6px;
              background: transparent; position: relative; overflow:hidden; outline:none;">
    <canvas id="plot_canvas_{iframe_id}"
            style="width:100%; height:100%; display:block; background: transparent;"></canvas>

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
  </div>
</div>

<style>
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
      if (event.data.iframeId === iframeId && event.data.type === "button_click") {{
        log("Queued button click:", event.data.buttonId, "with data:", event.data.data);
        window["_requests_" + iframeId].push(event.data);
      }}
      // Listen to ui_state updates from iframe (for legend toggles, opacity, etc)
      if (event.data.type === "ui_state") {{
        log("Received ui_state update from iframe");
        const s = window["_plotState_" + iframeId];
        
        // Update obs layer
        if (s && event.data.obs) {{
          if (event.data.obs.values) {{
            // Obs data is active
            s.obs.mode = event.data.obs.mode;
            s.obs.values = event.data.obs.values;
            s.obs.colors = event.data.obs.colors;
            s.obs.categories = event.data.obs.categories;
            s.obs.enabled = event.data.obs.enabled;
            s.obs.opacity = event.data.obs.opacity || 1.0;
          }} else {{
            // No obs data - clear obs layer
            s.obs.mode = null;
            s.obs.values = null;
            s.obs.colors = null;
            s.obs.categories = null;
            s.obs.enabled = null;
          }}
        }}
        
        // Update GEX layer
        if (s && event.data.gex) {{
          if (event.data.gex.active && event.data.gex.values) {{
            // GEX is active
            s.gex.values = event.data.gex.values;
            s.gex.opacity = event.data.gex.opacity || 1.0;
          }} else {{
            // No GEX - clear GEX layer
            s.gex.values = null;
          }}
        }}
        
        // Update selection mask and outline
        if (s && event.data.selection) {{
          s.selectionIndices = event.data.selection.indices || null;
          s.selectionPath = event.data.selection.path || null;
          s.selectionTool = event.data.selection.tool || null;
        }}
        
        // Update point size
        if (s && typeof event.data.pointSize === 'number') {{
          s.pointSize = event.data.pointSize;
        }}
        
        // Update label
        if (s) {{
          const hasObs = s.obs.values != null;
          const hasGex = s.gex.values != null;
          
          if (hasGex && hasObs) {{
            s.label = "obs: " + (event.data.obs.column || "") + " + GEX: " + (event.data.gex.active || "");
          }} else if (hasGex) {{
            s.label = "GEX: " + (event.data.gex.active || "");
          }} else if (hasObs) {{
            s.label = "obs: " + (event.data.obs.column || "");
          }} else {{
            s.label = "Embedding: " + currentEmbedding;
          }}
          setLabel(s.label);
        }}
        
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
        // Queue this as a special request to be handled by Python
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

  // STREAMING: Metadata only initially
  // ----------------------------
  const METADATA = {embeds_js};
  let currentEmbedding = "spatial";
  
  // Storage for loaded cells
  const LOADED_CELLS = new Map();  // Map<index, {{x, y, obs_value, gex_value}}>
  let currentPalette = null;
  let currentCategories = null;
  let currentObsColumn = null;
  let currentGexGene = null;
  
  // Storage for full embeddings (UMAP/PCA load all cells, no streaming)
  const EMBEDS = {{}};  // Will be populated when switching to UMAP/PCA
  
  // Streaming state management
  let isRequestPending = false;
  let lastViewportRequest = null;
  let lastSuccessfulLoadTime = Date.now();
  
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
  
  console.log("[Streaming] Metadata loaded:", METADATA);
  
  // Decode binary coordinates helper
  function decodeBinaryCoords(binaryStr, count) {{
    if (!binaryStr || count === 0) return null;
    const binary = atob(binaryStr);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const floats = new Float32Array(bytes.buffer);
    return floats;  // Returns Float32Array with [x1,y1,x2,y2,...]
  }}
  
  let points = [];  // Keep for compatibility but not used

  // ----------------------------
  // Navigation state (zoom, pan, rotation)
  // ----------------------------
  let zoom = 1.0;
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
  let animationDuration = 1000; // 1 second for embedding transitions
  let startPoints = null;
  let endPoints = null;
  
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
      values: null,         // per-point continuous values
      opacity: 1.0
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
      
      // Download with user's filename
      const link = document.createElement("a");
      link.download = filename.trim() + ".png";
      link.href = canvas.toDataURL("image/png");
      link.click();
    }});
  }}

  // ----------------------------
  // Plot updater: receives Python callback payloads
  // ----------------------------
  window["updatePlot_" + iframeId] = function(payload) {{
    if (!payload || !payload.type) return;

    // STREAMING: Handle viewport cells response
    if (payload.type === "viewport_cells") {{
      console.log("[Streaming] Received viewport cells:", payload);
      
      // Clear pending flag
      isRequestPending = false;
      
      // Decode coordinates
      const coords = decodeBinaryCoords(payload.coords_binary, payload.coords_count);
      if (!coords) {{
        console.error("Failed to decode coordinates");
        return;
      }}
      
      // ADD cells (don't clear existing ones!)
      let newCellCount = 0;
      for (let i = 0; i < payload.indices.length; i++) {{
        const idx = payload.indices[i];
        // Only add if not already present
        if (!LOADED_CELLS.has(idx)) {{
          LOADED_CELLS.set(idx, {{
            x: coords[i * 2],
            y: coords[i * 2 + 1],
            obs_value: payload.obs_values ? payload.obs_values[i] : undefined,
            gex_value: payload.gex_values ? payload.gex_values[i] : undefined
          }});
          newCellCount++;
        }} else {{
          // Update color data if present
          const cell = LOADED_CELLS.get(idx);
          if (payload.obs_values) cell.obs_value = payload.obs_values[i];
          if (payload.gex_values) cell.gex_value = payload.gex_values[i];
        }}
      }}
      
      console.log(`[Streaming] Added ${{newCellCount}} new cells, total now: ${{LOADED_CELLS.size}}`);
      
      // Update health check timestamp
      lastSuccessfulLoadTime = Date.now();
      
      // Memory management: Remove cells far outside viewport (keep 2x viewport area)
      if (LOADED_CELLS.size > 100000) {{  // Prune more aggressively to prevent connection issues
        const embedMeta = METADATA[currentEmbedding];
        if (embedMeta) {{
          const rect = panel.getBoundingClientRect();
          const W = rect.width;
          const H = rect.height;
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
          
          // Calculate expanded viewport (2x size for buffer)
          const viewMinX = minX + (-offX) / scale;
          const viewMaxX = minX + (W - offX) / scale;
          const viewMinY = minY + (-offY) / scale;
          const viewMaxY = minY + (H - offY) / scale;
          const bufferX = (viewMaxX - viewMinX) * 0.5;
          const bufferY = (viewMaxY - viewMinY) * 0.5;
          
          let pruned = 0;
          for (const [idx, cell] of LOADED_CELLS.entries()) {{
            if (cell.x < viewMinX - bufferX || cell.x > viewMaxX + bufferX ||
                cell.y < viewMinY - bufferY || cell.y > viewMaxY + bufferY) {{
              LOADED_CELLS.delete(idx);
              pruned++;
            }}
          }}
          if (pruned > 0) {{
            console.log(`[Streaming] Pruned ${{pruned}} distant cells, now have ${{LOADED_CELLS.size}}`);
          }}
        }}
      }}
      
      // Store palette/categories globally
      if (payload.obs_colors) {{
        currentPalette = payload.obs_colors;
        currentCategories = payload.obs_categories;
        currentObsColumn = payload.obs_column;
      }}
      if (payload.gex_gene) {{
        currentGexGene = payload.gex_gene;
      }}
      
      draw();
      return;
    }}

    // Switch embedding (UMAP/PCA/Spatial)
    if (payload.type === "set_embedding") {{
      const name = (payload.embedding || "spatial");
      
      // Don't switch if already on this embedding
      if (currentEmbedding === name) return;
      
      // Switching TO spatial - clear non-spatial data and use streaming mode
      if (name === "spatial") {{
        console.log("[Embedding] Switching to spatial - clearing UMAP/PCA data");
        
        // IMPORTANT: Clear all loaded cells since they have wrong coordinates
        LOADED_CELLS.clear();
        
        currentEmbedding = name;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        rotation = 0;
        draw();
        
        // Request initial viewport for spatial
        if (METADATA.streaming && METADATA.spatial) {{
          const embedMeta = METADATA.spatial;
          const newView = getViewportBounds(embedMeta);
          isRequestPending = true;
          lastViewportRequest = newView;
          requestViewportCells(newView.minX, newView.maxX, newView.minY, newView.maxY);
        }}
      }}
      return;
    }}
    
    // Receive full embedding data chunks (for UMAP/PCA)
    if (payload.type === "full_embedding_chunk") {{
      const name = payload.embedding;
      
      console.log(`[Embedding] Chunk ${{payload.chunk + 1}}: ${{payload.loaded_cells.toLocaleString()}} / ${{payload.total_cells.toLocaleString()}} cells (${{Math.round(payload.loaded_cells / payload.total_cells * 100)}}%)`);
      
      // Decode coordinates
      const coords = decodeBinaryCoords(payload.coords_binary, payload.coords_count);
      if (!coords) {{
        console.error("Failed to decode embedding coordinates");
        return;
      }}
      
      // On first chunk, clear existing data (switching embeddings)
      if (payload.chunk === 0) {{
        console.log(`[Embedding] Starting fresh load of ${{name}} - clearing old data`);
        LOADED_CELLS.clear();
        
        currentEmbedding = name;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        rotation = 0;
      }}
      
      // Add cells to LOADED_CELLS (accumulate chunks!)
      for (let i = 0; i < payload.indices.length; i++) {{
        const idx = payload.indices[i];
        LOADED_CELLS.set(idx, {{
          x: coords[i * 2],
          y: coords[i * 2 + 1],
          obs_value: payload.obs_values ? payload.obs_values[i] : undefined,
          gex_value: payload.gex_values ? payload.gex_values[i] : undefined
        }});
      }}
      
      // Redraw with what we have so far
      draw();
      
      // If not final chunk, request next chunk
      if (!payload.is_final) {{
        console.log(`[Embedding] Requesting next chunk...`);
        sendButtonClick(name + "Btn", {{ chunk: payload.chunk + 1 }});
      }} else {{
        console.log(`[Embedding] ✓ Loaded all ${{payload.total_cells.toLocaleString()}} cells for ${{name}}`);
        EMBEDS[name] = true;  // Mark as fully loaded
      }}
      
      return;
    }}

    // Clear overlay (back to neutral grey)
    if (payload.type === "clear_plot") {{
      const s = window["_plotState_" + iframeId];
      if (s) {{
        // Clear obs layer
        s.obs.mode = null;
        s.obs.values = null;
        s.obs.colors = null;
        s.obs.categories = null;
        s.obs.enabled = null;
        // Clear GEX layer
        s.gex.values = null;
        s.label = "Embedding: " + currentEmbedding;
      }}
      setLabel("Embedding: " + currentEmbedding);
      draw();
      return;
    }}

    // GEX continuous - store in GEX layer
    if (payload.type === "gex_values") {{
      const s = window["_plotState_" + iframeId];
      if (s) {{
        s.gex.values = payload.values || null;
        s.label = "GEX: " + (payload.gene || "");
      }}
      setLabel("GEX: " + (payload.gene || ""));
      draw();
      return;
    }}

    // obs values - STREAMING: Update colors without clearing cells
    if (payload.type === "obs_values") {{
      console.log("[Streaming] Obs column selected, requesting color update...");
      currentObsColumn = payload.column;
      currentPalette = payload.colors;
      currentCategories = payload.categories;
      
      // Request color data for currently loaded cells
      if (LOADED_CELLS.size > 0) {{
        const indices = Array.from(LOADED_CELLS.keys());
        console.log(`[Streaming] Requesting colors for ${{indices.length}} loaded cells`);
        // We'll need a separate callback for this - for now just redraw
        draw();
      }}
      return;
    }}
  }};

  // ----------------------------
  // Plotting
  // ----------------------------
  const canvas = document.getElementById("plot_canvas_" + iframeId);
  const panel = document.getElementById("plot_panel_" + iframeId);
  const ctx = canvas.getContext("2d");
  
  const minimap = document.getElementById("minimap_" + iframeId);
  const minimapCtx = minimap.getContext("2d");
  
  // Set actual minimap canvas dimensions (not just CSS)
  minimap.width = 120;
  minimap.height = 120;

  function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}

  // Minimal viridis approximation via stops
  const VIRIDIS = [
    [0.0,  [68, 1, 84]],
    [0.13, [71, 44, 122]],
    [0.25, [59, 81, 139]],
    [0.38, [44, 113, 142]],
    [0.50, [33, 144, 141]],
    [0.63, [39, 173, 129]],
    [0.75, [92, 200, 99]],
    [0.88, [170, 220, 50]],
    [1.0,  [253, 231, 37]]
  ];

  function viridis(t) {{
    t = clamp01(t);
    for (let i = 0; i < VIRIDIS.length - 1; i++) {{
      const a = VIRIDIS[i], b = VIRIDIS[i + 1];
      if (t >= a[0] && t <= b[0]) {{
        const u = (t - a[0]) / (b[0] - a[0] || 1);
        const r = Math.round(a[1][0] + u*(b[1][0]-a[1][0]));
        const g = Math.round(a[1][1] + u*(b[1][1]-a[1][1]));
        const bl = Math.round(a[1][2] + u*(b[1][2]-a[1][2]));
        return `rgb(${{r}},${{g}},${{bl}})`;
      }}
    }}
    const c = VIRIDIS[VIRIDIS.length - 1][1];
    return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
  }}

  function fallbackCatColor(k) {{
    const h = (k * 137.508) % 360;
    return `hsl(${{h}}, 55%, 55%)`;
  }}

  // ----------------------------
  // Streaming viewport management
  // ----------------------------
  let viewportRefreshTimer = null;
  // Note: isRequestPending and lastViewportRequest declared earlier
  
  // Helper: Calculate circular viewport that contains entire view at any rotation
  function getViewportBounds(embedMeta) {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
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
    
    // Calculate center of viewport in data coordinates
    const centerDataX = minX + (W / 2 - offX) / scale;
    const centerDataY = minY + (H / 2 - offY) / scale;
    
    // Calculate radius: diagonal of viewport / 2 (so circle contains entire rectangle at any rotation)
    const viewWidthData = W / scale;
    const viewHeightData = H / scale;
    const radius = Math.sqrt(viewWidthData * viewWidthData + viewHeightData * viewHeightData) / 2;
    
    // Add 20% buffer for smooth loading
    const bufferedRadius = radius * 1.2;
    
    // Return circular bounds as a square bounding box
    return {{
      minX: centerDataX - bufferedRadius,
      maxX: centerDataX + bufferedRadius,
      minY: centerDataY - bufferedRadius,
      maxY: centerDataY + bufferedRadius
    }};
  }}
  
  // Health check: if we haven't received data in 10 seconds, refresh viewport
  setInterval(() => {{
    if (METADATA.streaming && !isRequestPending) {{
      const timeSinceLastLoad = Date.now() - lastSuccessfulLoadTime;
      // If more than 10 seconds and we have very few cells, request refresh
      if (timeSinceLastLoad > 10000 && LOADED_CELLS.size < 5000) {{
        console.log("[Health Check] Low cell count detected, refreshing viewport...");
        const embedMeta = METADATA[currentEmbedding];
        if (embedMeta) {{
          const newView = getViewportBounds(embedMeta);
          isRequestPending = true;
          lastViewportRequest = newView;
          requestViewportCells(newView.minX, newView.maxX, newView.minY, newView.maxY);
        }}
      }}
    }}
  }}, 5000);  // Check every 5 seconds
  
  // Check if viewport has changed (more sensitive for smooth streaming)
  function hasViewportChanged(oldView, newView) {{
    if (!oldView) return true;
    const overlapX = Math.min(oldView.maxX, newView.maxX) - Math.max(oldView.minX, newView.minX);
    const overlapY = Math.min(oldView.maxY, newView.maxY) - Math.max(oldView.minY, newView.minY);
    const oldAreaX = oldView.maxX - oldView.minX;
    const oldAreaY = oldView.maxY - oldView.minY;
    const overlapRatio = (overlapX * overlapY) / (oldAreaX * oldAreaY);
    return overlapRatio < 0.8;  // Refresh at 80% overlap (more frequent requests)
  }}

  function resizeCanvas() {{
    const rect = panel.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * window.devicePixelRatio));
    canvas.height = Math.max(1, Math.floor(rect.height * window.devicePixelRatio));
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    draw();
  }}

  function draw() {{
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    ctx.clearRect(0, 0, W, H);

    // Check if we have metadata
    const embedMeta = METADATA[currentEmbedding];
    if (!embedMeta || !METADATA.streaming) {{
      ctx.font = "13px ui-monospace, Menlo, Monaco, monospace";
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillText("Embedding not available.", 12, 40);
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
    
    // Calculate viewport bounds in data coordinates (accounting for zoom/pan)
    // The visible area in data coordinates
    const viewMinX = minX + (-offX) / scale;
    const viewMaxX = minX + (W - offX) / scale;
    const viewMinY = minY + (-offY) / scale;
    const viewMaxY = minY + (H - offY) / scale;
    
    // Check if we have cells loaded
    if (LOADED_CELLS.size === 0) {{
      // Only request if not already pending
      if (!isRequestPending) {{
        const newView = {{minX: viewMinX, maxX: viewMaxX, minY: viewMinY, maxY: viewMaxY}};
        if (hasViewportChanged(lastViewportRequest, newView)) {{
          console.log("[Streaming] Requesting initial viewport cells...");
          isRequestPending = true;
          lastViewportRequest = newView;
          requestViewportCells(viewMinX, viewMaxX, viewMinY, viewMaxY);
        }}
      }}
      
      ctx.font = "16px ui-monospace, Menlo, Monaco, monospace";
      ctx.fillStyle = "rgba(0,0,0,0.8)";
      ctx.textAlign = "center";
      ctx.fillText("Loading cells...", W/2, H/2);
      ctx.textAlign = "left";
      return;
    }}
    
    ctx.save();
    ctx.translate(W / 2, H / 2);
    ctx.rotate(rotation);
    ctx.translate(-W / 2, -H / 2);

    const state = window["_plotState_" + iframeId] || {{ obs: {{}}, gex: {{}} }};
    const r = state.pointSize || 1.1;
    
    let renderedCount = 0;
    
    // Render all loaded cells (no viewport filtering - we already loaded the right ones!)
    for (const [idx, cell] of LOADED_CELLS.entries()) {{
      const px = offX + (cell.x - minX) * scale;
      const py = offY + (cell.y - minY) * scale;
      
      // Determine color
      let color = "rgba(150, 150, 150, 0.6)";  // Default grey
      
      if (cell.obs_value !== undefined && currentPalette) {{
        // Categorical color
        color = currentPalette[cell.obs_value] || fallbackCatColor(cell.obs_value);
      }} else if (cell.gex_value !== undefined) {{
        // GEX color
        const t = clamp01(cell.gex_value / 5.0);  // Simple normalization
        color = viridis(t);
      }}
      
      ctx.fillStyle = color;
      ctx.globalAlpha = 1.0;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      
      renderedCount++;
    }}

    ctx.globalAlpha = 1.0;
    ctx.restore();
    
    drawMinimap();
  }}
  
  // Request viewport cells from Python
  function requestViewportCells(viewMinX, viewMaxX, viewMinY, viewMaxY) {{
    // Build request data
    const requestData = {{
      embedding: currentEmbedding,
      viewMinX: viewMinX,
      viewMaxX: viewMaxX,
      viewMinY: viewMinY,
      viewMaxY: viewMaxY,
      activeColumn: currentObsColumn,
      activeGene: currentGexGene
    }};
    
    console.log("[Streaming] Requesting cells:", requestData);
    
    // Send via iframe postMessage
    window.parent.postMessage({{
      type: 'button_click',
      iframeId: iframeId,
      buttonId: 'viewportBtn',
      data: requestData,
      timestamp: Date.now()
    }}, '*');
  }}
  
  // ----------------------------
  // Minimap drawing
  // ----------------------------
  function drawMinimap() {{
    const mmW = minimap.width;
    const mmH = minimap.height;
    
    minimapCtx.fillStyle = "#000";
    minimapCtx.fillRect(0, 0, mmW, mmH);
    
    // Use minimap sample points (100K global sample)
    if (!MINIMAP_POINTS || MINIMAP_POINTS.length === 0) return;
    
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
    minimapCtx.fillStyle = "rgba(200, 200, 200, 0.8)";  // Brighter for visibility
    for (let i = 0; i < MINIMAP_POINTS.length; i++) {{
      const [x, y] = MINIMAP_POINTS[i];
      const px = offX + (x - minX) * scale;
      const py = offY + (y - minY) * scale;
      minimapCtx.fillRect(px - 0.5, py - 0.5, 1, 1);
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

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // ----------------------------
  // Pan/Zoom/Rotation controls
  // ----------------------------
  
  // Wheel zoom (slowed to 0.5x) with viewport refresh
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
    
    draw();
    
    // STREAMING: Request new viewport after zoom stops (debounced)
    clearTimeout(viewportRefreshTimer);
    viewportRefreshTimer = setTimeout(() => {{
      if (METADATA.streaming && !isRequestPending) {{
        const embedMeta = METADATA[currentEmbedding];
        if (embedMeta) {{
          const newView = getViewportBounds(embedMeta);
          
          if (hasViewportChanged(lastViewportRequest, newView)) {{
            console.log("[Streaming] Zoom stopped, requesting incremental fill...");
            isRequestPending = true;
            lastViewportRequest = newView;
            requestViewportCells(newView.minX, newView.maxX, newView.minY, newView.maxY);
          }}
        }}
      }}
    }}, 200);  // 200ms zoom response (reduced request frequency)
  }});
  
  // Double-click to reset view
  canvas.addEventListener("dblclick", () => {{
    zoom = 1.0;
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
    
    if (tool) {{
      // Selection drawing mode
      window["_isDrawing_" + iframeId] = true;
      window["_selectionPath_" + iframeId] = [[x, y]];
      
      if (tool === "rectangle" || tool === "circle") {{
        window["_selectionStart_" + iframeId] = [x, y];
      }}
    }} else if (!rotationMode) {{
      // Panning mode
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing";
    }}
  }});
  
  canvas.addEventListener("mousemove", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (tool && window["_isDrawing_" + iframeId]) {{
      // Selection drawing
      if (tool === "lasso" || tool === "polygon") {{
        window["_selectionPath_" + iframeId].push([x, y]);
      }} else if (tool === "rectangle" || tool === "circle") {{
        window["_selectionPath_" + iframeId] = [window["_selectionStart_" + iframeId], [x, y]];
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
    }}
  }});
  
  canvas.addEventListener("mouseup", () => {{
    const tool = window["_selectionTool_" + iframeId];
    
    if (tool && window["_isDrawing_" + iframeId]) {{
      window["_isDrawing_" + iframeId] = false;
      
      // For polygon, don't complete on mouseup
      if (tool === "polygon") {{
        return;
      }}
      
      completeSelection();
    }}
    
    // End panning and refresh viewport
    if (isDragging) {{
      isDragging = false;
      canvas.style.cursor = "default";
      
      // STREAMING: Request new viewport after pan stops
      if (METADATA.streaming && !isRequestPending) {{
        clearTimeout(viewportRefreshTimer);
        viewportRefreshTimer = setTimeout(() => {{
          const embedMeta = METADATA[currentEmbedding];
          if (embedMeta) {{
            const newView = getViewportBounds(embedMeta);
            
            if (hasViewportChanged(lastViewportRequest, newView)) {{
              console.log("[Streaming] Pan stopped, requesting incremental fill...");
              isRequestPending = true;
              lastViewportRequest = newView;
              requestViewportCells(newView.minX, newView.maxX, newView.minY, newView.maxY);
              draw();
            }}
          }}
        }}, 150);  // 150ms pan response (reduced request frequency)
      }}
    }}
  }});
  
  canvas.addEventListener("mouseleave", () => {{
    if (isDragging) {{
      isDragging = false;
      canvas.style.cursor = "default";
    }}
  }});
  
  // Polygon: click adds point, double-click completes
  canvas.addEventListener("click", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    if (tool !== "polygon" || !window["_isDrawing_" + iframeId]) return;
    
    // Point already added in mousemove
  }});
  
  canvas.addEventListener("dblclick", (e) => {{
    const tool = window["_selectionTool_" + iframeId];
    if (tool !== "polygon") return;
    
    e.preventDefault();
    completeSelection();
  }});
  
  // Draw white outline of current selection
  function drawSelectionOutline() {{
    draw(); // Redraw base plot
    
    const path = window["_selectionPath_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    if (!path || path.length === 0) return;
    
    const rect = panel.getBoundingClientRect();
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    if (tool === "lasso" && path.length > 1) {{
      ctx.beginPath();
      ctx.moveTo(path[0][0], path[0][1]);
      for (let i = 1; i < path.length; i++) {{
        ctx.lineTo(path[i][0], path[i][1]);
      }}
      ctx.stroke();
    }} else if (tool === "polygon" && path.length > 0) {{
      ctx.beginPath();
      ctx.moveTo(path[0][0], path[0][1]);
      for (let i = 1; i < path.length; i++) {{
        ctx.lineTo(path[i][0], path[i][1]);
      }}
      if (path.length > 2) {{
        ctx.closePath();
      }}
      ctx.stroke();
    }} else if (tool === "rectangle" && path.length === 2) {{
      const [x1, y1] = path[0];
      const [x2, y2] = path[1];
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }} else if (tool === "circle" && path.length === 2) {{
      const [x1, y1] = path[0];
      const [x2, y2] = path[1];
      const radius = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
      ctx.beginPath();
      ctx.arc(x1, y1, radius, 0, Math.PI * 2);
      ctx.stroke();
    }}
    
    ctx.restore();
  }}
  
  // Complete selection and detect points inside
  function completeSelection() {{
    const path = window["_selectionPath_" + iframeId];
    const tool = window["_selectionTool_" + iframeId];
    if (!path || path.length === 0) return;
    
    // Get current plot state to check categorical mask
    const state = window["_plotState_" + iframeId] || {{ obs: {{}} }};
    
    // Convert selection path from canvas coords to data coords
    const selectedIndices = [];
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    
    // Calculate bounds for current embedding
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < points.length; i++) {{
      const x = points[i][0], y = points[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }}
    
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
    for (let i = 0; i < points.length; i++) {{
      // IMPORTANT: Only select from visible (unmasked) points
      let passesMask = true;
      if (state.obs && state.obs.mode === "categorical" && state.obs.values && state.obs.values.length === points.length) {{
        const k = state.obs.values[i] | 0;
        const isEnabled = !state.obs.enabled || state.obs.enabled[k] !== false;
        passesMask = isEnabled;
      }}
      
      // Skip masked points
      if (!passesMask) continue;
      
      const [x, y] = points[i];
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
        const [x2, y2] = path[1];
        const radius = Math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2);
        const dist = Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
        inside = dist <= radius;
      }}
      
      if (inside) {{
        selectedIndices.push(i);
      }}
    }}
    
    // Send selection to iframe with path for dragging
    iframe.contentWindow.postMessage({{
      type: "selection_completed",
      indices: selectedIndices,
      path: path,
      tool: tool
    }}, "*");
    
    // Clear drawing state
    window["_selectionPath_" + iframeId] = [];
    window["_isDrawing_" + iframeId] = false;
    draw();
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
            layout=widgets.Layout(width="1px", height="1px", visibility="hidden"),
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
      width:12px;height:12px;border-radius:999px;
      border:1px solid rgba(0,0,0,.15);
      flex:0 0 auto;
    }
    .legend-row.off .legend-label{ opacity:.35; text-decoration:line-through; }
    .legend-row.off .legend-dot{ opacity:.15; }

    /* Gene chips */
    #geneChips{
      padding:8px;
      background:rgba(0,0,0,.02);
      border-radius:8px;
      border:1px solid rgba(0,0,0,.08);
      display:none;
      margin-top:10px;
    }
    #geneChipContainer{
      display:flex;
      flex-wrap:wrap;
      gap:6px;
    }
    .gene-chip{
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:5px 10px;
      background:rgba(141,236,245,.15);
      border:1px solid rgba(141,236,245,.4);
      border-radius:12px;
      font-size:11px;
      font-weight:800;
      color:#333;
      cursor:pointer;
      transition:all .15s;
    }
    .gene-chip:hover{
      background:rgba(141,236,245,.25);
      border-color:rgba(141,236,245,.6);
      box-shadow:0 2px 4px rgba(0,0,0,.08);
    }
    .gene-chip.active{
      background:rgba(141,236,245,.35);
      border-color:rgba(141,236,245,.9);
      box-shadow:0 2px 10px rgba(0,0,0,.10);
    }
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
        <span class="section-arrow expanded" id="size-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Size</label>
      </div>
      <div class="section-content expanded" id="size-content">
        <div class="opacity-row">
          <input type="range" id="sizeSlider" min="0.5" max="8" step="0.1" value="1.1">
          <div class="opacity-val" id="sizeVal">1.1</div>
        </div>
      </div>
    </div>

    <!-- Color by obs -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('colorby')">
        <span class="section-arrow expanded" id="colorby-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Color By (obs column)</label>
      </div>
      <div class="section-content expanded" id="colorby-content">
        <input
          type="text"
          id="obsInput"
          data-for="obsBtn"
          data-key="column"
        placeholder="Enter obs column name"
        list="obsList"
      />
      <datalist id="obsList"></datalist>

      <div style="margin-top:10px">
        <div class="opacity-row">
          <input type="range" id="obsOpacity" min="0" max="1" step="0.05" value="1.0">
          <div class="opacity-val" id="obsOpacityVal">1.00</div>
        </div>
      </div>

      <div class="btn-row">
        <!-- IMPORTANT: keep id="obsBtn" so your Python callback is triggered -->
        <button class="btn-primary" id="obsBtn">Apply</button>
        <button class="btn-secondary" id="clearObsBtn">Clear</button>
      </div>

      <div id="legend" class="legend"></div>
      </div>
    </div>

    <!-- GEX -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('gex')">
        <span class="section-arrow expanded" id="gex-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">GEX (gene)</label>
      </div>
      <div class="section-content expanded" id="gex-content">
        <input
          type="text"
          id="gexInput"
          data-for="geneBtn"
          data-key="gene"
          placeholder="Enter gene name"
          list="geneList"
        />
        <datalist id="geneList"></datalist>

        <div style="margin-top:10px">
          <div class="opacity-row">
            <input type="range" id="gexOpacity" min="0" max="1" step="0.05" value="1.0">
            <div class="opacity-val" id="gexOpacityVal">1.00</div>
          </div>
        </div>

      <div class="btn-row">
        <!-- IMPORTANT: keep id="geneBtn" so your Python callback is triggered -->
        <button class="btn-primary" id="geneBtn">Add Gene</button>
        <button class="btn-secondary" id="clearGexBtn">Clear GEX</button>
      </div>

      <div id="geneChips">
        <div style="font-size:11px;font-weight:800;color:#666;margin-bottom:6px;">
          Added Genes (click to visualize):
        </div>
        <div id="geneChipContainer"></div>
      </div>
      </div>
    </div>

    <!-- Selection Tools -->
    <div class="control-group">
      <div class="section-header" onclick="toggleSection('selection')">
        <span class="section-arrow expanded" id="selection-arrow">▶</span>
        <label class="control-label" style="margin:0;cursor:pointer;">Selection</label>
      </div>
      <div class="section-content expanded" id="selection-content">
        <div class="selection-tools" style="display:flex; gap:6px; margin-top:10px;">
          <button class="tool-btn" id="lassoBtn" data-tool="lasso" title="Lasso tool">Lasso</button>
          <button class="tool-btn" id="polygonBtn" data-tool="polygon" title="Polygon tool">Poly</button>
          <button class="tool-btn" id="rectangleBtn" data-tool="rectangle" title="Rectangle tool">Square</button>
          <button class="tool-btn" id="circleBtn" data-tool="circle" title="Circle tool">Circle</button>
        </div>

        <div class="btn-row" style="margin-top:10px;">
        <button class="btn-primary" id="groupSelectionBtn">Group</button>
        <button class="btn-secondary" id="clearSelectionBtn">Clear All</button>
      </div>

      <div id="selectionChips" style="display:none; margin-top:10px;">
        <div style="font-size:11px;font-weight:800;color:#666;margin-bottom:6px;">
          Selections (double-click to rename):
        </div>
        <div id="selectionChipContainer"></div>
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
      
      if (STATE.selection.active) {
        // Check if active is a folder
        if (STATE.selection.folders[STATE.selection.active]) {
          // Combine all selections in folder
          const folder = STATE.selection.folders[STATE.selection.active];
          const combinedIndices = new Set();
          folder.selections.forEach(selName => {
            const indices = STATE.selection.selections[selName]?.indices || [];
            indices.forEach(idx => combinedIndices.add(idx));
          });
          selectionIndices = Array.from(combinedIndices);
          // Don't show outline for folders (multiple selections)
        } else {
          // Regular selection - get indices and path
          const sel = STATE.selection.selections[STATE.selection.active];
          selectionIndices = sel?.indices || null;
          selectionPath = sel?.path || null;
          selectionTool = sel?.tool || null;
        }
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
          active: STATE.gex.active,  // gene name or null
          values: STATE.gex.active ? (STATE.gex.genes[STATE.gex.active] || null) : null,
          opacity: STATE.gex.opacity
        },
        selection: {
          active: STATE.selection.active,  // selection/folder name or null
          indices: selectionIndices,
          path: selectionPath,  // for drawing outline
          tool: selectionTool    // for drawing outline
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
    // Legend rendering + toggles
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
        row.onclick = () => {
          STATE.obs.enabled[i] = !STATE.obs.enabled[i];
          renderLegend();
          postUIState();
        };

        const label = document.createElement("div");
        label.className = "legend-label";
        label.textContent = String(c);

        const dot = document.createElement("div");
        dot.className = "legend-dot";
        dot.style.background = pal[i] ? pal[i] : "hsl(" + ((i * 137.508) % 360) + ",55%,55%)";

        row.appendChild(label);
        row.appendChild(dot);
        body.appendChild(row);
      });

      legend.innerHTML = "";
      legend.appendChild(header);
      legend.appendChild(body);
    }

    // ----------------------------
    // Gene chips rendering
    // ----------------------------
    function renderGeneChips() {
      const box = document.getElementById("geneChips");
      const cont = document.getElementById("geneChipContainer");
      cont.innerHTML = "";

      if (!STATE.gex.order.length) {
        box.style.display = "none";
        return;
      }
      box.style.display = "block";

      STATE.gex.order.forEach((g) => {
        const chip = document.createElement("div");
        chip.className = "gene-chip" + (STATE.gex.active === g ? " active" : "");
        chip.title = STATE.gex.active === g ? "Click to hide" : "Click to visualize";
        chip.onclick = (e) => {
          // avoid remove click bubbling
          if (e.target && e.target.classList && e.target.classList.contains("gene-chip-remove")) return;
          
          // Toggle: if already active, deactivate; otherwise activate
          if (STATE.gex.active === g) {
            STATE.gex.active = null;  // Turn off GEX overlay
          } else {
            STATE.gex.active = g;  // Turn on this gene
          }
          
          renderGeneChips();
          postUIState();
        };

        const txt = document.createElement("div");
        txt.textContent = g;

        const rm = document.createElement("div");
        rm.className = "gene-chip-remove";
        rm.textContent = "×";
        rm.title = "Remove gene";
        rm.onclick = (e) => {
          e.stopPropagation();
          delete STATE.gex.genes[g];
          STATE.gex.order = STATE.gex.order.filter(x => x !== g);
          // If removing the active gene, clear the overlay
          if (STATE.gex.active === g) {
            STATE.gex.active = null;
          }
          renderGeneChips();
          postUIState();
        };

        chip.appendChild(txt);
        chip.appendChild(rm);
        cont.appendChild(chip);
      });
    }

    // ----------------------------
    // Selection chips rendering (with folders and multi-select)
    // ----------------------------
    function renderSelectionChips() {
      const box = document.getElementById("selectionChips");
      const cont = document.getElementById("selectionChipContainer");
      cont.innerHTML = "";

      if (!STATE.selection.order.length) {
        box.style.display = "none";
        return;
      }
      box.style.display = "block";

      STATE.selection.order.forEach((name) => {
        // Check if this is a folder
        if (STATE.selection.folders[name]) {
          renderFolder(name, cont);
        } else {
          // Regular selection
          renderSelectionChip(name, cont);
        }
      });
    }

    function renderFolder(folderName, container) {
      const folder = STATE.selection.folders[folderName];
      const folderDiv = document.createElement("div");
      folderDiv.className = "selection-folder";

      // Folder header
      const header = document.createElement("div");
      const isActive = STATE.selection.active === folderName;
      header.className = "folder-header" + (isActive ? " active" : "");
      
      // Arrow for expand/collapse
      const arrow = document.createElement("span");
      arrow.className = "folder-arrow" + (folder.expanded ? " expanded" : "");
      arrow.textContent = "▶";
      
      // Folder name
      const nameSpan = document.createElement("span");
      nameSpan.className = "folder-name";
      nameSpan.textContent = folderName;
      
      // Save button (saves folder as obs column)
      const saveBtn = document.createElement("button");
      saveBtn.className = "folder-save-btn";
      saveBtn.textContent = "Save";
      saveBtn.title = "Save to adata.obs";
      saveBtn.style.cssText = `
        background:rgba(141,236,245,0.2);
        border:1px solid rgba(141,236,245,0.5);
        border-radius:4px;
        padding:4px 8px;
        cursor:pointer;
        font-size:10px;
        font-weight:700;
        margin-left:auto;
        color:#333;
      `;
      saveBtn.onclick = (e) => {
        e.stopPropagation();
        saveFolderToObs(folderName);
      };
      
      // Remove button
      const rm = document.createElement("div");
      rm.className = "gene-chip-remove";
      rm.textContent = "×";
      rm.title = "Delete folder";
      rm.onclick = (e) => {
        e.stopPropagation();
        // Move selections back to top level
        folder.selections.forEach(sel => {
          if (!STATE.selection.order.includes(sel)) {
            STATE.selection.order.push(sel);
          }
        });
        // Remove folder
        delete STATE.selection.folders[folderName];
        STATE.selection.order = STATE.selection.order.filter(x => x !== folderName);
        if (STATE.selection.active === folderName) {
          STATE.selection.active = null;
        }
        renderSelectionChips();
        postUIState();
      };
      
      // Header click: toggle folder active state
      header.onclick = (e) => {
        if (e.target === rm) return;
        if (STATE.selection.active === folderName) {
          STATE.selection.active = null;
        } else {
          STATE.selection.active = folderName;
        }
        renderSelectionChips();
        postUIState();
      };
      
      // Arrow click: toggle expansion
      arrow.onclick = (e) => {
        e.stopPropagation();
        folder.expanded = !folder.expanded;
        renderSelectionChips();
      };
      
      // Double-click: rename
      header.ondblclick = (e) => {
        e.stopPropagation();
        const newName = prompt("Rename folder:", folderName);
        if (!newName || newName.trim() === "" || newName === folderName) return;
        
        STATE.selection.folders[newName.trim()] = folder;
        delete STATE.selection.folders[folderName];
        const idx = STATE.selection.order.indexOf(folderName);
        STATE.selection.order[idx] = newName.trim();
        if (STATE.selection.active === folderName) {
          STATE.selection.active = newName.trim();
        }
        renderSelectionChips();
        postUIState();
      };
      
      header.appendChild(arrow);
      header.appendChild(nameSpan);
      header.appendChild(saveBtn);
      header.appendChild(rm);
      folderDiv.appendChild(header);
      
      // Folder children (selections)
      if (folder.expanded) {
        const childrenDiv = document.createElement("div");
        childrenDiv.className = "folder-children expanded";
        folder.selections.forEach(selName => {
          renderSelectionChip(selName, childrenDiv, true);
        });
        folderDiv.appendChild(childrenDiv);
      }
      
      container.appendChild(folderDiv);
    }

    // Save folder as obs column in adata
    function saveFolderToObs(folderName) {
      const folder = STATE.selection.folders[folderName];
      if (!folder) return;
      
      // Build mapping: cell index -> selection name
      const columnData = {};
      folder.selections.forEach(selName => {
        const sel = STATE.selection.selections[selName];
        if (sel && sel.indices) {
          sel.indices.forEach(idx => {
            columnData[idx] = selName;
          });
        }
      });
      
      // Send to Python backend to save to adata.obs
      window.parent.postMessage({
        type: "save_obs_column",
        iframeId: window.frameElement?.id || "",
        columnName: folderName,
        columnData: columnData  // {index: value} mapping
      }, "*");
      
      alert(`Saved "${folderName}" to adata.obs!\n\nCells in selections are labeled by selection name.\nOther cells are NaN.`);
    }

    function renderSelectionChip(name, container, isInFolder = false) {
      const chip = document.createElement("div");
      const isActive = STATE.selection.active === name;
      const isMultiSelected = STATE.selection.multiSelect.includes(name);
      chip.className = "gene-chip" 
        + (isActive ? " active" : "")
        + (isMultiSelected ? " multi-selected" : "");
      chip.title = isActive ? "Click to deactivate (show all cells)" : "Click to activate (dim unselected cells)";
      
      // Click: toggle active OR multi-select with Shift
      chip.onclick = (e) => {
        if (e.target && e.target.classList && e.target.classList.contains("gene-chip-remove")) return;
        
        if (e.shiftKey) {
          // Shift+click: add to multi-select
          const idx = STATE.selection.multiSelect.indexOf(name);
          if (idx === -1) {
            STATE.selection.multiSelect.push(name);
          } else {
            STATE.selection.multiSelect.splice(idx, 1);
          }
          renderSelectionChips();
        } else {
          // Normal click: toggle active
          if (STATE.selection.active === name) {
            STATE.selection.active = null;
          } else {
            STATE.selection.active = name;
          }
          renderSelectionChips();
          postUIState();
        }
      };

      // Double-click: rename
      chip.ondblclick = (e) => {
        e.stopPropagation();
        const newName = prompt("Rename selection:", name);
        if (!newName || newName.trim() === "" || newName === name) return;
        
        STATE.selection.selections[newName.trim()] = STATE.selection.selections[name];
        delete STATE.selection.selections[name];
        const idx = STATE.selection.order.indexOf(name);
        if (idx !== -1) STATE.selection.order[idx] = newName.trim();
        if (STATE.selection.active === name) {
          STATE.selection.active = newName.trim();
        }
        
        // Update in multi-select
        const msIdx = STATE.selection.multiSelect.indexOf(name);
        if (msIdx !== -1) {
          STATE.selection.multiSelect[msIdx] = newName.trim();
        }
        
        // Update in any folders
        Object.values(STATE.selection.folders).forEach(folder => {
          const fIdx = folder.selections.indexOf(name);
          if (fIdx !== -1) {
            folder.selections[fIdx] = newName.trim();
          }
        });
        
        renderSelectionChips();
        postUIState();
      };

      const txt = document.createElement("div");
      txt.textContent = name;

      const rm = document.createElement("div");
      rm.className = "gene-chip-remove";
      rm.textContent = "×";
      rm.title = "Remove selection";
      rm.onclick = (e) => {
        e.stopPropagation();
        delete STATE.selection.selections[name];
        STATE.selection.order = STATE.selection.order.filter(x => x !== name);
        STATE.selection.multiSelect = STATE.selection.multiSelect.filter(x => x !== name);
        if (STATE.selection.active === name) {
          STATE.selection.active = null;
        }
        
        // Remove from folders
        Object.values(STATE.selection.folders).forEach(folder => {
          folder.selections = folder.selections.filter(x => x !== name);
        });
        
        renderSelectionChips();
        postUIState();
      };

      chip.appendChild(txt);
      chip.appendChild(rm);
      container.appendChild(chip);
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

    document.getElementById("groupSelectionBtn").addEventListener("click", () => {
      // Group all multi-selected items into a folder
      if (STATE.selection.multiSelect.length === 0) {
        alert("Shift+click selections to group them together");
        return;
      }
      
      const folderName = `Group ${STATE.selection.folderCounter}`;
      STATE.selection.folderCounter++;
      
      // Create folder
      STATE.selection.folders[folderName] = {
        selections: [...STATE.selection.multiSelect],
        expanded: true
      };
      
      // Remove grouped selections from top-level order
      STATE.selection.multiSelect.forEach(sel => {
        STATE.selection.order = STATE.selection.order.filter(x => x !== sel);
      });
      
      // Add folder to order
      STATE.selection.order.push(folderName);
      
      // Clear multi-select
      STATE.selection.multiSelect = [];
      
      renderSelectionChips();
      postUIState();
    });

    document.getElementById("clearSelectionBtn").addEventListener("click", () => {
      STATE.selection.selections = {};
      STATE.selection.order = [];
      STATE.selection.active = null;
      STATE.selection.counter = 1;
      STATE.selection.folders = {};
      STATE.selection.folderCounter = 1;
      STATE.selection.multiSelect = [];
      renderSelectionChips();
      postUIState();
    });

    // Enter-to-apply
    document.getElementById("obsInput").addEventListener("keypress", (e) => {
      if (e.key === "Enter") document.getElementById("obsBtn").click();
    });
    document.getElementById("gexInput").addEventListener("keypress", (e) => {
      if (e.key === "Enter") document.getElementById("geneBtn").click();
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
        const gene = data.gene || null;
        const vals = data.values || null;
        if (!gene || !vals) return;

        STATE.gex.genes[gene] = vals;
        if (!STATE.gex.order.includes(gene)) STATE.gex.order.push(gene);

        // newest gene becomes active by default
        STATE.gex.active = gene;

        // clear input box
        const inp = document.getElementById("gexInput");
        if (inp) inp.value = "";

        renderGeneChips();
        postUIState();
      }
    });

    // Listen for selection completed from parent canvas
    window.addEventListener("message", (event) => {
      if (!event.data || event.data.type !== "selection_completed") return;
      
      const indices = event.data.indices || [];
      if (!indices.length) return;
      
      // Create selection name
      const name = `Selection ${STATE.selection.counter}`;
      STATE.selection.counter++;
      
      // Store selection with path for future dragging
      STATE.selection.selections[name] = {
        indices: indices,
        tool: STATE.selection.tool,
        path: event.data.path || null  // Store original path for dragging
      };
      
      // Add to order and make active
      if (!STATE.selection.order.includes(name)) STATE.selection.order.push(name);
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

def load_full_embedding(data, adata=None, __sample_idx=None):
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


def set_embedding_spatial(data, adata=None, __sample_idx=None):
    """Switch to spatial embedding - return message to trigger streaming"""
    return {"type": "set_embedding", "embedding": "spatial"}


def set_embedding_umap(data, adata=None, __sample_idx=None):
    """Switch to UMAP embedding - load in chunks"""
    if adata is None or "X_umap" not in adata.obsm:
        return {"type": "error", "message": "No UMAP embedding"}
    
    coords = np.asarray(adata.obsm["X_umap"])[:, :2]
    total_cells = len(coords)
    
    # Get chunk info
    chunk_size = 500_000  # 500K cells per chunk
    chunk_idx = data.get("chunk", 0)
    
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_cells)
    
    # Get this chunk
    chunk_coords = coords[start_idx:end_idx]
    chunk_indices = np.arange(start_idx, end_idx)
    
    print(f"[Full Embedding] Loading UMAP chunk {chunk_idx + 1}: cells {start_idx:,} to {end_idx:,}")
    
    coords_binary = base64.b64encode(chunk_coords.astype(np.float32).tobytes()).decode('ascii')
    
    is_final = end_idx >= total_cells
    
    return {
        "type": "full_embedding_chunk",
        "embedding": "umap",
        "indices": chunk_indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(chunk_indices),
        "chunk": chunk_idx,
        "is_final": is_final,
        "total_cells": total_cells,
        "loaded_cells": end_idx
    }


def set_embedding_pca(data, adata=None, __sample_idx=None):
    """Switch to PCA embedding - load in chunks"""
    if adata is None or "X_pca" not in adata.obsm:
        return {"type": "error", "message": "No PCA embedding"}
    
    coords = np.asarray(adata.obsm["X_pca"])[:, :2]
    total_cells = len(coords)
    
    # Get chunk info
    chunk_size = 500_000  # 500K cells per chunk
    chunk_idx = data.get("chunk", 0)
    
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_cells)
    
    # Get this chunk
    chunk_coords = coords[start_idx:end_idx]
    chunk_indices = np.arange(start_idx, end_idx)
    
    print(f"[Full Embedding] Loading PCA chunk {chunk_idx + 1}: cells {start_idx:,} to {end_idx:,}")
    
    coords_binary = base64.b64encode(chunk_coords.astype(np.float32).tobytes()).decode('ascii')
    
    is_final = end_idx >= total_cells
    
    return {
        "type": "full_embedding_chunk",
        "embedding": "pca",
        "indices": chunk_indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(chunk_indices),
        "chunk": chunk_idx,
        "is_final": is_final,
        "total_cells": total_cells,
        "loaded_cells": end_idx
    }


def get_viewport_cells(data, adata=None, __sample_idx=None):
    """
    STREAMING: Load cells in viewport (coordinates + any active colors/genes)
    Max 100K cells returned at once.
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
    # JavaScript sends us a square bounding box, but we filter circularly
    center_x = (view_minX + view_maxX) / 2
    center_y = (view_minY + view_maxY) / 2
    
    # The bounding box represents a circle - calculate radius
    radius = (view_maxX - view_minX) / 2  # Half the width = radius
    
    # Calculate distance from center for all points
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)
    
    # Select cells within circular radius (not rectangular!)
    in_circle = distances <= radius
    candidate_indices = np.where(in_circle)[0]
    
    # Calculate normalized distance from center (for probability weighting)
    normalized_dist = distances[candidate_indices] / radius  # 0 at center, 1 at edge
    
    # Gaussian falloff: 1.0 at center, smoothly to 0 at edges
    sigma = 0.5  # Controls falloff sharpness
    probabilities = np.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
    
    # Limit to 30K cells per request (lighter transfers, less connection issues)
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
    
    # Build response
    response = {
        "type": "viewport_cells",
        "embedding": embedding,
        "indices": visible_indices.tolist(),
        "coords_binary": coords_binary,
        "coords_count": len(visible_indices)
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

def get_obs_column(data, adata=None, __sample_idx=None):
    """STREAMING: Just return column metadata, JavaScript will request viewport refresh"""
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

    # Get categories and palette (but NO values!)
    vals = adata.obs[column]
    
    # Numeric → continuous
    if pd.api.types.is_numeric_dtype(vals):
        return {
            "type": "obs_values",
            "column": column,
            "mode": "continuous",
            "categories": None,
            "colors": None,
        }

    # Categorical → just categories + colors (NO values!)
    if pd.api.types.is_categorical_dtype(vals):
        categories = list(vals.cat.categories)
    else:
        categories = list(pd.unique(vals))

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
        "categories": categories,
        "colors": palette,
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
            
            # STREAMING: viewport cell loader
            "viewportBtn": get_viewport_cells,
            
            # FULL EMBEDDING LOADER: For UMAP/PCA (load all cells)
            "loadEmbeddingBtn": load_full_embedding,
            
            # NEW: save selection folder to obs
            "__save_obs_column__": save_obs_column,
        },
        callback_args={"adata": adata},
        height=height,
        width=width,
        debug=debug,
        initial_data=initial_data,
    )