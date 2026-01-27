"""
Bridge module - links HTML buttons to Python callbacks.
"""

import base64
import json
import uuid
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from IPython.display import display, HTML, Javascript
import ipywidgets as widgets


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
    from IPython.display import clear_output
    clear_output(wait=True)
    
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
    # Precompute embedding points + sample indices (aligned everywhere)
    # ----------------------------
    adata = callback_args.get("adata", None)

    # pick base embedding to define sample_idx
    spatial = None
    if adata is not None:
        if "spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["spatial"])
        elif "X_spatial" in adata.obsm:
            spatial = np.asarray(adata.obsm["X_spatial"])

    if spatial is None or spatial.ndim != 2 or spatial.shape[1] < 2:
        sample_idx = np.array([], dtype=int)
        spatial_points = []
    else:
        n = spatial.shape[0]
        max_points = 1_00_000_000  # <-- THIS controls how many cells/points are loaded
        if n > max_points:
            sample_idx = np.random.choice(n, size=max_points, replace=False)
        else:
            sample_idx = np.arange(n, dtype=int)
        spatial_points = spatial[sample_idx, :2].tolist()

    # Optional embeddings (same sample_idx; empty if missing)
    umap_points: list = []
    pca_points: list = []
    if adata is not None and sample_idx.size > 0:
        if "X_umap" in adata.obsm:
            um = np.asarray(adata.obsm["X_umap"])
            if um.ndim == 2 and um.shape[1] >= 2 and um.shape[0] == (spatial.shape[0] if spatial is not None else um.shape[0]):
                umap_points = um[sample_idx, :2].tolist()

        if "X_pca" in adata.obsm:
            pc = np.asarray(adata.obsm["X_pca"])
            if pc.ndim == 2 and pc.shape[1] >= 2 and pc.shape[0] == (spatial.shape[0] if spatial is not None else pc.shape[0]):
                pca_points = pc[sample_idx, :2].tolist()

    embeds_js = json.dumps({
        "spatial": spatial_points,
        "umap": umap_points,
        "pca": pca_points,
    })

    # IMPORTANT: pass sample_idx into callbacks so obs/gene are subset to plotted points
    callback_args = dict(callback_args)
    callback_args["__sample_idx"] = sample_idx.tolist()  # JSON-friendly

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

  // ----------------------------
  // Embeddings (preloaded)
  // ----------------------------
  const EMBEDS = {embeds_js};
  let currentEmbedding = "spatial";
  let points = (EMBEDS[currentEmbedding] || []);

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

    // Switch embedding (UMAP/PCA/Spatial) - PRESERVE COLOR STATE WITH ANIMATION
    if (payload.type === "set_embedding") {{
      const name = (payload.embedding || "spatial");
      const newPoints = (EMBEDS[name] || []);
      
      // Don't animate if already on this embedding
      if (currentEmbedding === name) return;
      
      // Start animation
      startPoints = points.slice(); // Copy current points
      endPoints = newPoints;
      currentEmbedding = name;
      isAnimating = true;
      animationStartTime = Date.now();
      
      // Reset navigation on embedding switch
      zoom = 1.0;
      panX = 0;
      panY = 0;
      rotation = 0;
      
      function animateEmbedding() {{
        const elapsed = Date.now() - animationStartTime;
        const progress = Math.min(elapsed / animationDuration, 1);
        const eased = easeInOutQuad(progress);
        
        // Interpolate all points
        points = [];
        for (let i = 0; i < startPoints.length; i++) {{
          const [sx, sy] = startPoints[i];
          const [ex, ey] = endPoints[i];
          points.push([
            lerp(sx, ex, eased),
            lerp(sy, ey, eased)
          ]);
        }}
        
        draw();
        
        if (progress < 1) {{
          requestAnimationFrame(animateEmbedding);
        }} else {{
          points = endPoints;
          isAnimating = false;
          draw();
        }}
      }}
      
      animateEmbedding();

      // Update label
      const s = window["_plotState_" + iframeId];
      if (s) {{
        if (s.mode === "none") {{
          s.label = "Embedding: " + name;
        }}
      }}
      setLabel(s ? s.label : "Embedding: " + name);
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

    // obs values (categorical or continuous) - store in obs layer
    if (payload.type === "obs_values") {{
      const s = window["_plotState_" + iframeId];
      if (s) {{
        if (payload.mode === "categorical") {{
          s.obs.mode = "categorical";
          s.obs.values = payload.values || null;
          s.obs.colors = payload.colors || null;
          s.obs.categories = payload.categories || null;
          // Initialize enabled array - all true by default
          if (s.obs.categories) {{
            s.obs.enabled = s.obs.categories.map(() => true);
          }}
        }} else {{
          s.obs.mode = "continuous";
          s.obs.values = payload.values || null;
          s.obs.colors = null;
          s.obs.categories = null;
          s.obs.enabled = null;
        }}
        s.label = "obs: " + (payload.column || "");
      }}
      setLabel("obs: " + (payload.column || ""));
      draw();
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

    if (!points || points.length === 0) {{
      ctx.font = "13px ui-monospace, Menlo, Monaco, monospace";
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillText("Embedding not available (missing in adata.obsm).", 12, 40);
      return;
    }}

    // bounds
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
    
    // Apply zoom
    const scale = baseScale * zoom;

    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (W - usedW) / 2 + panX;
    const offY = (H - usedH) / 2 + panY;
    
    // Apply rotation transform
    ctx.save();
    ctx.translate(W / 2, H / 2);
    ctx.rotate(rotation);
    ctx.translate(-W / 2, -H / 2);

    const state = window["_plotState_" + iframeId] || {{ obs: {{}}, gex: {{}} }};

    // Calculate GEX continuous range if needed
    let gexMin = 0, gexMax = 1;
    if (state.gex && state.gex.values && state.gex.values.length === points.length) {{
      gexMin = Infinity; gexMax = -Infinity;
      for (let i = 0; i < state.gex.values.length; i++) {{
        const v = state.gex.values[i];
        if (v == null || Number.isNaN(v)) continue;
        if (v < gexMin) gexMin = v;
        if (v > gexMax) gexMax = v;
      }}
      if (!isFinite(gexMin) || !isFinite(gexMax) || gexMin === gexMax) {{ gexMin = 0; gexMax = 1; }}
    }}

    // Calculate obs continuous range if needed
    let obsMin = 0, obsMax = 1;
    if (state.obs && state.obs.mode === "continuous" && state.obs.values && state.obs.values.length === points.length) {{
      obsMin = Infinity; obsMax = -Infinity;
      for (let i = 0; i < state.obs.values.length; i++) {{
        const v = state.obs.values[i];
        if (v == null || Number.isNaN(v)) continue;
        if (v < obsMin) obsMin = v;
        if (v > obsMax) obsMax = v;
      }}
      if (!isFinite(obsMin) || !isFinite(obsMax) || obsMin === obsMax) {{ obsMin = 0; obsMax = 1; }}
    }}

    const r = state.pointSize || 1.1;

    // Layered rendering with categorical masking
    for (let i = 0; i < points.length; i++) {{
      const x = points[i][0];
      const y = points[i][1];
      const px = offX + (x - minX) * scale;
      const py = offY + (y - minY) * scale;

      // MASKING: Check if this point passes categorical mask
      let passesMask = true;
      
      // Categorical mask (from obs toggles)
      if (state.obs && state.obs.mode === "categorical" && state.obs.values && state.obs.values.length === points.length) {{
        const k = state.obs.values[i] | 0;
        const isEnabled = !state.obs.enabled || state.obs.enabled[k] !== false;
        passesMask = passesMask && isEnabled;
      }}
      
      // If point doesn't pass categorical mask, skip it entirely
      if (!passesMask) {{
        continue;
      }}
      
      // SELECTION OPACITY: Check if this point is in active selection
      let selectionOpacity = 1.0;
      if (state.selectionIndices && state.selectionIndices.length > 0) {{
        // If there's an active selection, selected points = 1.0, others = 0.2
        selectionOpacity = state.selectionIndices.includes(i) ? 1.0 : 0.2;
      }}
      
      // Determine what to draw for this point
      let hasObsColor = false;
      let hasGexColor = false;
      
      // Check obs layer
      if (state.obs && state.obs.mode === "categorical" && state.obs.values && state.obs.values.length === points.length) {{
        hasObsColor = true;
      }} else if (state.obs && state.obs.mode === "continuous" && state.obs.values && state.obs.values.length === points.length) {{
        hasObsColor = true;
      }}
      
      // Check GEX layer
      if (state.gex && state.gex.values && state.gex.values.length === points.length) {{
        hasGexColor = true;
      }}
      
      // Skip point entirely if no active layers (no obs, no GEX)
      if (!hasObsColor && !hasGexColor) {{
        continue;
      }}

      // Draw obs base layer first (if exists)
      if (hasObsColor) {{
        if (state.obs.mode === "categorical") {{
          const k = state.obs.values[i] | 0;
          const pal = state.obs.colors;
          ctx.globalAlpha = (state.obs.opacity || 1.0) * selectionOpacity;
          ctx.fillStyle = (pal && pal[k]) ? pal[k] : fallbackCatColor(k);
        }} else if (state.obs.mode === "continuous") {{
          const v = state.obs.values[i];
          const t = (v - obsMin) / (obsMax - obsMin || 1);
          ctx.globalAlpha = (state.obs.opacity || 1.0) * selectionOpacity;
          ctx.fillStyle = viridis(t);
        }}
        
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }}

      // Draw GEX overlay on top (if exists)
      if (hasGexColor) {{
        const v = state.gex.values[i];
        const t = (v - gexMin) / (gexMax - gexMin || 1);
        ctx.globalAlpha = (state.gex.opacity || 1.0) * selectionOpacity;
        ctx.fillStyle = viridis(t);
        
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    ctx.globalAlpha = 1.0;
    
    // Restore rotation transform
    ctx.restore();
    
    // NOTE: We don't draw stored selection outlines because they're stored in
    // screen coordinates and become incorrect after zoom/pan/rotate.
    // The opacity dimming (selected=1.0, unselected=0.2) is sufficient visual feedback.
    
    // Draw minimap
    drawMinimap();
  }}
  
  // ----------------------------
  // Minimap drawing
  // ----------------------------
  function drawMinimap() {{
    const mmW = minimap.width;
    const mmH = minimap.height;
    
    minimapCtx.fillStyle = "#000";
    minimapCtx.fillRect(0, 0, mmW, mmH);
    
    // Calculate bounds for current points
    if (!points || points.length === 0) return;
    
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < points.length; i++) {{
      const x = points[i][0], y = points[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }}
    
    const pad = 5;
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const scale = Math.min((mmW - 2*pad) / spanX, (mmH - 2*pad) / spanY);
    const usedW = spanX * scale;
    const usedH = spanY * scale;
    const offX = (mmW - usedW) / 2;
    const offY = (mmH - usedH) / 2;
    
    // Draw all points as small dots
    minimapCtx.fillStyle = "rgba(100, 100, 100, 0.5)";
    for (let i = 0; i < points.length; i++) {{
      const [x, y] = points[i];
      const px = offX + (x - minX) * scale;
      const py = offY + (y - minY) * scale;
      minimapCtx.fillRect(px - 0.5, py - 0.5, 1, 1);
    }}
    
    // Draw viewport rectangle
    const rect = panel.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    const baseScale = Math.min((W - 24) / spanX, (H - 24) / spanY);
    const viewScale = baseScale * zoom;
    const viewW = spanX * viewScale;
    const viewH = spanY * viewScale;
    const viewOffX = (W - viewW) / 2 + panX;
    const viewOffY = (H - viewH) / 2 + panY;
    
    // Map viewport to minimap coordinates
    const vpMinX = -viewOffX / viewScale * scale + offX;
    const vpMinY = -viewOffY / viewScale * scale + offY;
    const vpMaxX = (W - viewOffX) / viewScale * scale + offX;
    const vpMaxY = (H - viewOffY) / viewScale * scale + offY;
    
    minimapCtx.strokeStyle = "rgba(141, 236, 245, 0.8)";
    minimapCtx.lineWidth = 1;
    minimapCtx.strokeRect(vpMinX, vpMinY, vpMaxX - vpMinX, vpMaxY - vpMinY);
  }}

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // ----------------------------
  // Pan/Zoom/Rotation controls
  // ----------------------------
  
  // Wheel zoom (slowed to 0.5x)
  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const delta = e.deltaY * 0.0005;  // Was 0.001, now 0.0005 for slower zoom
    const zoomFactor = delta > 0 ? 0.975 : 1.025;  // Reduced from 0.95/1.05
    const oldZoom = zoom;
    zoom *= zoomFactor;
    zoom = Math.max(0.1, Math.min(zoom, 20));
    
    const zoomChange = zoom / oldZoom;
    panX = mouseX - (mouseX - panX) * zoomChange;
    panY = mouseY - (mouseY - panY) * zoomChange;
    
    draw();
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
    
    // End panning
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
