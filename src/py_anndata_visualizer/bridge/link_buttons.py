"""
Bridge module for connecting HTML UI to Python callbacks.

This module provides the core functionality for creating a two-panel interface
with an iframe-based control panel and a WebGL canvas, with bidirectional
communication between JavaScript and Python via ipywidgets.
"""

import base64
import json
import os
import traceback
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript

from ..tools.utils import _b64, _pack_coords_binary, _serialize_result


def _load_js_file(filename: str) -> str:
    """Load a JavaScript file from the js/ directory."""
    js_dir = Path(__file__).parent.parent / "js"
    filepath = js_dir / filename
    if filepath.exists():
        return filepath.read_text()
    else:
        raise FileNotFoundError(f"JavaScript file not found: {filepath}")


def _build_container_html(iframe_id: str, height: int) -> str:
    """Build the HTML for the two-panel container layout."""
    return f"""
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
          0 / 0 cells
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
"""


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
    
    Creates a 2-panel layout:
      - Left: iframe UI (buttons/inputs)
      - Right: plot canvas (transparent) that can:
          * render embedding points: spatial/umap/pca (loaded upfront)
          * color points by continuous (GEX/obs numeric) or categorical (obs)
          * clear overlay back to neutral grey
    
    Args:
        html_content: HTML template for the control panel UI
        button_callbacks: Dict mapping button IDs to Python callback functions
        callback_args: Additional arguments passed to all callbacks (e.g., {"adata": adata})
        height: Height of the visualization in pixels
        width: Width of the visualization in pixels (used for layout)
        debug: Enable verbose logging
        max_result_size: Maximum size for callback results in bytes
        initial_data: Initial data passed to JavaScript on load
        chunk_size: Number of cells per chunk for progressive loading
        
    Returns:
        str: The iframe ID for the created visualization
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
            layout=widgets.Layout(width="0px", height="0px", visibility="hidden", display="none"),
        )
        bridge.add_class(f"data-bridge-{iframe_id}")
        bridge.add_class(f"data-bridge-{bid}")
        data_bridges[bid] = bridge
        display(bridge)

    # ----------------------------
    # Build communication script for iframe
    # ----------------------------
    debug_log = "console.log('[iframe]', ...args);" if debug else ""
    
    try:
        communication_script_template = _load_js_file("iframe_communication.js")
        communication_script_body = communication_script_template.format(
            iframe_id=json.dumps(iframe_id),
            button_ids_js=button_ids_js,
            initial_data_js=initial_data_js,
            debug_log=debug_log,
        )
    except FileNotFoundError:
        # Fallback: inline the script if file not found
        communication_script_body = _get_inline_communication_script(
            iframe_id, button_ids_js, initial_data_js, debug_log
        )
    
    communication_script = f"<script>\n{communication_script_body}\n</script>"

    # Inject communication script into HTML content
    if "</body>" in html_content.lower():
        idx = html_content.lower().rfind("</body>")
        full_html = html_content[:idx] + communication_script + html_content[idx:]
    else:
        full_html = html_content + communication_script

    payload_b64 = _b64(full_html)

    # ----------------------------
    # Prepare embedding data for chunked loading
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
        
        # Dynamic chunk sizing
        CELLS_PER_CHUNK = chunk_size
        NUM_CHUNKS = max(1, (n + CELLS_PER_CHUNK - 1) // CELLS_PER_CHUNK)
        NUM_CHUNKS = min(NUM_CHUNKS, 50)  # Cap at 50 chunks
        print(f"[Chunked Loading] {n:,} cells → {NUM_CHUNKS} chunks (~{n // NUM_CHUNKS:,} cells each)")
        
        # Chunk assignment
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
        
        # Chunk 0: embed in HTML for instant display
        chunk0_mask = (chunk_assignments == 0)
        chunk0_indices = np.where(chunk0_mask)[0]
        chunk0_binary = _pack_coords_binary(spatial_coords[chunk0_indices], compress=USE_COMPRESSION)
        
        chunk0_umap_binary = None
        chunk0_pca_binary = None
        if umap_coords is not None:
            chunk0_umap_binary = _pack_coords_binary(umap_coords[chunk0_indices], compress=USE_COMPRESSION)
        if pca_coords is not None:
            chunk0_pca_binary = _pack_coords_binary(pca_coords[chunk0_indices], compress=USE_COMPRESSION)
        
        # Sample for minimap
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
        
        # Sample metadata for layout computation
        sample_id_col = initial_data.get("sample_id") if initial_data else None
        sample_meta_js = "null"
        if sample_id_col and sample_id_col in adata.obs.columns:
            samp_arr = adata.obs[sample_id_col].astype(str).values
            unique_samps = list(pd.Series(samp_arr).unique())
            n_samps = len(unique_samps)
            samp_to_idx = {s: i for i, s in enumerate(unique_samps)}
            
            # Per-cell sample index
            cell_sample_ids = np.array([samp_to_idx[s] for s in samp_arr], dtype=np.uint16)
            
            # Per-sample spatial metadata
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
            
            # Chunk0 sample IDs
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
    # Build container HTML with canvas WebGL script
    # ----------------------------
    container_base = _build_container_html(iframe_id, height)
    
    # Build the canvas WebGL script
    debug_log_parent = "console.log('[parent]', ...args);" if debug else ""
    
    try:
        canvas_script_template = _load_js_file("canvas_webgl.js")
        canvas_script_body = canvas_script_template.format(
            iframe_id=json.dumps(iframe_id),
            payload_b64=json.dumps(payload_b64),
            debug_log=debug_log_parent,
            embeds_js=embeds_js,
            sample_meta_js=sample_meta_js,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "canvas_webgl.js not found. Please ensure the js/ directory contains the required files."
        )
    
    container_html = container_base + f"\n<script>\n{canvas_script_body}\n</script>"
    
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
    # Dispatcher script
    # ----------------------------
    try:
        dispatcher_template = _load_js_file("dispatcher.js")
        dispatcher_body = dispatcher_template.format(
            iframe_id=json.dumps(iframe_id),
        )
    except FileNotFoundError:
        dispatcher_body = _get_inline_dispatcher_script(iframe_id)
    
    dispatch_script = f"<script>\n{dispatcher_body}\n</script>"
    display(HTML(dispatch_script))

    output.layout.visibility = "hidden"
    output.layout.height = "0px"
    display(output)

    return iframe_id


def _get_inline_communication_script(iframe_id: str, button_ids_js: str, initial_data_js: str, debug_log: str) -> str:
    """Fallback inline communication script if JS file not found."""
    return f"""
(function() {{
  const iframeId = {json.dumps(iframe_id)};
  const buttonIds = {button_ids_js};
  const initialData = {initial_data_js};
  let pendingRequest = null;

  function log(...args) {{ {debug_log} }}

  window.INITIAL_DATA = initialData;
  window._iframeId = iframeId;

  const skipBridgeButtons = new Set([
    "computeLayoutBtn", "deleteLayoutBtn", "loadLayoutBtn", "saveLayoutBtn",
    "viewportBtn", "chunkBtn", "loadEmbeddingBtn", "__save_obs_column__"
  ]);
  
  buttonIds.forEach((bid) => {{
    if (skipBridgeButtons.has(bid)) return;
    const button = document.getElementById(bid);
    if (!button) return;

    button.addEventListener('click', () => {{
      if (pendingRequest === bid) return;
      pendingRequest = bid;
      button.disabled = true;

      const data = {{}};
      for (let attr of button.attributes) {{
        if (attr.name.startsWith('data-')) {{
          data[attr.name.substring(5)] = attr.value;
        }}
      }}

      const inputs = document.querySelectorAll('[data-for="' + bid + '"]');
      inputs.forEach(input => {{
        const key = input.getAttribute('data-key') || input.id || input.name;
        if (key) data[key] = input.value;
      }});

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
      const data = event.data.data;
      if (pendingRequest) {{
        const button = document.getElementById(pendingRequest);
        if (button) button.disabled = false;
        pendingRequest = null;
      }}
      if (data && data.type === 'sample_meta') {{
        window.parent.postMessage(data, '*');
      }}
      window.dispatchEvent(new CustomEvent('pythonResponse', {{ detail: data }}));
    }}
  }});
}})();
"""


def _get_inline_dispatcher_script(iframe_id: str) -> str:
    """Fallback inline dispatcher script if JS file not found."""
    return f"""
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
    const iframeId2 = parts.slice(2, parts.length - 1).join('_');

    const scopedSelector = '.data-bridge-' + iframeId2 + '.data-bridge-' + buttonId + ' input';
    const scoped = document.querySelector(scopedSelector);
    if (scoped) {{
      scoped.value = value;
      fireWidgetEvents(scoped);
      return true;
    }}

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
"""
