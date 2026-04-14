/**
 * canvas_webgl.js
 * 
 * Main WebGL rendering engine for the scatter plot visualization.
 * Handles:
 * - WebGL context and shader initialization
 * - Point rendering with instancing
 * - Embedding transitions (spatial/umap/pca/layout)
 * - Color mapping (categorical and continuous)
 * - Selection tools (lasso, rectangle, circle, polygon)
 * - Pan/zoom/rotate navigation
 * - Minimap rendering
 * - Progressive chunk loading
 * 
 * Placeholders (replaced via Python .format()):
 *   {iframe_id}      - Unique iframe identifier (JSON-encoded string)
 *   {payload_b64}    - Base64-encoded HTML payload for iframe
 *   {debug_log}      - Debug logging statement
 *   {debug_flag}     - Boolean true/false for debug mode (enables 3D webcam overlay)
 *   {embeds_js}      - JSON object with embedding data
 *   {sample_meta_js} - JSON object with sample metadata (or "null")
 */
(function() {{
  const iframeId = {iframe_id};

  const iframe = document.getElementById({iframe_id});
  const b64 = {payload_b64};
  const html = decodeURIComponent(escape(atob(b64)));

  // ---- write iframe content ----
  const doc = iframe.contentWindow.document;
  doc.open();
  doc.write(html);
  doc.close();

  // ---- request queue for button bridge ----
  window["_requests_" + iframeId] = [];

  function log(...args) {{ {debug_log} }}

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
      // Handle processing overlay messages
      if (event.data.type === "show_processing") {{
        showProcessing(event.data.message);
        return;
      }}
      if (event.data.type === "hide_processing") {{
        hideProcessing();
        return;
      }}
      if (event.data.type === "update_processing_progress") {{
        updateProcessingProgress(event.data.pct, event.data.label);
        return;
      }}
      
      // Direct embedding switch (for custom embeddings that don't need a Python round-trip)
      if (event.data.type === "switch_embedding_direct") {{
        const updatePlot = window["updatePlot_" + iframeId];
        if (updatePlot) updatePlot({{ type: "set_embedding", embedding: event.data.embedding }});
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
          if (event.data.obs.outlineMode !== undefined && _obsOutlineMode !== !!event.data.obs.outlineMode) {{
            _obsOutlineMode = !!event.data.obs.outlineMode; colorsDirty = true;
          }}
        }}
        
        // Sync gex colormap/opacity/vmin/vmax
        if (s && event.data.gex) {{
          const newOp = event.data.gex.opacity != null ? event.data.gex.opacity : 1.0;
          if (s.gex.opacity !== newOp) {{ s.gex.opacity = newOp; colorsDirty = true; }}
          if (event.data.gex.colormap && s.gex.colormap !== event.data.gex.colormap) {{
            s.gex.colormap = event.data.gex.colormap; currentColormap = event.data.gex.colormap; colorsDirty = true;
            // Re-render heatmap with new colormap
            if (_lastHeatmapData) renderHeatmapPanel(_lastHeatmapData);
          }}
          // Sync user vmin/vmax overrides
          const newVmin = event.data.gex.userVmin !== undefined ? event.data.gex.userVmin : s.gex.userVmin;
          const newVmax = event.data.gex.userVmax !== undefined ? event.data.gex.userVmax : s.gex.userVmax;
          if (s.gex.userVmin !== newVmin || s.gex.userVmax !== newVmax) {{
            s.gex.userVmin = newVmin; s.gex.userVmax = newVmax; colorsDirty = true;
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
          s.selectionShape = event.data.selection.shape || null;  // shape type for handles (not drawing mode)
          
          // Restore path to canvas for editing with transform handles
          // Path is stored in DATA coordinates, convert to CANVAS coords
          // NOTE: _selectionTool_ is NOT set here to avoid entering drawing mode.
          // The loaded selection's shape type is stored in _loadedSelShape_ for use
          // by completeSelection() when a loaded selection is edited.
          if (s.selectionPath && s.selectionPath.length > 0) {{
            window["_selectionPathData_" + iframeId] = s.selectionPath.map(p => [...p]);  // Store data coords
            window["_selectionPath_" + iframeId] = pathDataToCanvas(s.selectionPath);  // Convert to canvas coords
            window["_loadedSelShape_" + iframeId] = s.selectionShape || s.selectionTool;  // Shape type for handles/completeSelection
            window["_isDrawing_" + iframeId] = false;  // Not drawing, just editing
            drawSelectionOutline();
          }} else {{
            // Clear canvas selection if no active selection
            window["_selectionPath_" + iframeId] = [];
            window["_selectionPathData_" + iframeId] = null;
            window["_selectionBounds_" + iframeId] = null;
            window["_selectionHandles_" + iframeId] = null;
            window["_loadedSelShape_" + iframeId] = null;
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
      
      // Region group → adata.obs (full indices resolved server-side)
      if (event.data.type === "save_region_group_to_obs" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "saveRegionGroupToObsBtn",
          data: {{
            group_name: event.data.group_name,
            region_names: event.data.region_names,
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

      // 3D stack mode toggle
      if (event.data.type === "set_3d_stack_mode") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}

      // 3D depth strength slider
      if (event.data.type === "set_depth_strength") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}

      // 3D toggles: parallax, mirror, auto-rotate, rotate speed
      if (event.data.type === "toggle_3d_parallax" ||
          event.data.type === "toggle_3d_mirror" ||
          event.data.type === "set_3d_auto_rotate" ||
          event.data.type === "set_3d_rotate_speed") {{
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
      
      // Region tool: route DBSCAN request to Python (inject active embedding)
      if (event.data.type === "run_dbscan" && event.data.iframeId === iframeId) {{
        const isContinuation = !!event.data.continue;
        window["_requests_" + iframeId].push({{
          buttonId: "dbscanBtn",
          data: isContinuation
            ? {{ continue: true }}
            : {{
                column: event.data.column,
                category: event.data.category,
                eps: event.data.eps,
                min_samples: event.data.min_samples,
                embedding: resolveEmbeddingKey()
              }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}

      // Region tool: route alpha shape request to Python (inject active embedding)
      // Send only cluster names — full indices live in adata.uns['_dbscan_tmp'] server-side
      if (event.data.type === "compute_alpha_shapes" && event.data.iframeId === iframeId) {{
        const clusterNames = (event.data.clusters || []).map(c => ({{ name: c.name }}));
        window["_requests_" + iframeId].push({{
          buttonId: "alphaShapeBtn",
          data: {{
            clusters: clusterNames,
            alpha: event.data.alpha,
            embedding: resolveEmbeddingKey()
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Region tool: canvas-only messages (render polygons, clear, opacity)
      if (event.data.type === "show_region_polygons" ||
          event.data.type === "show_dbscan_clusters" ||
          event.data.type === "clear_regions" ||
          event.data.type === "region_fill_opacity" ||
          event.data.type === "region_outline_weight" ||
          event.data.type === "toggle_region_labels" ||
          event.data.type === "toggle_selection_labels" ||
          event.data.type === "update_selection_labels") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
      
      // Manual selection: complete the active selection (point-in-polygon, same as finishing a draw)
      if (event.data.type === "complete_active_selection" && event.data.iframeId === iframeId) {{
        completeSelection(event.data.tool || null);
      }}

      // Manual selection: batch capture for all path-only selections on import
      if (event.data.type === "complete_all_path_selections" && event.data.iframeId === iframeId) {{
        const selections = event.data.selections || [];
        console.log("[BatchCapture] Received", selections.length, "selections, loadedCount:", loadedCount, "embedding:", currentEmbedding);
        if (!selections.length || loadedCount === 0) {{
          console.log("[BatchCapture] Skipping — no selections or no cells loaded yet");
          return;
        }}

        const positions = getEmbeddingPositions(currentEmbedding);
        const rect = panel.getBoundingClientRect();
        const W = rect.width, H = rect.height;
        const meta = METADATA[currentEmbedding] || METADATA.spatial || {{}};
        const minX = meta.minX ?? 0, maxX = meta.maxX ?? 1;
        const minY = meta.minY ?? 0, maxY = meta.maxY ?? 1;
        const pad = 12;
        const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
        const baseScale = Math.min((W - 2*pad) / spanX, (H - 2*pad) / spanY);
        const scale = baseScale * zoom;
        const offX = (W - spanX * scale) / 2 + panX;
        const offY = (H - spanY * scale) / 2 + panY;
        const centerX = W / 2, centerY = H / 2;
        const cos = Math.cos(rotation), sin = Math.sin(rotation);

        // Precompute canvas positions for all loaded cells
        const cellCX = new Float32Array(loadedCount);
        const cellCY = new Float32Array(loadedCount);
        for (let i = 0; i < loadedCount; i++) {{
          let px = offX + (positions[i * 2] - minX) * scale;
          let py = offY + (positions[i * 2 + 1] - minY) * scale;
          const dx = px - centerX, dy = py - centerY;
          cellCX[i] = centerX + (dx * cos - dy * sin);
          cellCY[i] = centerY + (dx * sin + dy * cos);
        }}

        const results = [];
        for (const {{name, path: dataPath, tool}} of selections) {{
          const canvasPath = pathDataToCanvas(dataPath);
          const found = [];
          for (let i = 0; i < loadedCount; i++) {{
            const px = cellCX[i], py = cellCY[i];
            let inside = false;
            if (tool === "lasso" || tool === "polygon") {{
              inside = pointInPolygon(px, py, canvasPath);
            }} else if (tool === "rectangle" && canvasPath.length >= 2) {{
              const [x1, y1] = canvasPath[0], [x2, y2] = canvasPath[1];
              inside = px >= Math.min(x1,x2) && px <= Math.max(x1,x2) && py >= Math.min(y1,y2) && py <= Math.max(y1,y2);
            }} else if (tool === "circle" && canvasPath.length >= 2) {{
              const [cx, cy] = canvasPath[0], [ex, ey] = canvasPath[1];
              const rx = Math.abs(ex - cx);
              const ry = canvasPath.length > 2 ? Math.abs(canvasPath[2][1] - cy) : rx;
              if (rx > 0 && ry > 0) {{
                const ddx = (px - cx) / rx, ddy = (py - cy) / ry;
                inside = (ddx*ddx + ddy*ddy) <= 1;
              }}
            }}
            if (inside) found.push(cellIds[i]);
          }}
          results.push({{ name, indices: found, path: dataPath, tool }});
        }}

        const totalFound = results.reduce((s, r) => s + r.indices.length, 0);
        console.log("[BatchCapture] Done —", results.length, "masks,", totalFound, "total cells found");
        iframe.contentWindow.postMessage({{
          type: "all_selections_captured",
          results
        }}, "*");
      }}

      // Region tool: save masks to adata.uns
      if (event.data.type === "save_region_masks" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "saveRegionMasksBtn",
          data: {{
            payload: event.data.payload,
            embedding: resolveEmbeddingKey()
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Region tool: load masks from adata.uns
      if (event.data.type === "load_region_masks" && event.data.iframeId === iframeId) {{
        if (!event.data.continue) showMaskLoading(event.data.source || "region_masks");
        window["_requests_" + iframeId].push({{
          buttonId: "loadRegionMasksBtn",
          data: {{ source: event.data.source || "region_masks", embedding: resolveEmbeddingKey() }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Manual selection: save masks to adata.uns
      if (event.data.type === "save_manual_masks" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "saveManualMasksBtn",
          data: {{
            payload: event.data.payload,
            embedding: resolveEmbeddingKey()
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Manual selection: load masks from adata.uns
      if (event.data.type === "load_manual_masks" && event.data.iframeId === iframeId) {{
        if (!event.data.continue) showMaskLoading(event.data.source || "manual_masks");
        window["_requests_" + iframeId].push({{
          buttonId: "loadManualMasksBtn",
          data: {{ source: event.data.source || "manual_masks", embedding: resolveEmbeddingKey() }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Manual selection: transform polygon paths to a new embedding via Python
      if (event.data.type === "transform_manual_paths" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "transformManualPathsBtn",
          data: JSON.parse(event.data.payload || "{{}}"),
          type: "button_click",
          iframeId: iframeId
        }});
      }}

      // Region tool: respond with the current embedding key (passes _purpose through so caller can distinguish)
      if (event.data.type === "request_current_embedding" && event.data.iframeId === iframeId) {{
        const embKey = resolveEmbeddingKey();
        const iframe2 = document.getElementById(iframeId);
        if (iframe2 && iframe2.contentWindow) {{
          iframe2.contentWindow.postMessage({{
            type: "current_embedding_response",
            embedding: embKey,
            alpha: event.data.alpha || 0.05,
            _purpose: event.data._purpose || null,
          }}, "*");
        }}
      }}

      // Region tool: rename a mask in all server-side caches
      if (event.data.type === "rename_region_mask" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "renameRegionMaskBtn",
          data: {{
            old_name: event.data.old_name,
            new_name: event.data.new_name
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}

      // Region tool: recapture cells in stored polygon masks using point-in-polygon
      if (event.data.type === "recapture_region_cells" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "recaptureRegionCellsPyBtn",
          data: {{
            payload: event.data.payload
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}

      // Region tool: recompute polygons for current embedding
      if (event.data.type === "recompute_region_polygons" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "recomputeRegionPolygonsBtn",
          data: {{
            payload: event.data.payload
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Heatmap tool: compute bins and gene expression
      if (event.data.type === "compute_heatmap_bins" && event.data.iframeId === iframeId) {{
        window["_requests_" + iframeId].push({{
          buttonId: "computeHeatmapBtn",
          data: {{
            payload: event.data.payload
          }},
          type: "button_click",
          iframeId: iframeId
        }});
      }}
      
      // Heatmap tool: canvas-only messages (ribbon display, bin overlay)
      if (event.data.type === "show_heatmap_ribbon" ||
          event.data.type === "clear_heatmap_ribbon" ||
          event.data.type === "show_heatmap_bins" ||
          event.data.type === "start_heatmap_ribbon" ||
          event.data.type === "show_heatmap_panel" ||
          event.data.type === "hide_heatmap_panel" ||
          event.data.type === "heatmap_result") {{
        const updateFn = window["updatePlot_" + iframeId];
        if (updateFn) updateFn(event.data);
      }}
    }});
  }}

  window["sendToIframe_" + iframeId] = function(data) {{
    // Hide mask loading overlay on completion or error
    if (data && (data.type === "region_masks_loaded" || data.type === "manual_masks_loaded" || data.type === "region_cells_recaptured" || data.type === "manual_paths_transformed" || data.type === "error")) {{
      hideMaskLoading();
    }}
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
  // Custom extra embeddings: key → Float32Array(TOTAL_CELLS * 2)
  const posCustom = {{}};
  (METADATA.customEmbeddings || []).forEach(em => {{
    posCustom[em.key] = new Float32Array(TOTAL_CELLS * 2);
    METADATA[em.key] = {{ minX: em.minX, maxX: em.maxX, minY: em.minY, maxY: em.maxY, count: em.count }};
  }});
  // Per-cell z-layer from 3D embeddings (normalized -1 to +1). Null = no z for that embedding.
  let posUmapZ = METADATA.hasUmapZ ? new Float32Array(TOTAL_CELLS) : null;
  let posPcaZ = METADATA.hasPcaZ ? new Float32Array(TOTAL_CELLS) : null;
  const posCustomZ = {{}};
  (METADATA.customEmbeddingsWithZ || []).forEach(key => {{ posCustomZ[key] = new Float32Array(TOTAL_CELLS); }});
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
  // Annotation cache keyed by obsm key (X_...) for cross-save/import within session
  const savedLayoutAnnotations = {{}};
  let activeLayoutName = null;
  // Returns the actual adata.obsm key for the current embedding.
  // When viewing a layout, currentEmbedding="layout" (JS sentinel) — resolve to the real obsm key.
  function resolveEmbeddingKey() {{
    if (currentEmbedding === "layout" && activeLayoutName) return activeLayoutName;
    return currentEmbedding;
  }}
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
  const maskLoadingOverlay = document.getElementById("mask_loading_" + iframeId);
  
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
  
  // Show/hide mask import loading overlay
  const maskLoadingBar  = document.getElementById("mask_loading_bar_"  + iframeId);
  const maskLoadingText = document.getElementById("mask_loading_text_" + iframeId);
  function showMaskLoading(sourceKey) {{
    if (maskLoadingOverlay) {{
      if (maskLoadingText && sourceKey) {{
        const label = sourceKey.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
        maskLoadingText.textContent = label;
      }}
      if (maskLoadingBar) maskLoadingBar.style.width = "0%";
      maskLoadingOverlay.style.display = "flex";
    }}
  }}
  function hideMaskLoading() {{
    if (maskLoadingOverlay) maskLoadingOverlay.style.display = "none";
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
  
  // Show/hide processing overlay (DBSCAN, alpha shapes, heatmap)
  const processingOverlay = document.getElementById("processing_overlay_" + iframeId);
  const processingText = document.getElementById("processing_text_" + iframeId);
  const processingBar = document.getElementById("processing_bar_" + iframeId);
  const processingBarText = document.getElementById("processing_bar_text_" + iframeId);
  let _processingIndeterminateTimer = null;

  function showProcessing(msg) {{
    if (processingOverlay) {{
      if (processingText) processingText.textContent = msg || "Processing...";
      if (processingBarText) processingBarText.textContent = "";
      processingOverlay.style.display = "flex";

      // Start indeterminate fill: 3% → 75% over ~4s
      // Alpha shape streaming will override this with exact progress
      if (processingBar) {{
        processingBar.style.transition = "none";
        processingBar.style.width = "3%";
        // Let the reset paint, then animate
        requestAnimationFrame(() => {{
          processingBar.style.transition = "width 4s cubic-bezier(0.1, 0.6, 0.4, 1)";
          processingBar.style.width = "75%";
        }});
      }}
      // Clear any stale timer
      if (_processingIndeterminateTimer) {{ clearTimeout(_processingIndeterminateTimer); _processingIndeterminateTimer = null; }}
    }}
  }}

  function updateProcessingProgress(pct, label) {{
    if (processingBar) {{
      processingBar.style.transition = "width 0.25s ease";
      processingBar.style.width = Math.min(100, Math.max(0, pct)) + "%";
    }}
    if (processingBarText && label !== undefined) {{
      processingBarText.textContent = label || "";
    }}
  }}

  function hideProcessing() {{
    if (!processingOverlay) return;
    // Snap to 100% then fade out
    if (processingBar) {{
      processingBar.style.transition = "width 0.15s ease";
      processingBar.style.width = "100%";
    }}
    _processingIndeterminateTimer = setTimeout(() => {{
      processingOverlay.style.display = "none";
      if (processingBar) {{ processingBar.style.width = "3%"; }}
      if (processingBarText) processingBarText.textContent = "";
      _processingIndeterminateTimer = null;
    }}, 200);
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
    const umapZCoords = chunk0.umap_z_binary ? decodeBinaryCoords(chunk0.umap_z_binary, count) : null;
    const pcaCoords = chunk0.pca_binary ? decodeBinaryCoords(chunk0.pca_binary, count) : null;
    const pcaZCoords = chunk0.pca_z_binary ? decodeBinaryCoords(chunk0.pca_z_binary, count) : null;

    // Decode custom embedding coords for chunk0
    const customCoords0 = {{}};
    const customZCoords0 = {{}};
    const customBinaries0 = chunk0.custom_binaries || {{}};
    const customZBinaries0 = chunk0.custom_z_binaries || {{}};
    Object.keys(customBinaries0).forEach(key => {{
      if (customBinaries0[key]) customCoords0[key] = decodeBinaryCoords(customBinaries0[key], count);
    }});
    Object.keys(customZBinaries0).forEach(key => {{
      if (customZBinaries0[key]) customZCoords0[key] = decodeBinaryCoords(customZBinaries0[key], count);
    }});

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
      if (umapZCoords && posUmapZ) posUmapZ[writePos] = umapZCoords[i];
      if (pcaCoords) {{
        posPca[writePos * 2] = pcaCoords[i * 2];
        posPca[writePos * 2 + 1] = pcaCoords[i * 2 + 1];
      }}
      if (pcaZCoords && posPcaZ) posPcaZ[writePos] = pcaZCoords[i];
      Object.keys(customCoords0).forEach(key => {{
        if (posCustom[key]) {{
          posCustom[key][writePos * 2]     = customCoords0[key][i * 2];
          posCustom[key][writePos * 2 + 1] = customCoords0[key][i * 2 + 1];
        }}
        if (customZCoords0[key] && posCustomZ[key]) posCustomZ[key][writePos] = customZCoords0[key][i];
      }});

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
  let chunkRecoveryTimer = null;
  let chunkRecoveryAttempted = false;
  
  function scheduleChunkRecovery() {{
    // Only attempt recovery ONCE after a reconnect stale flush
    if (chunkRecoveryAttempted) return;
    if (chunkRecoveryTimer) clearTimeout(chunkRecoveryTimer);
    chunkRecoveryTimer = setTimeout(() => {{
      chunkRecoveryTimer = null;
      chunkRecoveryAttempted = true;
      if (!isLoadingChunk && CHUNKS_LOADED.size < NUM_CHUNKS) {{
        console.log("[Chunk] Recovery: resuming chunk loading after stale flush");
        requestNextChunk();
      }}
    }}, 2000);
  }}
  
  function processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, count, chunkSids, responseRequestId, customCoordsMap, umapZCoords, pcaZCoords, customZCoordsMap) {{
    const now = Date.now();
    
    // Check if this is a stale response (from an old request)
    if (responseRequestId !== undefined && responseRequestId !== null && 
        lastChunkRequestId !== null && responseRequestId !== lastChunkRequestId) {{
      // Silently ignore stale responses — do NOT re-request immediately
      // as this causes infinite loops when Output widget replays cached responses
      isLoadingChunk = false;
      scheduleChunkRecovery();
      return;
    }}
    
    // Also handle case where response has no requestId (old cached response)
    if ((responseRequestId === undefined || responseRequestId === null) && CHUNKS_LOADED.has(chunkId)) {{
      // Silently ignore old cached responses
      isLoadingChunk = false;
      scheduleChunkRecovery();
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
      setTimeout(() => processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, count, chunkSids, responseRequestId, customCoordsMap, umapZCoords, pcaZCoords, customZCoordsMap), 50);
      return;
    }}
    
    chunkProcessingLock = true;
    CHUNKS_LOADED.add(chunkId);
    isLoadingChunk = false;
    chunkRecoveryAttempted = false;  // Reset: connection is working
    
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
      if (umapZCoords && posUmapZ) posUmapZ[writePos] = umapZCoords[i];
      if (pcaCoords) {{
        posPca[writePos * 2] = pcaCoords[i * 2];
        posPca[writePos * 2 + 1] = pcaCoords[i * 2 + 1];
      }}
      if (pcaZCoords && posPcaZ) posPcaZ[writePos] = pcaZCoords[i];
      if (customCoordsMap) {{
        Object.keys(customCoordsMap).forEach(key => {{
          if (posCustom[key] && customCoordsMap[key]) {{
            posCustom[key][writePos * 2]     = customCoordsMap[key][i * 2];
            posCustom[key][writePos * 2 + 1] = customCoordsMap[key][i * 2 + 1];
          }}
          if (customZCoordsMap && customZCoordsMap[key] && posCustomZ[key]) {{
            posCustomZ[key][writePos] = customZCoordsMap[key][i];
          }}
        }});
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
    if (posCustom[embeddingName]) return posCustom[embeddingName];
    return posSpatial;
  }}

  // Get z-layer array for an embedding (null if 2D only)
  function getEmbeddingZ(embeddingName) {{
    if (embeddingName === "umap") return posUmapZ;
    if (embeddingName === "pca") return posPcaZ;
    if (posCustomZ[embeddingName]) return posCustomZ[embeddingName];
    return null;
  }}

  // Update _embeddingHas3D based on active embedding name
  function updateEmbedding3DFlag(embeddingName) {{
    _embeddingHas3D = !!getEmbeddingZ(embeddingName);
    markGPUDirty();
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
      colormap: "viridis",
      userVmin: null,
      userVmax: null
    }},
    label: "Embedding: spatial",
    pointSize: 1.1,
    selectionIndices: null  // array of selected point indices (acts as mask)
  }};

  // Region overlay state
  let regionPolygons = [];    // array of {{name, polygons: [[[x,y],...]], color}}
  let regionFillOpacity = 0.1;
  let regionOutlineWeight = 2;
  let showRegionLabels = true;
  let showSelectionLabels = true;
  let selectionLabels = []; // [{{name, cx, cy, color}}]
  let regionColor = null;     // color from the cell type palette

  // --- Heatmap ribbon state ---
  let heatmapRibbon = null;    // {{start, end, controlPoints, widthStart, widthMid, widthEnd}}
  let heatmapBins = null;      // array of {{bin, quad, cell_count, ...}} from Python
  let heatmapVisible = false;  // whether to show ribbon overlay
  let heatmapMode = null;      // null, "placing_start", "placing_end", "editing"
  let heatmapDragging = null;  // which handle is being dragged

  // --- Dark mode state ---
  let _darkMode = true;  // true = dark (black bg) by default
  let _obsOutlineMode = false;

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
      
      // Force redraw everything to ensure overlays are up to date
      drawSampleLabels();
      drawRegionPolygons();
      drawSelectionLabels();
      drawHeatmapRibbon();
      
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

  // Dark/light mode toggle button
  const darkmodeBtn = document.getElementById("darkmode_btn_" + iframeId);
  const darkmodeSun = document.getElementById("darkmode_sun_" + iframeId);
  const darkmodeMoon = document.getElementById("darkmode_moon_" + iframeId);
  const plotPanel = document.getElementById("plot_panel_" + iframeId);
  function applyDarkMode() {{
    if (plotPanel) plotPanel.style.background = _darkMode ? "#000" : "inherit";
    if (darkmodeSun) darkmodeSun.style.display = _darkMode ? "none" : "";
    if (darkmodeMoon) darkmodeMoon.style.display = _darkMode ? "" : "none";
    if (darkmodeBtn) {{
      darkmodeBtn.style.background = _darkMode ? "rgba(30,30,30,0.9)" : "rgba(255,255,255,0.9)";
      darkmodeBtn.style.borderColor = _darkMode ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.12)";
    }}
  }}

  if (darkmodeBtn) {{
    darkmodeBtn.addEventListener("click", () => {{
      _darkMode = !_darkMode;
      applyDarkMode();
      markGPUDirty();
      draw();
    }});
  }}

  // Apply dark mode immediately on load
  applyDarkMode();

  // ----------------------------
  // 3D parallax mode + webcam face tracking
  // ----------------------------
  const DEBUG_3D = {debug_flag};
  let _3dMode = false;
  let _headX = 0;  // normalized -1..1, updated each frame by face tracking
  let _headY = 0;
  let _webcamStream = null;
  let _faceMeshReady = false;
  let _faceMesh = null;
  let _webcamVideo = null;
  let _trackingRafId = null;
  let _debugCanvas = null;
  let _debugCtx = null;
  let _layerTransitionStart = -Infinity; // timestamp of last mask change (for z-layer anim)
  let _depthStrength = 0.1; // shared depth scale for parallax + orbital separation

  // Orbital rotation mode (press 'o')
  let _orbitalX = 0;  // rotation around X axis (forward/back tilt), radians
  let _orbitalY = 0;  // rotation around Y axis (left/right tilt), radians
  let _orbitalMode = false;
  let _3dStackMode = false;    // true = each obs category at its own z-layer (stack), false = mask/active split
  let _3dMirror = false;       // true = draw mirror/floor reflection when 3D mode on
  let _autoRotate = false;     // true = slowly spin _orbitalY counter-clockwise via rAF
  let _autoRotateRafId = null;
  let _autoRotateSpeed = 0.006; // radians per frame
  let _embeddingHas3D = false; // true = current embedding has a 3rd dimension for z-layer
  const _MIRROR_PLANE_Y = -0.70;  // fixed clip-space Y for floor/mirror plane (doesn't move with cells)
  let _lastOrbitalKeyPress = 0;

  const threedBtn = document.getElementById("threed_btn_" + iframeId);


  function stopWebcam() {{
    if (_trackingRafId) {{ cancelAnimationFrame(_trackingRafId); _trackingRafId = null; }}
    if (_webcamStream) {{ _webcamStream.getTracks().forEach(t => t.stop()); _webcamStream = null; }}
    if (_webcamVideo) {{ _webcamVideo.srcObject = null; _webcamVideo = null; }}
    if (_debugCanvas) {{ _debugCanvas.remove(); _debugCanvas = null; _debugCtx = null; }}
    _faceMeshReady = false;
    _headX = 0;
    _headY = 0;
  }}

  function startFaceTracking() {{
    // Dynamically load MediaPipe Camera Utils + FaceMesh from CDN
    function loadScript(src) {{
      return new Promise((resolve, reject) => {{
        if (document.querySelector('script[src="' + src + '"]')) {{ resolve(); return; }}
        const s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      }});
    }}

    const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/";

    Promise.all([
      loadScript(CDN + "face_mesh.js"),
    ]).then(() => {{
      // Create hidden video element for webcam feed
      _webcamVideo = document.createElement("video");
      _webcamVideo.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;";
      _webcamVideo.setAttribute("playsinline", "");
      document.body.appendChild(_webcamVideo);

      navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user", width: 320, height: 240 }} }})
        .then(stream => {{
          _webcamStream = stream;
          _webcamVideo.srcObject = stream;
          _webcamVideo.play();

          // Debug overlay: floating canvas showing webcam + mesh landmarks
          if (DEBUG_3D) {{
            _debugCanvas = document.createElement("canvas");
            _debugCanvas.width = 240;
            _debugCanvas.height = 180;
            _debugCanvas.style.cssText = "position:absolute;bottom:10px;right:10px;z-index:200;" +
              "border:2px solid white;border-radius:4px;opacity:0.85;pointer-events:none;";
            panel.appendChild(_debugCanvas);
            _debugCtx = _debugCanvas.getContext("2d");
          }}

          _faceMesh = new FaceMesh({{ locateFile: (file) => CDN + file }});
          _faceMesh.setOptions({{
            maxNumFaces: 1,
            refineLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
          }});
          _faceMesh.onResults(results => {{
            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {{
              // Landmark 1 = nose tip; coords are 0..1 relative to video frame
              const nose = results.multiFaceLandmarks[0][1];
              // Center around 0.5, flip X so moving left shifts view left
              _headX = -(nose.x - 0.5) * 2;
              _headY =  (nose.y - 0.5) * 2;
            }}

            // Draw debug overlay: white mask + landmarks
            if (DEBUG_3D && _debugCtx && _debugCanvas) {{
              const dw = _debugCanvas.width;
              const dh = _debugCanvas.height;
              _debugCtx.fillStyle = "rgba(0,0,0,0.75)";
              _debugCtx.fillRect(0, 0, dw, dh);
              if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {{
                const lms = results.multiFaceLandmarks[0];
                _debugCtx.fillStyle = "rgba(255,255,255,0.5)";
                for (let i = 0; i < lms.length; i++) {{
                  const lx = (1 - lms[i].x) * dw;
                  const ly = lms[i].y * dh;
                  _debugCtx.beginPath();
                  _debugCtx.arc(lx, ly, 1.5, 0, Math.PI * 2);
                  _debugCtx.fill();
                }}
                // Nose tip in white
                const nx = (1 - lms[1].x) * dw;
                const ny = lms[1].y * dh;
                _debugCtx.fillStyle = "white";
                _debugCtx.beginPath();
                _debugCtx.arc(nx, ny, 5, 0, Math.PI * 2);
                _debugCtx.fill();
                // X/Y readout
                _debugCtx.fillStyle = "white";
                _debugCtx.font = "11px monospace";
                _debugCtx.fillText("X:" + _headX.toFixed(2) + " Y:" + _headY.toFixed(2), 6, dh - 8);
              }}
            }}
          }});

          _faceMeshReady = true;

          // Detection loop: send frames to FaceMesh each animation frame
          async function detectLoop() {{
            if (!_3dMode || !_faceMeshReady) return;
            if (_webcamVideo.readyState >= 2) {{
              await _faceMesh.send({{ image: _webcamVideo }});
            }}
            draw();  // redraw with updated head position
            _trackingRafId = requestAnimationFrame(detectLoop);
          }}
          detectLoop();
        }})
        .catch(err => {{
          console.warn("[3D] Webcam denied or unavailable:", err);
          // Fall back: disable 3D mode silently
          _3dMode = false;
          if (threedBtn) {{
            threedBtn.classList.remove("active");
            threedBtn.title = "Toggle 3D parallax mode";
          }}
          stopWebcam();
        }});
    }}).catch(err => {{
      console.warn("[3D] Failed to load MediaPipe:", err);
    }});
  }}

  if (threedBtn) {{
    threedBtn.addEventListener("click", () => {{
      _3dMode = !_3dMode;
      if (_3dMode) {{
        threedBtn.classList.add("active");
        threedBtn.title = "3D mode ON — click to disable";
        startFaceTracking();
      }} else {{
        threedBtn.classList.remove("active");
        threedBtn.title = "Toggle 3D parallax mode";
        stopWebcam();
        draw();
      }}
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

      // Decode custom embedding coords
      const customCoordsMap = {{}};
      const chunkCustomBinaries = payload.custom_binaries || {{}};
      Object.keys(chunkCustomBinaries).forEach(key => {{
        if (chunkCustomBinaries[key]) customCoordsMap[key] = decodeBinaryCoords(chunkCustomBinaries[key], payload.count);
      }});
      // Decode z coords for 3D embeddings
      const chunkUmapZ = payload.umap_z_binary ? decodeBinaryCoords(payload.umap_z_binary, payload.count) : null;
      const chunkPcaZ = payload.pca_z_binary ? decodeBinaryCoords(payload.pca_z_binary, payload.count) : null;
      const customZCoordsMap = {{}};
      Object.keys(payload.custom_z_binaries || {{}}).forEach(key => {{
        if (payload.custom_z_binaries[key]) customZCoordsMap[key] = decodeBinaryCoords(payload.custom_z_binaries[key], payload.count);
      }});

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
      processChunkData(chunkId, indices, spatialCoords, umapCoords, pcaCoords, payload.count, chunkSids, payload.requestId, customCoordsMap, chunkUmapZ, chunkPcaZ, customZCoordsMap);
      
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

    // 3D stack mode toggle
    if (payload.type === "set_3d_stack_mode") {{
      _3dStackMode = !!payload.stackMode;
      markGPUDirty();
      return;
    }}

    // 3D depth strength
    if (payload.type === "set_depth_strength") {{
      _depthStrength = payload.value;
      _layerTransitionStart = performance.now();
      draw();
      return;
    }}

    // 3D Parallax toggle (replaces viewport Px button)
    if (payload.type === "toggle_3d_parallax") {{
      _3dMode = !!payload.enabled;
      if (_3dMode) {{
        startFaceTracking();
      }} else {{
        stopWebcam();
        draw();
      }}
      return;
    }}

    // 3D Mirror toggle
    if (payload.type === "toggle_3d_mirror") {{
      _3dMirror = !!payload.enabled;
      draw();
      return;
    }}

    // 3D Auto-rotate speed
    if (payload.type === "set_3d_rotate_speed") {{
      _autoRotateSpeed = payload.value;
      return;
    }}

    // 3D Auto-rotate: slowly spin _orbitalY counter-clockwise
    if (payload.type === "set_3d_auto_rotate") {{
      _autoRotate = !!payload.enabled;
      if (_autoRotate) {{
        if (!_orbitalMode) {{
          _orbitalMode = true;
          canvas.style.cursor = "move";
        }}
        function autoRotateStep() {{
          if (!_autoRotate) return;
          _orbitalY -= _autoRotateSpeed;
          draw();
          _autoRotateRafId = requestAnimationFrame(autoRotateStep);
        }}
        if (_autoRotateRafId) cancelAnimationFrame(_autoRotateRafId);
        _autoRotateRafId = requestAnimationFrame(autoRotateStep);
      }} else {{
        if (_autoRotateRafId) {{ cancelAnimationFrame(_autoRotateRafId); _autoRotateRafId = null; }}
      }}
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

      // Resolve the layout name (obsm key format, e.g. "X_layout_1")
      const importName = payload.layout_name || "imported";

      // Restore annotation info: prefer Python's stored info (adata.uns), fallback to JS session cache
      const _pyInfo = (payload.layout_info && typeof payload.layout_info === "object") ? payload.layout_info : {{}};
      const _jsInfo = savedLayoutAnnotations[importName] || {{}};
      const _annotSrc = (Array.isArray(_pyInfo.group_labels) && _pyInfo.group_labels.length > 0) ? _pyInfo
                      : (Array.isArray(_jsInfo.group_labels) && _jsInfo.group_labels.length > 0) ? _jsInfo
                      : null;
      if (_annotSrc) {{
        layoutGroupLabels = _annotSrc.group_labels.map(g => ({{...g}}));
        layoutColLabels = Array.isArray(_annotSrc.col_labels) ? _annotSrc.col_labels.map(c => ({{...c}})) : [];
        layoutRowLabels = Array.isArray(_annotSrc.row_labels) ? _annotSrc.row_labels.map(r => ({{...r}})) : [];
        layoutAxisInfo = _annotSrc.axis_info ? {{..._annotSrc.axis_info}} : null;
        layoutParams = _annotSrc.params ? {{..._annotSrc.params}} : null;
      }}

      // Save to in-memory savedLayouts so it appears in the layout dropdown
      savedLayouts[importName] = {{
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
      activeLayoutName = importName;

      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "layout_saved",
          name: importName,
          all_names: Object.keys(savedLayouts),
        }}, "*");
        iframeEl.contentWindow.postMessage({{
          type: "layout_applied",
          sample_labels: layoutSampleLabels,
          layout_name: importName,
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
        zoom = 1.0; panX = 0; panY = 0; rotation = 0; _orbitalX = 0; _orbitalY = 0;
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
      // Also cache annotations under the obsm key for import lookup
      const obsmKey2 = name.startsWith("X_") ? name : ("X_" + name);
      savedLayoutAnnotations[obsmKey2] = {{
        group_labels: layoutGroupLabels.map(g => ({{...g}})),
        col_labels: layoutColLabels.map(c => ({{...c}})),
        row_labels: layoutRowLabels.map(r => ({{...r}})),
        axis_info: layoutAxisInfo ? {{...layoutAxisInfo}} : null,
        params: layoutParams ? {{...layoutParams}} : null,
      }};
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
        zoom = 1.0; panX = 0; panY = 0; rotation = 0; _orbitalX = 0; _orbitalY = 0;
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
      if (activeLayoutName === name) {{
        activeLayoutName = null;
        // Switch canvas back to spatial since the active layout was deleted
        currentEmbedding = "spatial";
        markGPUDirty();
        draw();
      }}

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

      // Bundle annotation info so Python can persist it in adata.uns
      const annotInfo = {{
        group_labels: src ? (src.groupLabels || []) : layoutGroupLabels.map(g => ({{...g}})),
        col_labels: src ? (src.colLabels || []) : layoutColLabels.map(c => ({{...c}})),
        row_labels: src ? (src.rowLabels || []) : layoutRowLabels.map(r => ({{...r}})),
        axis_info: src ? src.axisInfo : layoutAxisInfo,
        params: src ? src.params : layoutParams,
      }};

      // Cache annotations by obsm key for same-session import fallback
      const obsmKey = name.startsWith("X_") ? name : ("X_" + name);
      savedLayoutAnnotations[obsmKey] = annotInfo;

      window["_requests_" + iframeId].push({{
        buttonId: "obsmBtn",
        data: {{
          name: name,
          centroids_b64: compressedB64,
          sample_labels: sampleLabels,
          n_samples: nSamples,
          layout_info: JSON.stringify(annotInfo),
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

      updateEmbedding3DFlag(name);

      // Don't animate if we're still loading or if animation already in progress
      if (!isFullyLoaded || isAnimating) {{
        currentEmbedding = name;
        zoom = 1.0;
        panX = 0;
        panY = 0;
        rotation = 0;
        translateRegionPolygons();
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
      const iframeEl = document.getElementById(iframeId);
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

    // Region tool: store and render DBSCAN cluster visualization
    if (payload.type === "show_dbscan_clusters") {{
      // Store cluster centroids for labeling on canvas
      window["_dbscanCentroids_" + iframeId] = (payload.clusters || []).map(c => ({{
        name: c.name,
        cx: c.centroid_x,
        cy: c.centroid_y,
        count: c.count
      }}));
      console.log("[Regions] DBSCAN clusters received:", payload.clusters?.length,
                  "centroids stored for labeling");
      draw();
      return;
    }}

    // Region tool: store alpha shape polygons for overlay rendering
    if (payload.type === "show_region_polygons") {{
      // Update mask import progress bar if this is part of a progressive load
      if (payload.mask_progress) {{
        const {{index, total}} = payload.mask_progress;
        const pct = total > 0 ? Math.round((index / total) * 100) : 0;
        if (maskLoadingBar) maskLoadingBar.style.width = pct + "%";
        if (maskLoadingText) maskLoadingText.textContent = index + " / " + total + " masks";
        if (index >= total) hideMaskLoading();
      }}
      regionPolygons = (payload.regions || []).map(r => ({{
        name: r.name,
        polygons: r.polygons,
        centroid_x: r.centroid_x,
        centroid_y: r.centroid_y,
        color: r.color || null,
        indices: r.indices || []
      }}));
      regionFillOpacity = payload.fillOpacity ?? 0.1;

      // Fallback color if none provided per-region
      if (payload.color) {{
        regionColor = payload.color;
      }}

      // Only translate if NOT already positioned by Python (live DBSCAN only).
      // For loaded-from-uns masks, Python has already translated to current embedding,
      // so JS translate would double-shift using a biased (partially-loaded) sample.
      if (!payload.already_positioned) {{
        translateRegionPolygons();
      }}

      console.log(`[Regions] Rendering ${{regionPolygons.length}} alpha shape regions`);
      draw();
      return;
    }}

    // Region tool: clear all region overlays
    if (payload.type === "clear_regions") {{
      regionPolygons = [];
      regionColor = null;
      window["_dbscanCentroids_" + iframeId] = null;
      draw();
      return;
    }}

    // Region tool: update fill opacity
    if (payload.type === "region_fill_opacity") {{
      regionFillOpacity = payload.opacity ?? 0.1;
      draw();
      return;
    }}
    
    // Region tool: update outline weight
    if (payload.type === "region_outline_weight") {{
      regionOutlineWeight = payload.weight ?? 2;
      draw();
      return;
    }}
    
    // Region tool: toggle region name labels
    if (payload.type === "toggle_region_labels") {{
      showRegionLabels = payload.show !== false;
      draw();
      return;
    }}
    
    // Selection tool: toggle selection labels
    if (payload.type === "toggle_selection_labels") {{
      showSelectionLabels = payload.show !== false;
      draw();
      return;
    }}
    
    // Selection tool: update selection labels (centroids + names)
    if (payload.type === "update_selection_labels") {{
      // Compute centroids from indices using position data
      const labels = payload.labels || [];
      selectionLabels = [];
      const posArr = getEmbeddingPositions(currentEmbedding);
      if (posArr && posArr.length > 0) {{
        labels.forEach(lbl => {{
          const indices = lbl.indices || [];
          if (indices.length === 0) return;
          let sumX = 0, sumY = 0, count = 0;
          for (let i = 0; i < indices.length; i++) {{
            const cellId = indices[i];
            const writePos = cellIdToIndex.get(cellId);
            if (writePos !== undefined && writePos * 2 + 1 < posArr.length) {{
              sumX += posArr[writePos * 2];
              sumY += posArr[writePos * 2 + 1];
              count++;
            }}
          }}
          if (count > 0) {{
            selectionLabels.push({{
              name: lbl.name,
              cx: sumX / count,
              cy: sumY / count,
              color: lbl.color || "rgba(245,158,11,0.85)"
            }});
          }}
        }});
      }}
      draw();
      return;
    }}
    
    // Heatmap tool: show ribbon overlay
    if (payload.type === "show_heatmap_ribbon") {{
      heatmapRibbon = payload.ribbon || null;
      heatmapVisible = true;
      if (payload.editing) {{
        heatmapMode = "editing";
        window["_selectionTool_" + iframeId] = "ribbon";
      }} else {{
        heatmapMode = null;
        // Don't change tool — leave whatever was active
      }}
      draw();
      return;
    }}
    
    // Heatmap tool: start ribbon placement mode
    if (payload.type === "start_heatmap_ribbon") {{
      heatmapMode = "placing_start";
      heatmapRibbon = null;
      heatmapBins = null;
      heatmapVisible = true;
      // Set selection tool to "ribbon" so mouse handlers know
      window["_selectionTool_" + iframeId] = "ribbon";
      draw();
      return;
    }}
    
    // Heatmap tool: clear ribbon
    if (payload.type === "clear_heatmap_ribbon") {{
      heatmapRibbon = null;
      heatmapBins = null;
      heatmapVisible = false;
      heatmapMode = null;
      // Clear tool if it was ribbon
      if (window["_selectionTool_" + iframeId] === "ribbon") {{
        window["_selectionTool_" + iframeId] = null;
      }}
      draw();
      return;
    }}
    
    // Heatmap tool: show bin overlays from Python result
    if (payload.type === "show_heatmap_bins") {{
      heatmapBins = payload.bins || null;
      draw();
      return;
    }}
    
    // Heatmap tool: render heatmap panel from Python result
    if (payload.type === "heatmap_result") {{
      heatmapBins = payload.bins || null;
      renderHeatmapPanel(payload);
      draw();
      // Also forward to iframe for state tracking
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{ type: "heatmap_result_ack" }}, "*");
      }}
      return;
    }}
    
    // Heatmap tool: show/hide panel
    if (payload.type === "show_heatmap_panel") {{
      const panelEl = document.getElementById("heatmap_panel_" + iframeId);
      if (panelEl) panelEl.style.display = "block";
      return;
    }}
    if (payload.type === "hide_heatmap_panel") {{
      const panelEl = document.getElementById("heatmap_panel_" + iframeId);
      if (panelEl) panelEl.style.display = "none";
      return;
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
  let _mmScale = 1, _mmOffX = 0, _mmOffY = 0;

  // Label overlay for sample names (2D canvas on top of WebGL)
  const labelOverlay = document.getElementById("label_overlay_" + iframeId);
  const labelCtx = labelOverlay ? labelOverlay.getContext("2d") : null;

  // ----------------------------
  // WebGL Shaders
  // ----------------------------
  const vertexShaderSource = `
    attribute vec2 a_position;
    attribute vec3 a_color;
    attribute vec3 a_outlineColor;
    attribute float a_sizeScale;
    attribute float a_zLayer;

    uniform mat3 u_matrix;
    uniform float u_pointSize;
    uniform float u_defaultPointSize;
    uniform float u_headX;
    uniform float u_headY;
    uniform float u_layerStrength;
    uniform float u_rotX;
    uniform float u_rotY;
    uniform vec2 u_centroid;
    uniform float u_focalLength;
    uniform float u_orbitalZSep;
    uniform float u_reflectMode;
    uniform float u_reflectY;

    varying vec3 v_color;
    varying vec3 v_outlineColor;

    void main() {{
      // Active vs masked layer (used for both 3D separation and clip-space parallax)
      float zLayer = a_zLayer;

      // 3D orbital rotation around data centroid
      // Active cells placed slightly toward viewer (negative z), masked away (+z)
      vec2 rel = a_position - u_centroid;
      float x3 = rel.x;
      float y3 = rel.y;
      float z3 = -zLayer * u_orbitalZSep;

      // Rotate around Y axis (left/right tilt)
      float cosY = cos(u_rotY);
      float sinY = sin(u_rotY);
      float x3r = x3 * cosY + z3 * sinY;
      float z3r = -x3 * sinY + z3 * cosY;

      // Rotate around X axis (forward/back tilt)
      float cosX = cos(u_rotX);
      float sinX = sin(u_rotX);
      float y3r = y3 * cosX - z3r * sinX;
      float z3rr = y3 * sinX + z3r * cosX;

      // Perspective projection (depth scales size + position)
      float depth = (u_focalLength > 0.0) ? u_focalLength / (u_focalLength + z3rr) : 1.0;
      vec2 projected = vec2(x3r * depth, y3r * depth) + u_centroid;

      // Apply 2D view matrix
      vec3 pos = u_matrix * vec3(projected, 1.0);

      // Z-layer clip-space parallax (head tracking)
      pos.x += u_headX * zLayer * u_layerStrength;
      pos.y -= u_headY * zLayer * u_layerStrength;

      // Mirror reflection: compressed downward (floor perspective, not true mirror)
      // compressFactor << 1 squishes the reflection into a narrow band below the plane
      float compressFactor = 0.3;
      float reflected = u_reflectY - (pos.y - u_reflectY) * compressFactor;
      pos.y = mix(pos.y, reflected, u_reflectMode);

      gl_Position = vec4(pos.xy, 0.0, 1.0);
      gl_PointSize = mix(u_defaultPointSize, u_pointSize, a_sizeScale) * depth;
      v_color = a_color;
      v_outlineColor = a_outlineColor;
    }}
  `;
  
  const fragmentShaderSource = `
    precision mediump float;
    varying vec3 v_color;
    varying vec3 v_outlineColor;
    uniform float u_opacity;
    uniform float u_outlineMode;
    void main() {{
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;
      if (u_outlineMode > 0.5) {{
        float innerR = 0.36;
        if (dist < innerR) {{
          float a = 1.0 - smoothstep(innerR - 0.06, innerR, dist);
          gl_FragColor = vec4(v_color, a * u_opacity);
        }} else {{
          float a = 1.0 - smoothstep(0.45, 0.5, dist);
          gl_FragColor = vec4(v_outlineColor, a * u_opacity);
        }}
      }} else {{
        float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
        gl_FragColor = vec4(v_color, alpha * u_opacity);
      }}
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
  const u_outlineMode = gl.getUniformLocation(program, "u_outlineMode");
  const a_outlineColorLoc = gl.getAttribLocation(program, "a_outlineColor");
  const a_sizeScaleLoc = gl.getAttribLocation(program, "a_sizeScale");
  const a_zLayerLoc = gl.getAttribLocation(program, "a_zLayer");
  const u_defaultPointSize = gl.getUniformLocation(program, "u_defaultPointSize");
  const u_headX = gl.getUniformLocation(program, "u_headX");
  const u_headY = gl.getUniformLocation(program, "u_headY");
  const u_layerStrength = gl.getUniformLocation(program, "u_layerStrength");
  const u_rotX = gl.getUniformLocation(program, "u_rotX");
  const u_rotY = gl.getUniformLocation(program, "u_rotY");
  const u_centroid = gl.getUniformLocation(program, "u_centroid");
  const u_focalLength = gl.getUniformLocation(program, "u_focalLength");
  const u_orbitalZSep = gl.getUniformLocation(program, "u_orbitalZSep");
  const u_reflectMode = gl.getUniformLocation(program, "u_reflectMode");
  const u_reflectY = gl.getUniformLocation(program, "u_reflectY");
  gl.enableVertexAttribArray(a_outlineColorLoc);
  gl.enableVertexAttribArray(a_sizeScaleLoc);
  gl.enableVertexAttribArray(a_zLayerLoc);

  // Create buffers
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  const outlineColorBuffer = gl.createBuffer();
  const sizeScaleBuffer = gl.createBuffer();
  const zLayerBuffer = gl.createBuffer();

  // GPU data tracking
  let gpuPointCount = 0;
  let gpuDataDirty = true;  // Flag to rebuild GPU buffers
  let _hasMask = false;     // True when obs mask is active (some categories disabled)
  
  // Mark GPU data as needing rebuild
  function markGPUDirty() {{
    gpuDataDirty = true;
    if (_3dMode) {{
      _layerTransitionStart = performance.now();
      // Drive animation for the transition duration (detect loop handles it when tracking active)
      (function animLoop() {{
        draw();
        if (performance.now() - _layerTransitionStart < 600) requestAnimationFrame(animLoop);
      }})();
    }}
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

  let cachedPanelW = 0;
  let cachedPanelH = 0;
  
  let cachedCanvasW = 0;
  let cachedCanvasH = 0;
  
  // Lightweight: refresh cached dimensions, returns true if changed
  function refreshPanelDimensions() {{
    const panelRect = panel.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    cachedPanelW = panelRect.width;
    cachedPanelH = panelRect.height;
    const oldCW = cachedCanvasW;
    const oldCH = cachedCanvasH;
    cachedCanvasW = canvasRect.width;
    cachedCanvasH = canvasRect.height;
    const changed = (oldCW !== cachedCanvasW || oldCH !== cachedCanvasH);
    if (changed && cachedPanelW > 0 && cachedPanelH > 0) {{
      const dpr = window.devicePixelRatio || 1;
      const newW = Math.max(1, Math.floor(cachedPanelW * dpr));
      const newH = Math.max(1, Math.floor(cachedPanelH * dpr));
      if (canvas.width !== newW || canvas.height !== newH) {{
        canvas.width = newW;
        canvas.height = newH;
        gl.viewport(0, 0, canvas.width, canvas.height);
        if (labelOverlay) {{
          labelOverlay.width = canvas.width;
          labelOverlay.height = canvas.height;
          labelOverlay.style.width = cachedPanelW + "px";
          labelOverlay.style.height = cachedPanelH + "px";
        }}
        markGPUDirty();
        draw();
      }}
    }}
    return changed;
  }}
  
  function resizeCanvas() {{
    refreshPanelDimensions();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(cachedPanelW * dpr));
    canvas.height = Math.max(1, Math.floor(cachedPanelH * dpr));
    // Set CSS size explicitly to prevent mismatch with pixel buffer
    canvas.style.width = cachedPanelW + "px";
    canvas.style.height = cachedPanelH + "px";
    gl.viewport(0, 0, canvas.width, canvas.height);
    // Resize label overlay to match exactly
    if (labelOverlay) {{
      labelOverlay.width = canvas.width;
      labelOverlay.height = canvas.height;
      labelOverlay.style.width = cachedPanelW + "px";
      labelOverlay.style.height = cachedPanelH + "px";
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
  
  // Draw region polygon overlays (dashed outlines + optional fill)
  function drawRegionPolygons() {{
    if (!regionPolygons || regionPolygons.length === 0) return;
    if (!labelCtx || !labelOverlay) return;
    
    const dpr = window.devicePixelRatio || 1;
    
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    
    for (const region of regionPolygons) {{
      // Use per-region color, or fall back to global regionColor, or teal
      const baseColor = region.color || regionColor || "#14b8a6";
      
      for (const polygon of region.polygons) {{
        if (!polygon || polygon.length < 3) continue;
        
        // Convert data coordinates to canvas coordinates
        const canvasPoints = polygon.map(([dx, dy]) => dataToCanvas(dx, dy));
        
        // Draw filled polygon (optional, based on opacity)
        if (regionFillOpacity > 0.01) {{
          labelCtx.save();
          labelCtx.globalAlpha = regionFillOpacity;
          labelCtx.fillStyle = baseColor;
          labelCtx.beginPath();
          labelCtx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
          for (let i = 1; i < canvasPoints.length; i++) {{
            labelCtx.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
          }}
          labelCtx.closePath();
          labelCtx.fill();
          labelCtx.restore();
        }}
        
        // Draw dashed outline
        labelCtx.strokeStyle = baseColor;
        labelCtx.lineWidth = regionOutlineWeight;
        labelCtx.setLineDash([6, 4]);
        labelCtx.beginPath();
        labelCtx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
        for (let i = 1; i < canvasPoints.length; i++) {{
          labelCtx.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
        }}
        labelCtx.closePath();
        labelCtx.stroke();
      }}
      
      // Draw region name label at centroid (if labels are enabled)
      if (showRegionLabels && region.centroid_x != null && region.centroid_y != null) {{
        const [cx, cy] = dataToCanvas(region.centroid_x, region.centroid_y);
        labelCtx.font = "bold 10px ui-monospace, monospace";
        labelCtx.textAlign = "center";
        labelCtx.textBaseline = "middle";
        
        const text = region.name;
        const tw = labelCtx.measureText(text).width + 6;
        
        labelCtx.save();
        labelCtx.globalAlpha = 0.7;
        labelCtx.fillStyle = baseColor;
        labelCtx.beginPath();
        labelCtx.roundRect(cx - tw/2, cy - 7, tw, 14, 3);
        labelCtx.fill();
        labelCtx.restore();
        
        labelCtx.fillStyle = "#fff";
        labelCtx.fillText(text, cx, cy);
      }}
    }}
    
    labelCtx.setLineDash([]);
    labelCtx.restore();
    
    // Also draw DBSCAN cluster centroid labels (shown after DBSCAN, before alpha shapes)
    const dbscanCentroids = window["_dbscanCentroids_" + iframeId];
    if (showRegionLabels && dbscanCentroids && dbscanCentroids.length > 0 && regionPolygons.length === 0) {{
      // Only show DBSCAN labels when no alpha shapes are rendered yet
      labelCtx.save();
      labelCtx.scale(dpr, dpr);
      labelCtx.font = "bold 10px ui-monospace, monospace";
      labelCtx.textAlign = "center";
      labelCtx.textBaseline = "middle";
      
      dbscanCentroids.forEach((c, i) => {{
        if (c.cx == null || c.cy == null) return;
        const [cx, cy] = dataToCanvas(c.cx, c.cy);
        
        const text = c.name + " (" + c.count + ")";
        const tw = labelCtx.measureText(text).width + 8;
        
        // Background pill
        labelCtx.fillStyle = "rgba(20,184,166,0.85)";
        labelCtx.beginPath();
        labelCtx.roundRect(cx - tw/2, cy - 8, tw, 16, 4);
        labelCtx.fill();
        
        // Text
        labelCtx.fillStyle = "#fff";
        labelCtx.fillText(text, cx, cy);
      }});
      
      labelCtx.restore();
    }}
  }}
  
  // ----------------------------
  // Selection Label Drawing (independent of regions)
  // ----------------------------
  function drawSelectionLabels() {{
    if (!showSelectionLabels || selectionLabels.length === 0) return;
    if (!labelCtx || !labelOverlay) return;
    
    const dpr = window.devicePixelRatio || 1;
    
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    labelCtx.font = "bold 10px ui-monospace, monospace";
    labelCtx.textAlign = "center";
    labelCtx.textBaseline = "middle";
    
    selectionLabels.forEach(lbl => {{
      if (lbl.cx == null || lbl.cy == null) return;
      const [cx, cy] = dataToCanvas(lbl.cx, lbl.cy);
      
      const text = lbl.name;
      const tw = labelCtx.measureText(text).width + 8;
      const color = lbl.color || "rgba(245,158,11,0.85)";
      
      // Background pill
      labelCtx.fillStyle = color;
      labelCtx.beginPath();
      labelCtx.roundRect(cx - tw/2, cy - 8, tw, 16, 4);
      labelCtx.fill();
      
      // Text
      labelCtx.fillStyle = "#fff";
      labelCtx.fillText(text, cx, cy);
    }});
    
    labelCtx.restore();
  }}
  
  // ----------------------------
  // Heatmap Ribbon Drawing
  // ----------------------------
  function drawHeatmapRibbon() {{
    if (!heatmapVisible || !heatmapRibbon) return;
    if (!labelOverlay || !labelCtx) return;
    
    const dpr = window.devicePixelRatio || 1;
    
    // If only start is placed, just draw the S marker
    if (!heatmapRibbon.end) {{
      if (heatmapRibbon.start) {{
        labelCtx.save();
        labelCtx.scale(dpr, dpr);
        const [sCanvasX, sCanvasY] = dataToCanvas(heatmapRibbon.start.x, heatmapRibbon.start.y);
        labelCtx.font = "bold 12px ui-monospace, monospace";
        labelCtx.textAlign = "center";
        labelCtx.textBaseline = "middle";
        labelCtx.fillStyle = "rgba(255, 255, 255, 0.85)";
        labelCtx.beginPath();
        labelCtx.arc(sCanvasX, sCanvasY, 10, 0, Math.PI * 2);
        labelCtx.fill();
        labelCtx.fillStyle = "#fff";
        labelCtx.fillText("S", sCanvasX, sCanvasY);
        labelCtx.restore();
      }}
      return;
    }}
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    
    const r = heatmapRibbon;
    const sx = r.start.x, sy = r.start.y;
    const ex = r.end.x, ey = r.end.y;
    const cp1 = r.controlPoints?.[0] || {{x: sx + (ex-sx)/3, y: sy + (ey-sy)/3}};
    const cp2 = r.controlPoints?.[1] || {{x: sx + 2*(ex-sx)/3, y: sy + 2*(ey-sy)/3}};
    const wStart = r.widthStart || 50;
    const wMid = r.widthMid || 50;
    const wEnd = r.widthEnd || 50;
    
    // Bezier evaluation helpers
    function bezierPt(t) {{
      const u = 1-t;
      return [
        u*u*u*sx + 3*u*u*t*cp1.x + 3*u*t*t*cp2.x + t*t*t*ex,
        u*u*u*sy + 3*u*u*t*cp1.y + 3*u*t*t*cp2.y + t*t*t*ey
      ];
    }}
    function bezierTan(t) {{
      const u = 1-t;
      let tx = 3*u*u*(cp1.x-sx) + 6*u*t*(cp2.x-cp1.x) + 3*t*t*(ex-cp2.x);
      let ty = 3*u*u*(cp1.y-sy) + 6*u*t*(cp2.y-cp1.y) + 3*t*t*(ey-cp2.y);
      const len = Math.sqrt(tx*tx + ty*ty) || 1;
      return [tx/len, ty/len];
    }}
    function interpWidth(t) {{
      const L0 = (t-0.5)*(t-1)/0.5;
      const L1 = t*(t-1)/(-0.25);
      const L2 = t*(t-0.5)/0.5;
      return wStart*L0 + wMid*L1 + wEnd*L2;
    }}
    
    const numSteps = 40;
    const topEdge = [];
    const botEdge = [];
    
    for (let i = 0; i <= numSteps; i++) {{
      const t = i / numSteps;
      const [px, py] = bezierPt(t);
      const [tx, ty] = bezierTan(t);
      const nx = -ty, ny = tx; // normal
      const w = interpWidth(t) / 2;
      
      const [topX, topY] = dataToCanvas(px + nx*w, py + ny*w);
      const [botX, botY] = dataToCanvas(px - nx*w, py - ny*w);
      topEdge.push([topX, topY]);
      botEdge.push([botX, botY]);
    }}
    
    // Draw ribbon fill (semi-transparent)
    labelCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
    labelCtx.beginPath();
    labelCtx.moveTo(topEdge[0][0], topEdge[0][1]);
    for (let i = 1; i < topEdge.length; i++) labelCtx.lineTo(topEdge[i][0], topEdge[i][1]);
    for (let i = botEdge.length - 1; i >= 0; i--) labelCtx.lineTo(botEdge[i][0], botEdge[i][1]);
    labelCtx.closePath();
    labelCtx.fill();
    
    // Draw ribbon outline (dashed, purple)
    labelCtx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    labelCtx.lineWidth = 1.5;
    labelCtx.setLineDash([6, 3]);
    
    // Top edge
    labelCtx.beginPath();
    labelCtx.moveTo(topEdge[0][0], topEdge[0][1]);
    for (let i = 1; i < topEdge.length; i++) labelCtx.lineTo(topEdge[i][0], topEdge[i][1]);
    labelCtx.stroke();
    
    // Bottom edge
    labelCtx.beginPath();
    labelCtx.moveTo(botEdge[0][0], botEdge[0][1]);
    for (let i = 1; i < botEdge.length; i++) labelCtx.lineTo(botEdge[i][0], botEdge[i][1]);
    labelCtx.stroke();
    
    // Draw center spine (solid, thin)
    labelCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    labelCtx.lineWidth = 1;
    labelCtx.setLineDash([]);
    labelCtx.beginPath();
    for (let i = 0; i <= numSteps; i++) {{
      const t = i / numSteps;
      const [px, py] = bezierPt(t);
      const [cx, cy] = dataToCanvas(px, py);
      if (i === 0) labelCtx.moveTo(cx, cy);
      else labelCtx.lineTo(cx, cy);
    }}
    labelCtx.stroke();
    
    // Draw bin dividers if bins exist
    if (heatmapBins && heatmapBins.length > 0) {{
      labelCtx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      labelCtx.lineWidth = 1;
      labelCtx.setLineDash([3, 3]);
      
      const numBins = heatmapBins.length;
      for (let i = 1; i < numBins; i++) {{
        const t = i / numBins;
        const [px, py] = bezierPt(t);
        const [tx, ty] = bezierTan(t);
        const nx = -ty, ny = tx;
        const w = interpWidth(t) / 2;
        
        const [x1, y1] = dataToCanvas(px + nx*w, py + ny*w);
        const [x2, y2] = dataToCanvas(px - nx*w, py - ny*w);
        
        labelCtx.beginPath();
        labelCtx.moveTo(x1, y1);
        labelCtx.lineTo(x2, y2);
        labelCtx.stroke();
      }}
    }}
    
    // Draw S and E labels
    labelCtx.setLineDash([]);
    const [sCanvasX, sCanvasY] = dataToCanvas(sx, sy);
    const [eCanvasX, eCanvasY] = dataToCanvas(ex, ey);
    
    labelCtx.font = "bold 12px ui-monospace, monospace";
    labelCtx.textAlign = "center";
    labelCtx.textBaseline = "middle";
    
    // S label
    labelCtx.fillStyle = "rgba(255, 255, 255, 0.85)";
    labelCtx.beginPath();
    labelCtx.arc(sCanvasX, sCanvasY, 10, 0, Math.PI * 2);
    labelCtx.fill();
    labelCtx.fillStyle = "#fff";
    labelCtx.fillText("S", sCanvasX, sCanvasY);
    
    // E label
    labelCtx.fillStyle = "rgba(255, 255, 255, 0.85)";
    labelCtx.beginPath();
    labelCtx.arc(eCanvasX, eCanvasY, 10, 0, Math.PI * 2);
    labelCtx.fill();
    labelCtx.fillStyle = "#fff";
    labelCtx.fillText("E", eCanvasX, eCanvasY);
    
    // Draw width handles (small circles at start, mid, end on both edges)
    if (heatmapMode === "editing") {{
      const handlePositions = [0, 0.5, 1];
      labelCtx.fillStyle = "rgba(20, 184, 166, 0.8)";
      
      handlePositions.forEach(t => {{
        const [px, py] = bezierPt(t);
        const [tx, ty] = bezierTan(t);
        const nx = -ty, ny = tx;
        const w = interpWidth(t) / 2;
        
        // Top handle
        const [htx, hty] = dataToCanvas(px + nx*w, py + ny*w);
        labelCtx.beginPath();
        labelCtx.arc(htx, hty, 5, 0, Math.PI * 2);
        labelCtx.fill();
        
        // Bottom handle
        const [hbx, hby] = dataToCanvas(px - nx*w, py - ny*w);
        labelCtx.beginPath();
        labelCtx.arc(hbx, hby, 5, 0, Math.PI * 2);
        labelCtx.fill();
      }});
      
      // Draw bezier control point handles (white)
      labelCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      labelCtx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      labelCtx.lineWidth = 1.5;
      const [c1x, c1y] = dataToCanvas(cp1.x, cp1.y);
      const [c2x, c2y] = dataToCanvas(cp2.x, cp2.y);
      
      // Control point 1
      labelCtx.beginPath();
      labelCtx.arc(c1x, c1y, 5, 0, Math.PI * 2);
      labelCtx.fill();
      labelCtx.stroke();
      
      // Control point 2
      labelCtx.beginPath();
      labelCtx.arc(c2x, c2y, 5, 0, Math.PI * 2);
      labelCtx.fill();
      labelCtx.stroke();
      
      // Lines from S to CP1, CP2 to E (handle arms)
      labelCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      labelCtx.lineWidth = 1;
      labelCtx.beginPath();
      labelCtx.moveTo(sCanvasX, sCanvasY);
      labelCtx.lineTo(c1x, c1y);
      labelCtx.stroke();
      labelCtx.beginPath();
      labelCtx.moveTo(eCanvasX, eCanvasY);
      labelCtx.lineTo(c2x, c2y);
      labelCtx.stroke();
      
      // Midpoint handle (white square with purple border)
      const [mx, my] = bezierPt(0.5);
      const [mcx, mcy] = dataToCanvas(mx, my);
      labelCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
      labelCtx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      labelCtx.lineWidth = 1.5;
      labelCtx.fillRect(mcx - 5, mcy - 5, 10, 10);
      labelCtx.strokeRect(mcx - 5, mcy - 5, 10, 10);
    }}
    
    labelCtx.restore();
  }}
  
  // ----------------------------
  // Heatmap Panel Rendering (in parent page)
  // ----------------------------
  let _lastHeatmapData = null;
  function renderHeatmapPanel(data) {{
    _lastHeatmapData = data;
    const panelEl = document.getElementById("heatmap_panel_" + iframeId);
    const hCanvas = document.getElementById("heatmap_canvas_" + iframeId);
    const titleEl = document.getElementById("heatmap_title_" + iframeId);
    const infoEl = document.getElementById("heatmap_info_" + iframeId);
    
    if (!panelEl || !hCanvas) return;
    
    panelEl.style.display = "block";
    
    const genes = data.genes || [];
    const numBins = data.numBins || 15;
    const heatmap = data.heatmap || {{}};
    const bins = data.bins || [];
    
    if (genes.length === 0) return;
    
    if (infoEl) infoEl.textContent = `${{genes.length}} genes × ${{numBins}} bins (${{data.totalCells || 0}} cells)`;
    
    const ctx = hCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    
    // Layout — fit to available width
    const rowLabelWidth = 65;
    const cellCountHeight = 0; // removed — too noisy at high bin counts
    const cellH = 22;
    const availableW = panelEl.clientWidth - 20; // padding
    const totalW = Math.max(availableW, rowLabelWidth + numBins * 2);
    const cellW = Math.max(1, (totalW - rowLabelWidth) / numBins);
    const totalH = cellH * genes.length;
    
    hCanvas.width = totalW * dpr;
    hCanvas.height = totalH * dpr;
    hCanvas.style.width = totalW + "px";
    hCanvas.style.height = totalH + "px";
    ctx.scale(dpr, dpr);
    
    ctx.clearRect(0, 0, totalW, totalH);
    
    // Find global min/max from data
    let gMin = Infinity, gMax = -Infinity;
    genes.forEach(gene => {{
      (heatmap[gene] || []).forEach(v => {{
        if (v < gMin) gMin = v;
        if (v > gMax) gMax = v;
      }});
    }});
    if (gMin === gMax) gMax = gMin + 1;

    // Apply custom min/max overrides if set
    const _hmMinInput = document.getElementById("heatmap_min_" + iframeId);
    const _hmMaxInput = document.getElementById("heatmap_max_" + iframeId);
    const _customMin = _hmMinInput && _hmMinInput.value.trim() !== "" ? parseFloat(_hmMinInput.value) : null;
    const _customMax = _hmMaxInput && _hmMaxInput.value.trim() !== "" ? parseFloat(_hmMaxInput.value) : null;
    if (_customMin !== null && !isNaN(_customMin)) gMin = _customMin;
    if (_customMax !== null && !isNaN(_customMax)) gMax = _customMax;
    if (gMin >= gMax) gMax = gMin + 1;
    
    // Rows — no column headers, just gene labels + colored cells
    genes.forEach((gene, gi) => {{
      const y = gi * cellH;
      const row = heatmap[gene] || [];
      
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = "#7c3aed";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(gene.length > 8 ? gene.slice(0,7) + "…" : gene, rowLabelWidth - 4, y + cellH / 2);
      
      for (let b = 0; b < numBins; b++) {{
        const v = row[b] || 0;
        const norm = (v - gMin) / (gMax - gMin);
        ctx.fillStyle = viridisHex(norm);
        const bx = rowLabelWidth + b * cellW;
        const gap = cellW > 3 ? 0.5 : 0;
        ctx.fillRect(bx, y, cellW - gap, cellH - 1);
      }}
    }});
    
    // Render color scale bar above
    renderHeatmapScale(gMin, gMax, rowLabelWidth, totalW);
  }}
  
  // Colormap function for heatmap — uses the same colormap as GEX
  function viridisHex(t) {{
    const rgb = cmapRGB(Math.max(0, Math.min(1, t)), currentColormap);
    return `rgb(${{Math.round(rgb[0]*255)}},${{Math.round(rgb[1]*255)}},${{Math.round(rgb[2]*255)}})`;
  }}
  
  // Heatmap camera button
  const heatmapCamBtn = document.getElementById("heatmap_camera_" + iframeId);
  if (heatmapCamBtn) {{
    heatmapCamBtn.addEventListener("click", () => {{
      const hCanvas = document.getElementById("heatmap_canvas_" + iframeId);
      if (!hCanvas) return;
      const defaultName = "heatmap_" + new Date().toISOString().slice(0,10);
      const filename = prompt("Enter filename for heatmap PNG:", defaultName);
      if (!filename || !filename.trim()) return;
      const link = document.createElement("a");
      link.download = filename.trim() + ".png";
      link.href = hCanvas.toDataURL("image/png");
      link.click();
    }});
  }}
  
  // Heatmap bins slider
  const heatmapBinsSlider = document.getElementById("heatmap_bins_" + iframeId);
  const heatmapBinsVal = document.getElementById("heatmap_bins_val_" + iframeId);
  if (heatmapBinsSlider) {{
    heatmapBinsSlider.addEventListener("input", () => {{
      if (heatmapBinsVal) heatmapBinsVal.textContent = heatmapBinsSlider.value;
    }});
    heatmapBinsSlider.addEventListener("change", () => {{
      // Tell iframe about the new bin count so it recomputes
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "heatmap_bins_changed",
          numBins: parseInt(heatmapBinsSlider.value),
        }}, "*");
      }}
    }});
  }}
  
  // Heatmap min/max inputs — re-render when changed
  const _hmMinInp = document.getElementById("heatmap_min_" + iframeId);
  const _hmMaxInp = document.getElementById("heatmap_max_" + iframeId);
  [_hmMinInp, _hmMaxInp].forEach(inp => {{
    if (inp) inp.addEventListener("change", () => {{ if (_lastHeatmapData) renderHeatmapPanel(_lastHeatmapData); }});
  }});

  // Render color scale bar
  function renderHeatmapScale(gMin, gMax, rowLabelWidth, totalW) {{
    const scaleCanvas = document.getElementById("heatmap_scale_" + iframeId);
    if (!scaleCanvas) return;
    
    const dpr = window.devicePixelRatio || 1;
    const scaleH = 30;
    scaleCanvas.width = totalW * dpr;
    scaleCanvas.height = scaleH * dpr;
    scaleCanvas.style.width = totalW + "px";
    scaleCanvas.style.height = scaleH + "px";

    const ctx = scaleCanvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalW, scaleH);

    // Draw gradient bar
    const barX = rowLabelWidth;
    const barW = totalW - rowLabelWidth;
    const barH = 12;
    const barY = 2;

    for (let i = 0; i < barW; i++) {{
      const t = i / barW;
      ctx.fillStyle = viridisHex(t);
      ctx.fillRect(barX + i, barY, 1, barH);
    }}

    // Min and max labels (below bar with enough vertical gap)
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = "#888";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(gMin.toFixed(2), barX, barY + barH + 3);
    ctx.textAlign = "right";
    ctx.fillText(gMax.toFixed(2), barX + barW, barY + barH + 3);
    ctx.textAlign = "center";
    ctx.fillText("expression", barX + barW / 2, barY + barH + 3);
  }}
  
  // Hit test heatmap handles — returns handle id or null
  function hitTestHeatmapHandle(canvasX, canvasY) {{
    if (!heatmapRibbon || !heatmapRibbon.end) return null;
    
    const r = heatmapRibbon;
    const hitRadius = 12;
    
    // Check S and E endpoints
    const [sx, sy] = dataToCanvas(r.start.x, r.start.y);
    if (Math.abs(canvasX - sx) < hitRadius && Math.abs(canvasY - sy) < hitRadius) return "start";
    
    const [ex, ey] = dataToCanvas(r.end.x, r.end.y);
    if (Math.abs(canvasX - ex) < hitRadius && Math.abs(canvasY - ey) < hitRadius) return "end";
    
    // Check control points
    if (r.controlPoints) {{
      const [c1x, c1y] = dataToCanvas(r.controlPoints[0].x, r.controlPoints[0].y);
      if (Math.abs(canvasX - c1x) < hitRadius && Math.abs(canvasY - c1y) < hitRadius) return "cp1";
      
      const [c2x, c2y] = dataToCanvas(r.controlPoints[1].x, r.controlPoints[1].y);
      if (Math.abs(canvasX - c2x) < hitRadius && Math.abs(canvasY - c2y) < hitRadius) return "cp2";
    }}
    
    // Check midpoint
    const cp1 = r.controlPoints?.[0] || {{x: r.start.x + (r.end.x - r.start.x)/3, y: r.start.y + (r.end.y - r.start.y)/3}};
    const cp2 = r.controlPoints?.[1] || {{x: r.start.x + 2*(r.end.x - r.start.x)/3, y: r.start.y + 2*(r.end.y - r.start.y)/3}};
    const t = 0.5, u = 0.5;
    const mx = u*u*u*r.start.x + 3*u*u*t*cp1.x + 3*u*t*t*cp2.x + t*t*t*r.end.x;
    const my = u*u*u*r.start.y + 3*u*u*t*cp1.y + 3*u*t*t*cp2.y + t*t*t*r.end.y;
    const [mcx, mcy] = dataToCanvas(mx, my);
    if (Math.abs(canvasX - mcx) < hitRadius && Math.abs(canvasY - mcy) < hitRadius) return "midpoint";
    
    // Check width handles at t=0, 0.5, 1
    const widthTs = [0, 0.5, 1];
    for (const wt of widthTs) {{
      const wu = 1 - wt;
      const bx = wu*wu*wu*r.start.x + 3*wu*wu*wt*cp1.x + 3*wu*wt*wt*cp2.x + wt*wt*wt*r.end.x;
      const by = wu*wu*wu*r.start.y + 3*wu*wu*wt*cp1.y + 3*wu*wt*wt*cp2.y + wt*wt*wt*r.end.y;
      
      // Tangent and normal
      let ttx = 3*wu*wu*(cp1.x-r.start.x) + 6*wu*wt*(cp2.x-cp1.x) + 3*wt*wt*(r.end.x-cp2.x);
      let tty = 3*wu*wu*(cp1.y-r.start.y) + 6*wu*wt*(cp2.y-cp1.y) + 3*wt*wt*(r.end.y-cp2.y);
      const tlen = Math.sqrt(ttx*ttx + tty*tty) || 1;
      const nx = -tty/tlen, ny = ttx/tlen;
      
      // Width at this t
      const L0 = (wt-0.5)*(wt-1)/0.5;
      const L1 = wt*(wt-1)/(-0.25);
      const L2 = wt*(wt-0.5)/0.5;
      const w = (r.widthStart*L0 + r.widthMid*L1 + r.widthEnd*L2) / 2;
      
      // Top and bottom handle positions
      const [htx, hty] = dataToCanvas(bx + nx*w, by + ny*w);
      if (Math.abs(canvasX - htx) < hitRadius && Math.abs(canvasY - hty) < hitRadius) return "width_" + wt;
      
      const [hbx, hby] = dataToCanvas(bx - nx*w, by - ny*w);
      if (Math.abs(canvasX - hbx) < hitRadius && Math.abs(canvasY - hby) < hitRadius) return "width_" + wt;
    }}
    
    return null;
  }}
  
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
      zoom = 1.0; panX = 0; panY = 0; rotation = 0; _orbitalX = 0; _orbitalY = 0;
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
    
    // Samples to skip in layout (unassigned cells from region-based sample_id columns)
    const _skipNames = new Set(["nan", "none", "", "__na__", "__NA__"]);

    const groups = new Array(nGroups);
    for (let g = 0; g < nGroups; g++) groups[g] = [];
    for (let si = 0; si < nSamp; si++) {{
      if (_skipNames.has((names[si] || "").toLowerCase())) continue;
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
    }} else if (posCustom[currentEmbedding]) {{
      posArray = posCustom[currentEmbedding];
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

    // Precompute GEX vmin/vmax scale (for per-cell colormap normalization)
    const _userVmin = (_ps2 && _ps2.gex && _ps2.gex.userVmin != null) ? _ps2.gex.userVmin : null;
    const _userVmax = (_ps2 && _ps2.gex && _ps2.gex.userVmax != null) ? _ps2.gex.userVmax : null;
    const _useCustomRange = (_userVmin != null || _userVmax != null) && currentGexVmax > 0;
    const _rangeLo = _useCustomRange ? (_userVmin != null ? _userVmin : 0) : 0;
    const _rangeHi = _useCustomRange ? (_userVmax != null ? _userVmax : currentGexVmax) : currentGexVmax;
    const _rangeSpan = (_rangeHi - _rangeLo) || 1;

    // Build selection index set for fast lookup
    const selIndices = _ps2 && _ps2.selectionIndices;
    const selectionSet = selIndices && selIndices.length > 0 ? new Set(selIndices) : null;
    const hasActiveSelection = selectionSet !== null;

    // LAYERED: GEX base + obs overlay
    const _defaultGrey = _darkMode ? 0.75 : 0.6;
    const colors = new Float32Array(loadedCount * 3);
    const outlineColors = new Float32Array(loadedCount * 3);
    for (let i = 0; i < loadedCount; i++) {{
      let r = _defaultGrey, g = _defaultGrey, b = _defaultGrey;
      let ro = _defaultGrey, go = _defaultGrey, bo = _defaultGrey;
      const obsVal = obsValues[i];
      const gexVal = gexValues[i];

      // Layer 1: GEX colormap — when gene is active, ALL cells get colored
      // gexVal 0-255 maps to expression 0-currentGexVmax; apply optional user vmin/vmax
      if (currentGexGene) {{
        let t;
        if (_useCustomRange) {{
          const actualVal = (gexVal / 255.0) * currentGexVmax;
          t = clamp01((actualVal - _rangeLo) / _rangeSpan);
        }} else {{
          t = clamp01(gexVal / 255.0);
        }}
        const c = cmapRGB(t, activeCmap);
        r = c[0]; g = c[1]; b = c[2];
      }}

      if (_obsOutlineMode) {{
        // Outline mode: base fill = GEX/grey, ring = obs category color
        if (obsVal > 0 && currentObsColumn) {{
          const ci = obsVal - 1;
          if (enabledArr && enabledArr[ci] === false) {{
            r = _defaultGrey * 0.45; g = _defaultGrey * 0.45; b = _defaultGrey * 0.45;
            ro = _defaultGrey * 0.45; go = _defaultGrey * 0.45; bo = _defaultGrey * 0.45;
          }} else {{
            const c = currentPalette && currentPalette[ci] ? parseColor(currentPalette[ci]) : fallbackCatColorRGB(ci);
            ro = c[0]; go = c[1]; bo = c[2];
          }}
        }}
      }} else {{
        // Normal mode: obs color alpha-blended on top of GEX
        if (obsVal > 0 && currentObsColumn) {{
          const ci = obsVal - 1;
          if (enabledArr && enabledArr[ci] === false) {{
            r *= 0.45; g *= 0.45; b *= 0.45;
          }} else {{
            let or2, og2, ob2;
            if (currentPalette && currentPalette[ci]) {{
              const c = parseColor(currentPalette[ci]); or2=c[0]; og2=c[1]; ob2=c[2];
            }} else {{
              const c = fallbackCatColorRGB(ci); or2=c[0]; og2=c[1]; ob2=c[2];
            }}
            r = r*(1-obsAlpha) + or2*obsAlpha;
            g = g*(1-obsAlpha) + og2*obsAlpha;
            b = b*(1-obsAlpha) + ob2*obsAlpha;
          }}
        }}
        ro = r; go = g; bo = b;
      }}

      // Layer 3: Selection highlighting - dim unselected cells to 20%
      if (hasActiveSelection && !selectionSet.has(cellIds[i])) {{
        r *= 0.2; g *= 0.2; b *= 0.2;
        ro *= 0.2; go *= 0.2; bo *= 0.2;
      }}

      colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
      outlineColors[i*3] = ro; outlineColors[i*3+1] = go; outlineColors[i*3+2] = bo;
    }}

    // Build per-vertex size scale: disabled cells get default size (0), enabled get user size (1)
    let hasMask = false;
    if (enabledArr) {{
      for (let ci = 0; ci < enabledArr.length; ci++) {{
        if (enabledArr[ci] === false) {{ hasMask = true; break; }}
      }}
    }}
    _hasMask = hasMask;
    const sizeScales = new Float32Array(loadedCount);
    for (let i = 0; i < loadedCount; i++) {{
      const ci = obsValues[i] - 1;
      if (hasMask && obsValues[i] > 0 && currentObsColumn && enabledArr && enabledArr[ci] === false) {{
        sizeScales[i] = 0.0;
      }} else {{
        sizeScales[i] = 1.0;
      }}
    }}

    // Build per-vertex z-layer values for 3D depth separation
    const zLayers = new Float32Array(loadedCount);
    const embZ = getEmbeddingZ(currentEmbedding);
    if (_embeddingHas3D && embZ) {{
      // 3D embedding: use normalized 3rd dimension as z-layer.
      // Mask mode still applies: masked cells pushed to back (-1).
      for (let i = 0; i < loadedCount; i++) {{
        zLayers[i] = embZ[i];
        if (hasMask && obsValues[i] > 0 && currentObsColumn && enabledArr && enabledArr[obsValues[i] - 1] === false) {{
          zLayers[i] = -1.0;  // push masked cell to back
        }}
      }}
    }} else if (currentGexGene && gexValues) {{
      // GEX mode: z-layer driven by expression value (low expr = back, high = front)
      for (let i = 0; i < loadedCount; i++) {{
        let t = gexValues[i] / 255.0;
        zLayers[i] = t * 2.0 - 1.0;
      }}
    }} else if (_3dStackMode && currentObsColumn && currentPalette) {{
      // Stack mode: ALL categories evenly spread from -1 to +1 based on their index
      const numCats = currentPalette.length;
      for (let i = 0; i < loadedCount; i++) {{
        const ci = obsValues[i] - 1;
        if (obsValues[i] > 0 && numCats > 1) {{
          zLayers[i] = (ci / (numCats - 1)) * 2.0 - 1.0;
        }} else {{
          zLayers[i] = 0.0;
        }}
      }}
    }} else {{
      // Mask mode (default): active/front = +1, masked/back = -1
      for (let i = 0; i < loadedCount; i++) {{
        zLayers[i] = sizeScales[i] > 0.5 ? 1.0 : -1.0;
      }}
    }}

    // Upload to GPU
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, outlineColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, outlineColors, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeScaleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sizeScales, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, zLayerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, zLayers, gl.DYNAMIC_DRAW);

    gpuPointCount = loadedCount;
    gpuDataDirty = false;
    gpuEmbedding = currentEmbedding;
    gpuLoadedCount = loadedCount;
    
    console.log(`[WebGL] Uploaded ${{loadedCount.toLocaleString()}} points for ${{currentEmbedding}}`);
  }}

  // ----------------------------
  // EMBEDDING ANIMATION FUNCTIONS (need WebGL context)
  // ----------------------------

  // Translate region polygon vertices when the active embedding changes.
  // Uses stored cell indices to compute the new centroid in the new embedding,
  // then shifts all vertices by (new_centroid - old_centroid).
  function translateRegionPolygons() {{
    if (!regionPolygons || regionPolygons.length === 0) return;
    const posArr = getEmbeddingPositions(currentEmbedding);
    if (!posArr || posArr.length === 0) return;

    regionPolygons.forEach(region => {{
      if (!region.indices || region.indices.length === 0) return;
      if (region.centroid_x == null || region.centroid_y == null) return;

      let sumX = 0, sumY = 0, count = 0;
      for (let i = 0; i < region.indices.length; i++) {{
        const cellIdx = region.indices[i];
        const writePos = cellIdToIndex.get(cellIdx);
        if (writePos !== undefined && writePos * 2 + 1 < posArr.length) {{
          sumX += posArr[writePos * 2];
          sumY += posArr[writePos * 2 + 1];
          count++;
        }}
      }}
      if (count === 0) return;

      const newCx = sumX / count;
      const newCy = sumY / count;
      const dx = newCx - region.centroid_x;
      const dy = newCy - region.centroid_y;

      if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return;

      region.polygons = region.polygons.map(poly => poly.map(([x, y]) => [x + dx, y + dy]));
      region.centroid_x = newCx;
      region.centroid_y = newCy;
    }});
  }}

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
      translateRegionPolygons();
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
    gl.uniform1f(u_defaultPointSize, finalSize);
    const _ps = window["_plotState_" + iframeId];
    const _gexOp = (_ps && _ps.gex && _ps.gex.opacity != null) ? _ps.gex.opacity : 0.85;
    gl.uniform1f(u_opacity, currentGexGene ? _gexOp : 0.85);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, outlineColorBuffer);
    gl.vertexAttribPointer(a_outlineColorLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeScaleBuffer);
    gl.enableVertexAttribArray(a_sizeScaleLoc);
    gl.vertexAttribPointer(a_sizeScaleLoc, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, zLayerBuffer);
    gl.enableVertexAttribArray(a_zLayerLoc);
    gl.vertexAttribPointer(a_zLayerLoc, 1, gl.FLOAT, false, 0, 0);
    gl.uniform1f(u_outlineMode, _obsOutlineMode ? 1.0 : 0.0);
    gl.uniform1f(u_headX, _3dMode && _faceMeshReady ? _headX : 0.0);
    gl.uniform1f(u_headY, _3dMode && _faceMeshReady ? _headY : 0.0);
    {{
      const layerT = _3dMode && _faceMeshReady
        ? Math.min(1, (performance.now() - _layerTransitionStart) / 500) : 0;
      const eased = 1 - Math.pow(1 - layerT, 3);
      gl.uniform1f(u_layerStrength, eased * _depthStrength);
    }}
    gl.uniform1f(u_rotX, _orbitalX);
    gl.uniform1f(u_rotY, _orbitalY);
    const _embedMetaAnim = METADATA[currentEmbedding];
    if (_embedMetaAnim) {{
      const _cx = (_embedMetaAnim.minX + _embedMetaAnim.maxX) / 2;
      const _cy = (_embedMetaAnim.minY + _embedMetaAnim.maxY) / 2;
      const _span = Math.max(_embedMetaAnim.maxX - _embedMetaAnim.minX, _embedMetaAnim.maxY - _embedMetaAnim.minY) || 1;
      gl.uniform2f(u_centroid, _cx, _cy);
      gl.uniform1f(u_focalLength, _span * 3);
      gl.uniform1f(u_orbitalZSep, (_orbitalMode || (_3dMode && (_3dStackMode || _embeddingHas3D))) ? _span * _depthStrength : 0.0);
    }} else {{
      gl.uniform2f(u_centroid, 0.0, 0.0);
      gl.uniform1f(u_focalLength, 0.0);
      gl.uniform1f(u_orbitalZSep, 0.0);
    }}

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.drawArrays(gl.POINTS, 0, loadedCount);

    drawMinimap();
  }}

  // ----------------------------
  // 3D mirror reflection: WebGL pass + gradient fade overlay
  // ----------------------------

  // Draw the floor plane: vanishing lines from viewport corners to mirror-plane backplate,
  // mirror gradient clipped to floor trapezoid, opaque triangles hiding sides of reflection.
  function drawMirrorFadeOverlay() {{
    if (!_3dMirror || !labelCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cachedPanelW || panel.getBoundingClientRect().width;
    const H = cachedPanelH || panel.getBoundingClientRect().height;

    // Fixed mirror position in CSS pixels
    const mirrorPxY = (1 - _MIRROR_PLANE_Y) / 2 * H;

    // Backplate inset corners at mirror plane (vanishing point targets)
    const bpL = W * 0.18;
    const bpR = W * 0.82;
    const bgColor = _darkMode ? "0,0,0" : "255,255,255";

    // --- 1: Mirror gradient clipped to floor trapezoid ---
    // Trapezoid: narrow at mirror plane (bpL→bpR), widens to full screen at bottom
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    labelCtx.beginPath();
    labelCtx.moveTo(bpL, mirrorPxY);
    labelCtx.lineTo(bpR, mirrorPxY);
    labelCtx.lineTo(W, H);
    labelCtx.lineTo(0, H);
    labelCtx.closePath();
    labelCtx.clip();

    const fadeH = Math.max(H - mirrorPxY, 1);
    const grad = labelCtx.createLinearGradient(0, mirrorPxY, 0, mirrorPxY + fadeH);
    grad.addColorStop(0,    `rgba(${{bgColor}},0)`);
    grad.addColorStop(0.3,  `rgba(${{bgColor}},0.65)`);
    grad.addColorStop(1,    `rgba(${{bgColor}},0.97)`);
    labelCtx.fillStyle = grad;
    labelCtx.fillRect(0, mirrorPxY, W, fadeH);
    labelCtx.restore();

    // --- 2: Side-fill triangles to mask reflection outside the floor trapezoid ---
    // Left triangle: (0, mirrorPxY) → (bpL, mirrorPxY) → (0, H)
    // Right triangle: (W, mirrorPxY) → (bpR, mirrorPxY) → (W, H)
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    labelCtx.fillStyle = `rgba(${{bgColor}},0.97)`;
    labelCtx.beginPath();
    labelCtx.moveTo(0, mirrorPxY);
    labelCtx.lineTo(bpL, mirrorPxY);
    labelCtx.lineTo(0, H);
    labelCtx.closePath();
    labelCtx.fill();
    labelCtx.beginPath();
    labelCtx.moveTo(W, mirrorPxY);
    labelCtx.lineTo(bpR, mirrorPxY);
    labelCtx.lineTo(W, H);
    labelCtx.closePath();
    labelCtx.fill();
    labelCtx.restore();

  }}

  // Show small orbital mode indicator on the 2D overlay
  function drawOrbitalIndicator() {{
    if (!_orbitalMode || !labelCtx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cachedPanelW || panel.getBoundingClientRect().width;
    labelCtx.save();
    labelCtx.scale(dpr, dpr);
    labelCtx.fillStyle = "rgba(0,0,0,0.5)";
    labelCtx.beginPath();
    labelCtx.roundRect(W - 90, 10, 80, 22, 4);
    labelCtx.fill();
    labelCtx.fillStyle = "white";
    labelCtx.font = "11px ui-monospace, monospace";
    labelCtx.textAlign = "center";
    labelCtx.fillText("ORBITAL [O]", W - 50, 25);
    labelCtx.textAlign = "left";
    labelCtx.restore();
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
    let c = rotTx * toClipX - 1;

    let d = sin * sx * toClipY;
    let e = cos * sy * toClipY;
    let f = rotTy * toClipY + 1;

    // 3D parallax: shift clip-space origin by head position
    if (_3dMode && _faceMeshReady) {{
      const pStrength = 0.7;
      c += _headX * pStrength;
      f -= _headY * pStrength;
    }}

    // WebGL uses column-major, so transpose
    const matrix = new Float32Array([
      a, d, 0,
      b, e, 0,
      c, f, 1
    ]);
    
    // Mirror plane is fixed in clip space — cells move past it, floor stays put.

    // Set up WebGL state
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Set uniforms for cell draw
    gl.useProgram(program);
    gl.uniformMatrix3fv(u_matrix, false, matrix);
    gl.uniform1f(u_reflectMode, 0.0);
    gl.uniform1f(u_reflectY, _MIRROR_PLANE_Y);

    const state = window["_plotState_" + iframeId] || {{}};
    const dpr2 = window.devicePixelRatio || 1;
    const userSize = (state.pointSize || 1.1) * dpr2 * 2;
    const baseDefaultSize = 1.1 * dpr2 * 2;
    gl.uniform1f(u_pointSize, userSize);
    gl.uniform1f(u_defaultPointSize, _hasMask ? baseDefaultSize : userSize);
    gl.uniform1f(u_opacity, 0.85);

    // Bind position buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    // Bind color buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(a_color);
    gl.vertexAttribPointer(a_color, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, outlineColorBuffer);
    gl.vertexAttribPointer(a_outlineColorLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeScaleBuffer);
    gl.enableVertexAttribArray(a_sizeScaleLoc);
    gl.vertexAttribPointer(a_sizeScaleLoc, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, zLayerBuffer);
    gl.enableVertexAttribArray(a_zLayerLoc);
    gl.vertexAttribPointer(a_zLayerLoc, 1, gl.FLOAT, false, 0, 0);
    gl.uniform1f(u_outlineMode, _obsOutlineMode ? 1.0 : 0.0);
    gl.uniform1f(u_headX, _3dMode && _faceMeshReady ? _headX : 0.0);
    gl.uniform1f(u_headY, _3dMode && _faceMeshReady ? _headY : 0.0);
    {{
      const layerT = _3dMode && _faceMeshReady
        ? Math.min(1, (performance.now() - _layerTransitionStart) / 500) : 0;
      const eased = 1 - Math.pow(1 - layerT, 3);
      gl.uniform1f(u_layerStrength, eased * _depthStrength);
    }}
    gl.uniform1f(u_rotX, _orbitalX);
    gl.uniform1f(u_rotY, _orbitalY);
    {{
      const _oMeta = METADATA[currentEmbedding];
      if (_oMeta) {{
        const _cx = (_oMeta.minX + _oMeta.maxX) / 2;
        const _cy = (_oMeta.minY + _oMeta.maxY) / 2;
        const _span = Math.max(_oMeta.maxX - _oMeta.minX, _oMeta.maxY - _oMeta.minY) || 1;
        gl.uniform2f(u_centroid, _cx, _cy);
        gl.uniform1f(u_focalLength, _span * 3);
        gl.uniform1f(u_orbitalZSep, (_orbitalMode || (_3dMode && (_3dStackMode || _embeddingHas3D || !!currentGexGene))) ? _span * _depthStrength : 0.0);
      }} else {{
        gl.uniform2f(u_centroid, 0.0, 0.0);
        gl.uniform1f(u_focalLength, 0.0);
        gl.uniform1f(u_orbitalZSep, 0.0);
      }}
    }}

    // Draw!
    gl.drawArrays(gl.POINTS, 0, gpuPointCount);

    // Mirror reflection pass — draw reflected points at low opacity (independent of parallax)
    if (_3dMirror) {{
      gl.uniform1f(u_reflectMode, 1.0);
      gl.uniform1f(u_reflectY, _MIRROR_PLANE_Y);
      gl.uniform1f(u_opacity, 0.2);
      gl.drawArrays(gl.POINTS, 0, gpuPointCount);
      gl.uniform1f(u_reflectMode, 0.0);
      gl.uniform1f(u_opacity, 0.85);
    }}

    drawSampleLabels();
    drawMirrorFadeOverlay();
    drawOrbitalIndicator();
    drawRegionPolygons();
    drawSelectionLabels();
    drawHeatmapRibbon();
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
    _mmScale = scale; _mmOffX = offX; _mmOffY = offY;

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

  // Minimap click: navigate viewport to the clicked data position
  minimap.style.cursor = "crosshair";
  minimap.addEventListener("click", (e) => {{
    const rect = minimap.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (minimap.width / rect.width);
    const py = (e.clientY - rect.top) * (minimap.height / rect.height);

    const embedMeta = METADATA[currentEmbedding];
    if (!embedMeta || _mmScale === 0) return;

    const dx = embedMeta.minX + (px - _mmOffX) / _mmScale;
    const dy = embedMeta.minY + (py - _mmOffY) / _mmScale;

    const panelRect = panel.getBoundingClientRect();
    const W = panelRect.width, H = panelRect.height;
    const spanX = (embedMeta.maxX - embedMeta.minX) || 1;
    const spanY = (embedMeta.maxY - embedMeta.minY) || 1;
    const baseScale = Math.min((W - 24) / spanX, (H - 24) / spanY);
    const viewScale = baseScale * zoom;
    const viewW = spanX * viewScale;
    const viewH = spanY * viewScale;

    panX = viewW / 2 - (dx - embedMeta.minX) * viewScale;
    panY = viewH / 2 - (dy - embedMeta.minY) * viewScale;

    markGPUDirty();
    draw();
  }});

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
  
  // Use ResizeObserver to catch ANY layout changes
  if (typeof ResizeObserver !== "undefined") {{
    const ro = new ResizeObserver(() => {{ resizeCanvas(); }});
    ro.observe(panel);
    // Also observe the parent container (catches Jupyter cell resize)
    if (panel.parentElement) ro.observe(panel.parentElement);
    if (panel.parentElement && panel.parentElement.parentElement) ro.observe(panel.parentElement.parentElement);
  }}
  
  // Aggressive resize scheduling to catch Jupyter layout settling
  // Check every 200ms for the first 5 seconds — if dimensions changed, resize
  let resizeCheckCount = 0;
  const resizeCheckInterval = setInterval(() => {{
    resizeCheckCount++;
    refreshPanelDimensions();
    if (resizeCheckCount >= 25) clearInterval(resizeCheckInterval); // stop after 5s
  }}, 200);
  
  // Force a window resize event dispatch — this is what DevTools opening does
  // that fixes the coordinate mapping. Dispatch after layout should be settled.
  [200, 500, 1000, 2000, 3000].forEach(ms => {{
    setTimeout(() => {{
      window.dispatchEvent(new Event("resize"));
    }}, ms);
  }});

  // ----------------------------
  // Pan/Zoom/Rotation controls
  // ----------------------------
  
  // Wheel zoom (WebGL handles all points efficiently - no viewport loading needed!)
  canvas.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const tool = window["_selectionTool_" + iframeId];

    // Expand tool: scroll wheel reserved for mask expand/collapse (handled later)
    if (tool === "expand") {{
      return;
    }}

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

    // O key for orbital 3D rotation
    if (e.key === "o" || e.key === "O") {{
      e.preventDefault();
      e.stopPropagation();
      const now = Date.now();
      if (now - _lastOrbitalKeyPress < 500) {{
        // Double-tap: reset orbital rotation
        _orbitalX = 0;
        _orbitalY = 0;
        _orbitalMode = false;
        canvas.style.cursor = "default";
      }} else {{
        _orbitalMode = !_orbitalMode;
        canvas.style.cursor = _orbitalMode ? "move" : "default";
      }}
      _lastOrbitalKeyPress = now;
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
    const tool = window["_selectionTool_" + iframeId] || window["_loadedSelShape_" + iframeId];
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
    
    // --- Ribbon tool (heatmap) ---
    if (tool === "ribbon") {{
      e.preventDefault();
      
      // Ensure cached panel dimensions are fresh before coordinate conversion
      refreshPanelDimensions();
      
      // Scale mouse coords to match canvas.width/dpr space (handles Jupyter CSS zoom)
      const dpr = window.devicePixelRatio || 1;
      const cssW = rect.width;
      const cssH = rect.height;
      const logicalW = canvas.width / dpr;
      const logicalH = canvas.height / dpr;
      const mx = x * (logicalW / cssW);
      const my = y * (logicalH / cssH);
      
      if (heatmapMode === "placing_start") {{
        const [dx, dy] = canvasToData(mx, my);
        heatmapRibbon = {{
          start: {{ x: dx, y: dy }},
          end: null,
          controlPoints: null,
          widthStart: 50,
          widthMid: 50,
          widthEnd: 50,
        }};
        heatmapMode = "placing_end";
        draw();
        return;
      }}
      
      if (heatmapMode === "placing_end") {{
        const [dx, dy] = canvasToData(mx, my);
        heatmapRibbon.end = {{ x: dx, y: dy }};
        
        const sx = heatmapRibbon.start.x, sy = heatmapRibbon.start.y;
        const ex = dx, ey = dy;
        heatmapRibbon.controlPoints = [
          {{ x: sx + (ex - sx) / 3, y: sy + (ey - sy) / 3 }},
          {{ x: sx + 2 * (ex - sx) / 3, y: sy + 2 * (ey - sy) / 3 }}
        ];
        
        const dist = Math.sqrt((ex - sx) * (ex - sx) + (ey - sy) * (ey - sy));
        const defaultWidth = Math.max(20, dist * 0.15);
        heatmapRibbon.widthStart = defaultWidth;
        heatmapRibbon.widthMid = defaultWidth;
        heatmapRibbon.widthEnd = defaultWidth;
        
        heatmapMode = "editing";
        
        const iframeEl = document.getElementById(iframeId);
        if (iframeEl && iframeEl.contentWindow) {{
          iframeEl.contentWindow.postMessage({{
            type: "heatmap_ribbon_placed",
            ribbon: heatmapRibbon,
          }}, "*");
        }}
        
        draw();
        return;
      }}
      
      // Editing mode — check handles
      if (heatmapMode === "editing" && heatmapRibbon) {{
        const handle = hitTestHeatmapHandle(mx, my);
        if (handle) {{
          heatmapDragging = handle;
          canvas.style.cursor = "grabbing";
          return;
        }}
      }}
      
      return; // Don't fall through to pan/zoom when ribbon tool active
    }}
    
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

    // Mask manipulation tools have their own interaction handlers - don't start drawing
    if (tool === "slice" || tool === "expand" || tool === "merge") {{
      e.preventDefault();
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
    }} else if (_orbitalMode) {{
      // Orbital rotation mode
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing";
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
    
    // --- Ribbon tool (heatmap) ---
    if (tool === "ribbon") {{
      // Ensure fresh dimensions
      refreshPanelDimensions();
      
      // Scale mouse coords to match canvas.width/dpr space
      const dpr = window.devicePixelRatio || 1;
      const cssW = rect.width;
      const cssH = rect.height;
      const logicalW = canvas.width / dpr;
      const logicalH = canvas.height / dpr;
      const mx = x * (logicalW / cssW);
      const my = y * (logicalH / cssH);
      
      // Handle dragging
      if (heatmapDragging && heatmapRibbon) {{
        const [dx, dy] = canvasToData(mx, my);
        const h = heatmapDragging;
        
        if (h === "start") {{
          heatmapRibbon.start = {{ x: dx, y: dy }};
        }} else if (h === "end") {{
          heatmapRibbon.end = {{ x: dx, y: dy }};
        }} else if (h === "cp1") {{
          heatmapRibbon.controlPoints[0] = {{ x: dx, y: dy }};
        }} else if (h === "cp2") {{
          heatmapRibbon.controlPoints[1] = {{ x: dx, y: dy }};
        }} else if (h === "midpoint") {{
          const sx = heatmapRibbon.start.x, sy = heatmapRibbon.start.y;
          const ex = heatmapRibbon.end.x, ey = heatmapRibbon.end.y;
          const mx = (sx + ex) / 2, my = (sy + ey) / 2;
          const offsetX = dx - mx, offsetY = dy - my;
          heatmapRibbon.controlPoints[0] = {{ x: sx + (ex - sx) / 3 + offsetX, y: sy + (ey - sy) / 3 + offsetY }};
          heatmapRibbon.controlPoints[1] = {{ x: sx + 2 * (ex - sx) / 3 + offsetX, y: sy + 2 * (ey - sy) / 3 + offsetY }};
        }} else if (h.startsWith("width_")) {{
          const t = parseFloat(h.split("_")[1]);
          const p0 = [heatmapRibbon.start.x, heatmapRibbon.start.y];
          const p3 = [heatmapRibbon.end.x, heatmapRibbon.end.y];
          const c1 = heatmapRibbon.controlPoints[0];
          const c2 = heatmapRibbon.controlPoints[1];
          const u = 1 - t;
          const bx = u*u*u*p0[0] + 3*u*u*t*c1.x + 3*u*t*t*c2.x + t*t*t*p3[0];
          const by = u*u*u*p0[1] + 3*u*u*t*c1.y + 3*u*t*t*c2.y + t*t*t*p3[1];
          const dist = Math.sqrt((dx - bx) * (dx - bx) + (dy - by) * (dy - by));
          const newWidth = dist * 2;
          if (t === 0) heatmapRibbon.widthStart = newWidth;
          else if (t === 0.5) heatmapRibbon.widthMid = newWidth;
          else if (t === 1) heatmapRibbon.widthEnd = newWidth;
        }}
        
        draw();
        return;
      }}
      
      // Cursor feedback
      if (heatmapMode === "placing_start" || heatmapMode === "placing_end") {{
        canvas.style.cursor = "crosshair";
      }} else if (heatmapMode === "editing" && heatmapRibbon) {{
        const handle = hitTestHeatmapHandle(mx, my);
        canvas.style.cursor = handle ? "grab" : "crosshair";
      }}
      return; // Don't fall through to other handlers
    }}
    
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
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;

      if (_orbitalMode) {{
        // Orbital 3D rotation: drag left/right → rotY, drag up/down → rotX
        _orbitalY += dx * 0.005;
        _orbitalX += dy * 0.005;
      }} else {{
        // Panning - rotation-aware
        const cos = Math.cos(-rotation);
        const sin = Math.sin(-rotation);
        panX += dx * cos - dy * sin;
        panY += dx * sin + dy * cos;
      }}
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
        const selTool = window["_selectionTool_" + iframeId] || window["_loadedSelShape_" + iframeId];

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

      if (tool === "slice") canvas.style.cursor = "crosshair";
      else if (tool === "expand") canvas.style.cursor = "ns-resize";
      else if (tool === "merge") canvas.style.cursor = "cell";
      else canvas.style.cursor = tool ? "crosshair" : "default";
    }}
  }});

  canvas.addEventListener("mouseup", () => {{
    const tool = window["_selectionTool_" + iframeId];
    
    // --- Ribbon tool drag end ---
    if (tool === "ribbon" && heatmapDragging && heatmapRibbon) {{
      heatmapDragging = null;
      canvas.style.cursor = "crosshair";
      
      // Notify iframe of updated ribbon
      const iframeEl = document.getElementById(iframeId);
      if (iframeEl && iframeEl.contentWindow) {{
        iframeEl.contentWindow.postMessage({{
          type: "heatmap_ribbon_updated",
          ribbon: heatmapRibbon,
        }}, "*");
      }}
      return;
    }}
    
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
    const tool = window["_selectionTool_" + iframeId] || window["_loadedSelShape_" + iframeId];
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
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    
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
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    
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
  function completeSelection(toolOverride) {{
    const path = window["_selectionPath_" + iframeId];
    // toolOverride > active drawing tool > loaded selection shape type
    const tool = toolOverride || window["_selectionTool_" + iframeId] || window["_loadedSelShape_" + iframeId];
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
        selectedIndices.push(cellIds[i]);
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