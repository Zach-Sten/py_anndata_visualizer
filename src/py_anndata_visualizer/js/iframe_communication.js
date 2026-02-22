/**
 * iframe_communication.js
 * 
 * This script is injected into the iframe and handles:
 * - Capturing button clicks + input fields
 * - Posting messages to parent window
 * - Receiving python responses and re-dispatching as CustomEvent('pythonResponse')
 * 
 * Placeholders (replaced via Python .format()):
 *   {{iframe_id}}      - Unique iframe identifier (JSON-encoded string)
 *   {{button_ids_js}}  - JSON array of button IDs
 *   {{initial_data_js}} - JSON object with initial data
 *   {{debug_log}}      - Debug logging statement (empty string if debug=False)
 */
(function() {{
  const iframeId = {iframe_id};
  const buttonIds = {button_ids_js};
  const initialData = {initial_data_js};
  let pendingRequest = null;

  function log(...args) {{ {debug_log} }}

  window.INITIAL_DATA = initialData;
  window._iframeId = iframeId;
  log("Initial data loaded:", initialData);

  // Buttons that are triggered programmatically (not UI buttons in iframe)
  const skipBridgeButtons = new Set([
    "computeLayoutBtn", "deleteLayoutBtn", "loadLayoutBtn", "saveLayoutBtn",
    "viewportBtn", "chunkBtn", "loadEmbeddingBtn", "__save_obs_column__",
    "dbscanBtn", "alphaShapeBtn"
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
