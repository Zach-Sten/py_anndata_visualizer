"""
Communication helpers for widget-based Python-JavaScript bridge.

This module provides utilities for managing the bidirectional communication
between the Python kernel and the JavaScript frontend via ipywidgets.
"""

import json
from typing import Any, Callable, Dict, List, Optional

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript


def create_data_bridges(
    iframe_id: str,
    button_ids: List[str],
) -> Dict[str, widgets.Text]:
    """
    Create hidden Text widgets that serve as data bridges for each button.
    
    These widgets allow JavaScript to pass data to Python by setting the
    widget's value, which can then be read by the Python callback.
    
    Args:
        iframe_id: Unique identifier for the iframe
        button_ids: List of button IDs that need data bridges
        
    Returns:
        Dict mapping button IDs to their corresponding Text widgets
    """
    data_bridges: Dict[str, widgets.Text] = {}
    
    for bid in button_ids:
        bridge_name = f"_data_{iframe_id}_{bid}"
        bridge = widgets.Text(
            value="",
            description=bridge_name,
            placeholder=f"Data bridge for {bid}",
            layout=widgets.Layout(
                width="0px",
                height="0px",
                visibility="hidden",
                display="none"
            ),
        )
        # Add CSS classes for JavaScript selection
        bridge.add_class(f"data-bridge-{iframe_id}")
        bridge.add_class(f"data-bridge-{bid}")
        data_bridges[bid] = bridge
        display(bridge)
    
    return data_bridges


def create_poll_button(
    iframe_id: str,
    button_id: str,
    handler: Callable,
) -> widgets.Button:
    """
    Create a hidden Button widget that triggers Python callbacks.
    
    JavaScript clicks this hidden button to trigger the Python handler.
    
    Args:
        iframe_id: Unique identifier for the iframe
        button_id: ID of the button this poll button corresponds to
        handler: Callback function to execute when clicked
        
    Returns:
        The created Button widget
    """
    label = f"_poll_{iframe_id}__{button_id}"
    poll_btn = widgets.Button(
        description=label,
        tooltip=label,
        layout=widgets.Layout(
            width="0px",
            height="0px",
            visibility="hidden",
            display="none"
        ),
    )
    poll_btn.on_click(handler)
    display(poll_btn)
    return poll_btn


def send_to_javascript(
    iframe_id: str,
    data: Dict[str, Any],
    output: widgets.Output,
) -> None:
    """
    Send data from Python to JavaScript.
    
    This function dispatches the data to both the iframe (via sendToIframe)
    and the plot canvas (via updatePlot).
    
    Args:
        iframe_id: Unique identifier for the iframe
        data: Data to send to JavaScript
        output: Output widget for executing JavaScript
    """
    with output:
        js_code = f"""
        (function() {{
          const iframeId = {json.dumps(iframe_id)};
          const sendFn = window["sendToIframe_" + iframeId];
          const updFn  = window["updatePlot_" + iframeId];
          if (sendFn) sendFn({json.dumps(data)});
          if (updFn)  updFn({json.dumps(data)});
        }})();
        """
        display(Javascript(js_code))


def make_callback_handler(
    button_id: str,
    button_callbacks: Dict[str, Callable],
    callback_args: Dict[str, Any],
    data_bridges: Dict[str, widgets.Text],
    iframe_id: str,
    output: widgets.Output,
    serialize_fn: Callable,
    max_result_size: int = 30_000_000,
    debug: bool = False,
) -> Callable:
    """
    Create a callback handler for a specific button.
    
    This factory function creates a handler that:
    1. Reads request data from the data bridge
    2. Calls the Python callback function
    3. Serializes the result
    4. Sends the result back to JavaScript
    
    Args:
        button_id: ID of the button
        button_callbacks: Dict mapping button IDs to callback functions
        callback_args: Additional arguments passed to callbacks
        data_bridges: Dict of data bridge widgets
        iframe_id: Unique identifier for the iframe
        output: Output widget for displaying JavaScript
        serialize_fn: Function to serialize Python objects to JSON
        max_result_size: Maximum size for results in bytes
        debug: Enable debug logging
        
    Returns:
        Handler function that can be attached to a button's on_click
    """
    import traceback
    
    def handler(_b):
        try:
            # Read request data from bridge
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

            # Call the callback
            cb = button_callbacks[button_id]
            result = cb(request_data, **callback_args)

            # Serialize and check size
            serialized = serialize_fn(result)
            json_str = json.dumps(serialized)
            size = len(json_str.encode("utf-8"))
            if size > max_result_size:
                serialized = {"type": "error", "message": f"Result too large: {size:,} bytes"}

            # Send to JavaScript
            send_to_javascript(iframe_id, serialized, output)

            # Clear the bridge
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
            send_to_javascript(iframe_id, error_result, output)

    return handler


def get_dispatcher_script(iframe_id: str) -> str:
    """
    Get the JavaScript dispatcher script.
    
    The dispatcher polls for queued requests and routes them to the
    appropriate hidden widgets to trigger Python callbacks.
    
    Args:
        iframe_id: Unique identifier for the iframe
        
    Returns:
        HTML string containing the dispatcher script
    """
    return f"""
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
</script>
"""


def get_communication_script(
    iframe_id: str,
    button_ids: List[str],
    initial_data: Dict[str, Any],
    debug: bool = False,
) -> str:
    """
    Get the JavaScript communication script for the iframe.
    
    This script handles button clicks in the iframe, collects input data,
    and posts messages to the parent window.
    
    Args:
        iframe_id: Unique identifier for the iframe
        button_ids: List of button IDs to attach listeners to
        initial_data: Initial data to make available in the iframe
        debug: Enable debug logging
        
    Returns:
        HTML string containing the communication script
    """
    button_ids_js = json.dumps(button_ids)
    initial_data_js = json.dumps(initial_data)
    debug_log = "console.log('[iframe]', ...args);" if debug else ""
    
    return f"""
<script>
(function() {{
  const iframeId = {json.dumps(iframe_id)};
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
