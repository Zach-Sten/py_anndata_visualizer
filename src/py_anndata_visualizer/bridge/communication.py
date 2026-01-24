"""
Bridge module for linking HTML buttons to Python callbacks.
Handles widget-based communication between iframe and Jupyter kernel.
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


def get_communication_script(iframe_id: str, button_ids_js: str, initial_data_js: str, debug: bool = False) -> str:
    """
    Generate the JavaScript communication script that gets injected into the iframe.
    This script captures button clicks and posts messages to the parent window.
    """
    return f"""
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


def get_dispatcher_script(iframe_id: str) -> str:
    """
    Generate the JavaScript dispatcher script that runs in the parent window.
    This script moves queued iframe clicks into widget bridges and triggers Python callbacks.
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
