/**
 * dispatcher.js
 * 
 * Moves queued iframe clicks into the correct hidden widget bridge
 * and triggers the corresponding poll button to execute Python callbacks.
 * 
 * Placeholders (replaced via Python .format()):
 *   {{iframe_id}} - Unique iframe identifier (JSON-encoded string)
 */
(function() {{
  const iframeId = {iframe_id};
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
