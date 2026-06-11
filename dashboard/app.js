(function () {
  const KEY = "mlx_middle_layer_api_key";

  function getKey() {
    return sessionStorage.getItem(KEY) || "";
  }

  function setStatus(msg, isErr) {
    const el = document.getElementById("status");
    el.textContent = msg;
    el.className = isErr ? "err" : "";
  }

  function authHeaders() {
    const k = getKey();
    const h = { Accept: "application/json" };
    if (k) h["X-API-Key"] = k;
    return h;
  }

  /** Path-absolute from site root so /dashboard and /dashboard/ both work. */
  function apiUrl(suffix) {
    return "/dashboard/api/" + suffix.replace(/^\//, "");
  }

  async function fetchJSON(path, opts) {
    opts = opts || {};
    let r;
    try {
      r = await fetch(path, {
        ...opts,
        headers: { ...authHeaders(), ...(opts.headers || {}) },
      });
    } catch (net) {
      const m = net && net.message ? net.message : String(net);
      const err = new Error(
        m +
          " — Check that middle_layerMLX is running, use http://127.0.0.1:5001/dashboard/ (with trailing slash), and same host/port as this page."
      );
      err.network = true;
      throw err;
    }
    const text = await r.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = { raw: text };
    }
    if (!r.ok) {
      const err = new Error((data && data.error) || r.statusText || String(r.status));
      err.status = r.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function esc(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderSnapshot(s) {
    const models = document.getElementById("models");
    const avail = (s.models_available || []).map((a) => `<span class="pill">${esc(a)}</span>`).join(" ");
    // Loaded models get an Unload button next to each pill so the
    // operator can free Metal RAM from the dashboard UI instead of
    // having to fire a DELETE /v1/models/<alias> by hand.
    const load = (s.models_loaded || [])
      .map(
        (a) =>
          `<span class="pill">${esc(a)} <button class="unload-btn" data-alias="${esc(a)}" title="Unload">×</button></span>`,
      )
      .join(" ");
    const errCount = Number(s.load_error_count || 0);
    const errSummary =
      errCount > 0
        ? `<p><strong>Recent load errors</strong> <span class="pill" style="background:#fee">${errCount}</span></p>`
        : "";
    models.innerHTML =
      "<p><strong>Loaded</strong> " +
      (load || "<em>none</em>") +
      "</p><p><strong>On disk</strong> " +
      (avail || "<em>none</em>") +
      "</p>" +
      errSummary;
    // Wire the Unload buttons after they're in the DOM.
    Array.from(models.querySelectorAll(".unload-btn")).forEach((btn) => {
      btn.addEventListener("click", async () => {
        const alias = btn.getAttribute("data-alias");
        if (!alias) return;
        if (!confirm(`Unload '${alias}' from MLX?`)) return;
        try {
          await fetchJSON("/v1/models/" + encodeURIComponent(alias), {
            method: "DELETE",
            headers: { "X-API-Key": getKey() },
          });
        } catch (e) {
          alert(e.message);
        }
        await refresh();
      });
    });

    const prefs = s.preferences || {};
    document.getElementById("defModel").value = prefs.default_model || "";
    const pr = prefs.swarm_presets || {};
    document.getElementById("presetsJson").value =
      Object.keys(pr).length > 0 ? JSON.stringify(pr, null, 2) : "";

    const act = s.active_by_alias || {};
    const actHtml =
      Object.keys(act).length === 0
        ? "<p><em>No active generations</em></p>"
        : "<table><thead><tr><th>Alias</th><th>Active</th></tr></thead><tbody>" +
          Object.entries(act)
            .map(([k, v]) => "<tr><td>" + esc(k) + "</td><td>" + esc(v) + "</td></tr>")
            .join("") +
          "</tbody></table>";
    document.getElementById("active").innerHTML = "<h2>Active generations</h2>" + actHtml;

    const evs = (s.events || []).slice().reverse();
    const rows = evs
      .map((e) => {
        const st = e.status === "ok" ? "ok" : "err";
        const tps = e.tokens_per_sec != null ? esc(e.tokens_per_sec) + " tok/s" : "—";
        const prev = e.preview && e.preview.last_user_preview ? esc(e.preview.last_user_preview).slice(0, 80) : "";
        return (
          "<tr>" +
          "<td><code>" +
          esc(e.request_id).slice(0, 8) +
          "…</code></td>" +
          "<td>" +
          esc(e.route_kind) +
          "</td>" +
          "<td>" +
          esc(e.resolved_model) +
          "</td>" +
          "<td>" +
          esc(e.completion_tokens) +
          "</td>" +
          "<td>" +
          esc(e.latency_ms) +
          "ms</td>" +
          "<td>" +
          tps +
          "</td>" +
          "<td class='" +
          st +
          "'>" +
          esc(e.status) +
          "</td>" +
          "<td title='" +
          esc(prev) +
          "'>" +
          (prev ? esc(prev) + (prev.length >= 80 ? "…" : "") : "—") +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
    document.getElementById("events").innerHTML =
      "<h2>Recent events</h2><table><thead><tr>" +
      "<th>Req</th><th>Route</th><th>Model</th><th>Out tok</th><th>Latency</th><th>Tok/s</th><th>Status</th><th>Preview</th>" +
      "</tr></thead><tbody>" +
      (rows || "<tr><td colspan='8'><em>No events yet</em></td></tr>") +
      "</tbody></table>";
  }

  async function refresh() {
    try {
      const s = await fetchJSON(apiUrl("snapshot"), { method: "GET" });
      renderSnapshot(s);
      setStatus("Updated " + new Date().toLocaleTimeString(), false);
    } catch (e) {
      if (e.status === 401) {
        setStatus("Unauthorized: set API key if MIDDLE_LAYER_API_KEY is configured.", true);
      } else if (e.network) {
        setStatus(e.message, true);
      } else {
        setStatus("Error: " + e.message, true);
      }
    }
  }

  document.getElementById("saveKey").addEventListener("click", function () {
    const v = document.getElementById("apiKey").value.trim();
    if (v) sessionStorage.setItem(KEY, v);
    else sessionStorage.removeItem(KEY);
    refresh();
  });

  document.getElementById("apiKey").value = getKey();

  document.getElementById("btnLoad").addEventListener("click", async function () {
    const alias = document.getElementById("loadAlias").value.trim();
    if (!alias) return;
    try {
      await fetchJSON(apiUrl("models/load"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alias: alias }),
      });
      document.getElementById("loadAlias").value = "";
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });

  document.getElementById("btnPrefs").addEventListener("click", async function () {
    const def = document.getElementById("defModel").value.trim();
    let presets = null;
    const raw = document.getElementById("presetsJson").value.trim();
    if (raw) {
      try {
        presets = JSON.parse(raw);
        if (typeof presets !== "object" || presets === null || Array.isArray(presets)) throw new Error("must be object");
      } catch (err) {
        alert("Invalid JSON: " + err.message);
        return;
      }
    }
    try {
      await fetchJSON(apiUrl("preferences"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ default_model: def || null, swarm_presets: presets }),
      });
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });

  refresh();
  setInterval(refresh, 2000);
})();
