/* Vectra web UI. No dependencies. All dynamic text goes through textContent: alt text and URLs
   from crawled pages are untrusted and must never be parsed as HTML. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = {
    drop: $("#drop"), dropEmpty: $("#drop-empty"), queryImg: $("#query-img"), file: $("#file"),
    k: $("#k"), kOut: $("#k-out"), flip: $("#flip"), exclude: $("#exclude"),
    status: $("#status"), grid: $("#grid"), empty: $("#empty"), emptyText: $("#empty-text"),
    samples: $("#samples"), facts: $("#facts"),
    dlg: $("#detail"), dClose: $("#d-close"), dQueryFig: $("#d-query-fig"), dQuery: $("#d-query"),
    dMatch: $("#d-match"), dCap: $("#d-cap"), dMeta: $("#d-meta"), dSimilar: $("#d-similar"),
    dOpen: $("#d-open"), dCompare: document.querySelector(".compare"),
  };

  const KIND_LABEL = { "exact": "Same file", "near-duplicate": "Near-duplicate", "likely-copy": "Likely copy", "similar": "Similar" };
  const state = { results: [], current: -1, query: null, rerun: null, token: 0, objectUrl: null };

  // ---- helpers --------------------------------------------------------------------------
  const h = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  async function api(path, opts) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status})`);
    return body;
  }

  function setStatus(text, isError) {
    el.status.textContent = text || "";
    el.status.classList.toggle("error", !!isError);
  }

  const params = () => new URLSearchParams({
    k: el.k.value, flip: el.flip.checked ? "1" : "0", exclude_exact: el.exclude.checked ? "1" : "0",
  });

  // ---- query preview --------------------------------------------------------------------
  function showQuery(src, label) {
    state.query = { src, label };
    el.queryImg.src = src;
    el.queryImg.hidden = false;
    el.dropEmpty.hidden = true;
    el.drop.classList.add("has-image");
    el.drop.setAttribute("aria-label", `${label}. Activate to choose a different image.`);
  }

  // ---- searching ------------------------------------------------------------------------
  async function run(task) {
    const token = ++state.token;
    setStatus("Searching…");
    try {
      const data = await task();
      if (token !== state.token) return;   // a newer search superseded this one
      render(data.results);
    } catch (e) {
      if (token !== state.token) return;
      setStatus(e.message, true);
    }
  }

  function searchFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      setStatus("That file isn't an image. Use a JPEG, PNG, WebP, GIF, BMP or TIFF.", true);
      return;
    }
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    showQuery(state.objectUrl, file.name || "Pasted image");
    state.rerun = () => run(() => api(`/api/search?${params()}`, {
      method: "POST", headers: { "X-Vectra": "1", "Content-Type": file.type || "application/octet-stream" }, body: file,
    }));
    state.rerun();
  }

  function searchSimilar(id) {
    showQuery(`/thumb/${id}?s=640`, `Library image ${id}`);
    state.rerun = () => run(() => api(`/api/similar/${id}?${new URLSearchParams({ k: el.k.value, flip: el.flip.checked ? "1" : "0" })}`));
    state.rerun();
  }

  // ---- results --------------------------------------------------------------------------
  function render(results) {
    state.results = results;
    el.grid.replaceChildren();
    el.empty.hidden = results.length > 0;
    el.grid.hidden = results.length === 0;
    if (!results.length) {
      el.emptyText.textContent = "Nothing in the library resembles this image.";
      setStatus("");
      return;
    }
    results.forEach((r, i) => {
      const li = h("li");
      const btn = h("button", "tile");
      btn.type = "button";
      btn.dataset.kind = r.kind;
      btn.setAttribute("aria-label", `Match ${i + 1}: ${KIND_LABEL[r.kind] || r.kind}, similarity ${r.score.toFixed(2)}. Open comparison.`);

      const frame = h("div", "frame");
      const img = h("img");
      img.src = r.thumb;
      img.alt = r.alt || r.name;
      img.loading = "lazy";
      frame.append(img);

      const meta = h("div", "meta");
      const score = h("div", "score");
      score.append(h("span", "num", r.score.toFixed(2)), h("span", "kind", KIND_LABEL[r.kind] || r.kind));
      const bar = h("div", "bar");
      const fill = h("i");
      fill.style.setProperty("--s", String(Math.max(0, Math.min(1, r.score))));
      bar.append(fill);
      const name = h("div", "name", r.name);
      name.title = r.name;
      meta.append(score, bar, name);

      btn.append(h("span", "rank", String(i + 1)), frame, meta);
      btn.addEventListener("click", () => openDetail(i));
      li.append(btn);
      el.grid.append(li);
    });
    el.grid.classList.remove("fresh");
    void el.grid.offsetWidth;            // restart the one-shot fade
    el.grid.classList.add("fresh");
    const exact = results.filter((r) => r.kind === "exact").length;
    const near = results.filter((r) => r.kind === "near-duplicate" || r.kind === "likely-copy").length;
    const bits = [`${results.length} matches`];
    if (exact) bits.push(`${exact} identical file${exact > 1 ? "s" : ""}`);
    if (near) bits.push(`${near} copy or near-duplicate${near > 1 ? "s" : ""}`);
    setStatus(bits.join(", "));
  }

  // ---- compare dialog -------------------------------------------------------------------
  function metaRow(label, value, href) {
    const dt = h("dt", null, label);
    const dd = h("dd");
    if (href) {
      const a = h("a", null, value);
      a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
      dd.append(a);
    } else {
      dd.textContent = value;
    }
    el.dMeta.append(dt, dd);
  }

  function openDetail(i) {
    const r = state.results[i];
    if (!r) return;
    state.current = i;
    el.dMatch.src = r.full;
    el.dMatch.alt = r.alt || r.name;
    el.dCap.textContent = `Library match ${i + 1} of ${state.results.length}`;
    if (state.query) {
      el.dQuery.src = state.query.src;
      el.dQueryFig.hidden = false;
      el.dCompare.classList.remove("single");
    } else {
      el.dQueryFig.hidden = true;
      el.dCompare.classList.add("single");
    }
    el.dMeta.replaceChildren();
    metaRow("Match type", KIND_LABEL[r.kind] || r.kind);
    metaRow("Similarity", r.score.toFixed(3));
    if (typeof r.phash_distance === "number") metaRow("Fingerprint", `${r.phash_distance} of 64 bits differ`);
    metaRow("Size", `${r.width} × ${r.height} px`);
    metaRow("File", r.name);
    if (r.source_url) metaRow("Found at", r.source_url, r.source_url);
    if (r.page_url) metaRow("On page", r.page_url, r.page_url);
    if (r.alt) metaRow("Alt text", r.alt);
    el.dOpen.href = r.full;
    el.dSimilar.onclick = () => { el.dlg.close(); searchSimilar(r.id); };
    if (!el.dlg.open) el.dlg.showModal();
  }

  function step(delta) {
    const n = state.results.length;
    if (!n || !el.dlg.open) return;
    openDetail((state.current + delta + n) % n);
  }

  el.dClose.addEventListener("click", () => el.dlg.close());
  el.dlg.addEventListener("click", (e) => { if (e.target === el.dlg) el.dlg.close(); });
  el.dlg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  });

  // ---- input: drop, choose, paste -------------------------------------------------------
  el.drop.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => { if (el.file.files[0]) searchFile(el.file.files[0]); el.file.value = ""; });
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) searchFile(f); });
  document.addEventListener("paste", (e) => {
    const files = e.clipboardData && e.clipboardData.files;
    if (files && files.length) { e.preventDefault(); searchFile(files[0]); }
  });
  // stop the browser navigating to a file dropped outside the zone
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));

  // ---- options re-run the current search ------------------------------------------------
  let timer = null;
  const rerunSoon = () => { if (state.rerun) { clearTimeout(timer); timer = setTimeout(state.rerun, 220); } };
  el.k.addEventListener("input", () => { el.kOut.textContent = el.k.value; rerunSoon(); });
  el.flip.addEventListener("change", rerunSoon);
  el.exclude.addEventListener("change", rerunSoon);

  // ---- library facts + starter images -----------------------------------------------------
  function fact(label, value) {
    const s = h("span");
    s.append(h("b", null, String(value)), document.createTextNode(" " + label));
    return s;
  }

  async function init() {
    try {
      const info = await api("/api/info");
      el.facts.replaceChildren(
        fact(info.images === 1 ? "image" : "images", info.images.toLocaleString()),
        fact("features", info.backend),
        fact("model", info.active_model),
        ...(info.index_kind ? [fact("index", info.index_kind)] : []),
      );
      if (!info.images) {
        el.emptyText.replaceChildren(
          document.createTextNode("This library is empty. Add images with "),
          h("code", null, "vectra ingest <folder>"),
          document.createTextNode(", then reload."),
        );
        return;
      }
      const { samples } = await api("/api/samples?n=8");
      if (samples.length) {
        el.samples.setAttribute("role", "group");
        samples.forEach((s) => {
          const b = h("button", "sample");
          b.type = "button";
          b.setAttribute("aria-label", `Search with library image ${s.id}`);
          const img = h("img");
          img.src = s.thumb; img.alt = "";
          b.append(img);
          b.addEventListener("click", () => searchSimilar(s.id));
          el.samples.append(b);
        });
        el.emptyText.textContent = "Drop an image to find the same picture, or close edits of it, in your library. Or start from one of these:";
      }
    } catch (e) {
      setStatus(e.message, true);
    }
  }
  init();
})();
