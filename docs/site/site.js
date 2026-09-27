/* HEARSAY documentation site: lazy search, tooltip dismissal, cost explorer.
   Classic script, no modules, no network. Everything degrades: the pages read without it. */
(function () {
  "use strict";
  /* ---- mobile navigation --------------------------------------------------------------- */
  var menu = document.querySelector(".menu-toggle");
  var nav = document.getElementById("site-nav");
  function closeMenu() {
    if (!menu) return;
    document.body.classList.remove("nav-open");
    menu.setAttribute("aria-expanded", "false");
  }
  if (menu && nav) {
    menu.addEventListener("click", function () {
      var open = menu.getAttribute("aria-expanded") !== "true";
      document.body.classList.toggle("nav-open", open);
      menu.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.addEventListener("click", function (ev) {
      if (ev.target.closest("a")) closeMenu();
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && menu.getAttribute("aria-expanded") === "true") {
        closeMenu();
        menu.focus();
      }
    });
    window.addEventListener("resize", function () {
      if (window.innerWidth >= 900) closeMenu();
    });
  }

  /* ---- tooltips: Escape closes the focused one ------------------------------------------ */
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && document.activeElement && document.activeElement.classList.contains("term")) {
      document.activeElement.blur();
    }
  });
  var terms = document.querySelectorAll(".term");
  for (var i = 0; i < terms.length; i++) {
    terms[i].addEventListener("focus", function (ev) {
      var r = ev.target.getBoundingClientRect();
      if (window.innerWidth <= 640 && r.bottom > window.innerHeight - 140) {
        window.scrollBy({ top: r.bottom - (window.innerHeight - 160), behavior: "smooth" });
      }
    });
  }

  /* ---- search --------------------------------------------------------------------------- */
  var box = document.getElementById("q");
  var panel = document.getElementById("search-results");
  var indexUrl = document.body.getAttribute("data-search-index");
  var loading = false;
  var pendingIndexCallbacks = [];

  function loadIndex(cb) {
    if (window.HEARSAY_INDEX) { cb(); return; }
    pendingIndexCallbacks.push(cb);
    if (loading) return;
    loading = true;
    var s = document.createElement("script");
    s.src = indexUrl;
    s.onload = function () {
      loading = false;
      var callbacks = pendingIndexCallbacks.splice(0);
      for (var i = 0; i < callbacks.length; i++) callbacks[i]();
    };
    s.onerror = function () { loading = false; pendingIndexCallbacks.length = 0; };
    document.head.appendChild(s);
  }

  function lowerBound(arr, key) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function search(query) {
    var idx = window.HEARSAY_INDEX;
    if (!idx) return [];
    var words = query.toLowerCase().match(/[a-z0-9_]+/g) || [];
    if (!words.length) return [];
    var termList = idx._terms || (idx._terms = Object.keys(idx.terms));
    var scores = null;
    for (var w = 0; w < words.length; w++) {
      var word = words[w];
      var hits = {};
      var start = lowerBound(termList, word);
      for (var t = start; t < termList.length && termList[t].indexOf(word) === 0; t++) {
        var pages = idx.terms[termList[t]];
        var exact = termList[t] === word ? 2 : 1;
        for (var p = 0; p < pages.length; p++) hits[pages[p]] = (hits[pages[p]] || 0) + exact;
      }
      if (scores === null) { scores = hits; continue; }
      var merged = {};
      for (var k in scores) if (hits[k]) merged[k] = scores[k] + hits[k];
      scores = merged;
    }
    var out = [];
    for (var id in scores) {
      var page = idx.pages[id];
      var bonus = 0;
      var tl = page.t.toLowerCase(), hl = page.h.toLowerCase();
      for (var w2 = 0; w2 < words.length; w2++) {
        if (tl.indexOf(words[w2]) >= 0) bonus += 10;
        if (hl.indexOf(words[w2]) >= 0) bonus += 3;
      }
      out.push({ page: page, score: scores[id] + bonus });
    }
    out.sort(function (a, b) { return b.score - a.score || a.page.t.localeCompare(b.page.t); });
    return out.slice(0, 40);
  }

  function relativeTo(url) {
    // page urls in the index are site-relative; section pages live one directory down
    var section = document.body.getAttribute("data-section");
    var base = (section === "specs" || section === "library") ? "../" : "";
    return base + url;
  }

  function render(results, query) {
    if (!panel) return;
    if (!query) { panel.hidden = true; panel.innerHTML = ""; return; }
    var html = "";
    if (!results.length) {
      html = '<p class="s-none">No page matches "' + escapeHtml(query) + '".</p>';
    } else {
      html = "<ol>";
      for (var i = 0; i < results.length; i++) {
        var p = results[i].page;
        html += '<li><a href="' + escapeHtml(relativeTo(p.u)) + '">' + escapeHtml(p.t) + "</a>" +
          '<span class="s-section">' + escapeHtml(p.s) + "</span>" +
          (p.h ? '<span class="s-head">' + escapeHtml(p.h.slice(0, 140)) + "</span>" : "") + "</li>";
      }
      html += "</ol>";
    }
    panel.innerHTML = html;
    panel.hidden = false;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  if (box && panel) {
    box.addEventListener("focus", function () { loadIndex(function () {}); });
    box.addEventListener("input", function () {
      var q = box.value.trim();
      if (!q) { render([], ""); return; }
      loadIndex(function () { render(search(q), q); });
    });
    box.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") { box.value = ""; render([], ""); box.blur(); }
    });
    document.addEventListener("click", function (ev) {
      if (!panel.hidden && !panel.contains(ev.target) && ev.target !== box) render([], "");
    });
  }

  /* ---- cost explorer -------------------------------------------------------------------- */
  var explorer = document.querySelector(".cost-explorer");
  if (explorer) {
    var expr = explorer.getAttribute("data-expr");
    var pfa = document.getElementById("pfa");
    var pmiss = document.getElementById("pmiss");
    var pfaOut = document.getElementById("pfa-out");
    var pmissOut = document.getElementById("pmiss-out");
    var costOut = document.getElementById("cost-out");
    var fn;
    try { fn = new Function("pfa", "pmiss", "return " + expr + ";"); } catch (e) { fn = null; }
    function update() {
      if (!fn) return;
      var a = Number(pfa.value) / 100, b = Number(pmiss.value) / 100;
      pfaOut.textContent = pfa.value + "%";
      pmissOut.textContent = pmiss.value + "%";
      costOut.textContent = fn(a, b).toFixed(3);
    }
    pfa.addEventListener("input", update);
    pmiss.addEventListener("input", update);
    update();
  }

  /* Open the source-backed appendix when the deck links to it. */
  var appendix = document.getElementById("full-docs");
  if (appendix) {
    function openAppendixFromHash() {
      if (window.location.hash === "#full-docs") {
        appendix.open = true;
        appendix.scrollIntoView({ block: "start" });
      }
    }
    window.addEventListener("hashchange", openAppendixFromHash);
    openAppendixFromHash();
  }
})();
