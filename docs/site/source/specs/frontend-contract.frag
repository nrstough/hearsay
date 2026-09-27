<!-- {"title": "The frontend contract", "group": "Delivery", "order": 81, "lead": "The demo frontend is a separate application built by another team member. What it may rely on is written down as a contract over the explanation record, so the UI and the pipeline can move independently. This page documents the contract; the frontend itself is outside this site.", "sources": ["docs/handoffs/2026-09-26_frontend-contract.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The frontend shows a file's verdict, each detector's sentence and the routing log. Rather than reading the pipeline's internals, it reads one JSON object per file, the same one the runner writes and the API returns, and the contract says which fields mean what and what will not change under it.</p>
<h2 id="io">What the UI may rely on</h2>
<ul>
<li>Every detector's <code>score</code> is in [0, 1] and higher means synthetic; <code>evidence</code> is never empty; <code>status</code> is ok, skipped or error, and an errored detector carries a neutral score and an error string.</li>
<li><code>role</code> is <code>fused</code>, <code>evidence</code> (ENF, splice, speaker drift, compression), <code>routing</code> (container) or <code>gate</code> (speech gate).</li>
<li><code>verdict</code> is synthetic at or above {{num:0.5|docs/handoffs/2026-09-26_frontend-contract.md}}, real below, and undetermined when there is no speech or the decode failed.</li>
<li>For the test set, <code>probability_synthetic</code> equals the logged TSV value.</li>
<li>Under the shipped rule, <code>fusion.weights</code> names four inputs (Spectra at zero) and <code>fusion.detail.final</code> names the rule; the UI renders them from the JSON rather than hard-coding them.</li>
</ul>
<h2 id="where">Where it lives</h2>
<p>{{src:docs/handoffs/2026-09-26_frontend-contract.md|The contract}}, with the example schema and the rewire note. The static dump the UI reads is the shipped rule's full-set run; the live path is <a href="local-api.html">the local API</a>.</p>
<h2 id="params">Boundaries</h2>
<ul>
<li>No LLM or hosted service on the scoring path; the "copilot" may not present canned text as analysis.</li>
<li>The frontend lives in its own directory and does not edit other lanes' files.</li>
<li>A toy analyser that predates the pipeline must be moved out of the package and must never export a TSV.</li>
</ul>
<h2 id="status">Status</h2>
<p>Contract v0 written about 06:30 on Saturday; a rewire note added at 09:50 pointing the UI at the real API and the shipped rule's dump.</p>
<h2 id="numbers">Numbers</h2>
<p>None; the UI displays the pipeline's.</p>
<h2 id="limits">Known limits</h2>
<p>The contract's verdict rule reads the final probability while the code compares the pre-map value; they agree on every judged file but the wording differs. Its line reference into the gate module is stale.</p>
<h2 id="tests">Tests that pin it</h2>
<p>The contract's guarantees are the pipeline's tests: the response shape, the role labels, the verdict rule and the map applied once in <code>tests/test_pipeline.py</code>; the API's behaviour in <code>tests/test_api.py</code>.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/handoffs/2026-09-26_frontend-contract.md|The frontend contract}}; {{src:docs/handoffs/2026-09-26_frontend-wiring-handoff.md|the wiring handoff}}.</p>
