<!-- {"title": "The local API", "group": "Orchestration", "order": 22, "lead": "A thin FastAPI wrapper around the pipeline for the demo frontend: upload a file, get the same explanation record the runner writes. It is not on the scoring path.", "sources": ["src/hearsay/api.py", "docs/handoffs/2026-09-26_frontend-contract.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The demo needs to score one file on demand and read back records from a finished run. The API does exactly that and nothing more: models load lazily on the first request and stay loaded, one clip is analysed at a time, and every response is the runner's own record.</p>
<h2 id="io">What goes in and comes out</h2>
<ul>
<li><code>POST /analyze</code>: a multipart upload; returns the explanation record plus the API's own timing. An undecodable upload returns an "undetermined" record rather than an error.</li>
<li><code>GET /results/{filename}</code> and <code>GET /results</code>: records from a results directory named by an environment variable.</li>
<li><code>GET /health</code>: whether models are loaded, the rule, the policy, the scorers and the version block.</li>
</ul>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/api.py|src/hearsay/api.py}}. Start it with <code>uv run uvicorn hearsay.api:app --port {{num:8000|docs/handoffs/2026-09-26_frontend-contract.md}}</code>.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>Settings come from the environment: rule, policy, detectors, constants file, model directories, results directory, thread count ({{src:src/hearsay/api.py#L82|api.py, lines 82–105}}).</li>
<li>Models load under a lock on the first analyse call; the M5 hash gate runs before and after; a load failure is remembered so the API does not retry per request ({{src:src/hearsay/api.py#L108|api.py, lines 108–131}}).</li>
<li>Results lookups accept bare file names only and answer with an error status when the directory is unset or the name is absent ({{src:src/hearsay/api.py#L136|api.py, lines 136–143}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Built and tested. A separate toy analyser that predates the real pipeline still exists in the package; the frontend contract says it must be rewired away from and must never export a TSV.</p>
<h2 id="numbers">Numbers</h2>
<p>None.</p>
<h2 id="limits">Known limits</h2>
<p>Single-clip concurrency by design. No authentication; it is a local demo server.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_api.py</code>: health without loading models, probe-only mode, results lookup and its environment requirement, analyse uses the pipeline, rule and policy from the environment, the default rule mapped once, an undecodable upload is undetermined, health reports the file's scorers, M5 loads only when the constants need it.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/handoffs/2026-09-26_frontend-contract.md|The frontend contract}}; {{src:docs/handoffs/2026-09-26_frontend-wiring-handoff.md|the frontend wiring handoff}}.</p>
