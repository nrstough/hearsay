<!-- {"title": "The routing log and explanation JSON", "group": "Orchestration", "order": 21, "lead": "For every file the runner writes one JSON record: every detector's score, status, sentence and features, the fusion inputs and arithmetic, the routing log, and the version of everything that produced it. It is the explainability deliverable and the frontend's data.", "sources": ["src/hearsay/pipeline.py", "docs/handoffs/2026-09-26_frontend-contract.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The brief asks for explainable results: which techniques said what, and why the file got its number. The record answers that in plain sentences a person can check, one per detector, plus a routing log that says what each rule decided. The eight worked examples on <a href="../how-it-decides.html#eight-files">How it decides</a> are these records, printed.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the outputs of <code>analyze_clip</code> for one file. Out: <code>results/&lt;basename&gt;.json</code>, and the same object as the API's response.</p>
<h2 id="where">Where it lives</h2>
<p><code>analyze_clip</code>, <code>detector_entry</code>, <code>fusion_block</code>, <code>routing_log</code> and <code>fusion_line</code> in {{src:src/hearsay/pipeline.py#L537|pipeline.py, lines 537–750}}.</p>
<h2 id="params">The record's shape</h2>
<ul>
<li><strong>Top level:</strong> <code>filename</code>, <code>duration_s</code>, <code>probability_synthetic</code>, <code>verdict</code>, <code>is_speech</code>, <code>default_answer_applied</code>, <code>fusion</code>, <code>detectors</code>, <code>routing_log</code>, <code>flag</code> (empty or <code>decode_error</code>), <code>seconds</code>, <code>version</code> ({{src:src/hearsay/pipeline.py#L737|pipeline.py, lines 737–750}}).</li>
<li><strong>Each detector entry:</strong> <code>name</code>, <code>role</code> (fused, evidence, routing, gate), <code>status</code>, <code>score</code>, <code>evidence</code>, <code>features</code>, <code>seconds</code>, <code>error</code>.</li>
<li><strong>The fusion block:</strong> <code>rule</code>, <code>inputs</code>, <code>weights</code>, <code>terms</code>, <code>fused</code>, <code>p_fused</code>, <code>imputed</code>, a <code>detail</code> with the rule's name, the base rank and whether the Spectra step fired, and the block bounds ({{src:src/hearsay/pipeline.py#L671|pipeline.py, lines 671–674}}).</li>
<li><strong>The routing log</strong> is a list of lines in a fixed order: decode, container, compression, enf, splice, speaker_drift, speech_gate, fusion. The fusion line prints the weighted terms and the Spectra outcome, for example "M3 margin +{{num:9.50|docs/reports/2026-09-26_worked-examples.md#4-hgt1046947wav-the-deep-detectors-disagree-drift-flagged-p--0003-z-mean-0686}}, no suppression (M3 never promotes)" ({{src:src/hearsay/pipeline.py#L574|pipeline.py, lines 574–596}}).</li>
<li><strong>The version block</strong> names the git commit, every model directory with the M5 checkpoint's hash prefixes, the constants file, the rule, the policy and the polarity, so a record can be traced to the exact build that made it ({{src:src/hearsay/pipeline.py#L730|pipeline.py, lines 730–736}}).</li>
<li><strong>The verdict</strong> is "undetermined" if the file did not decode or has no speech, else "synthetic" when the pre-map fused probability is at least {{num:0.5|src/hearsay/pipeline.py#L652}}, else "real" ({{src:src/hearsay/pipeline.py#L650|pipeline.py, lines 650–653}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Built; contract version v0. The frontend reads a static dump of these records for the test set under the shipped rule, and the live API returns the same object.</p>
<h2 id="numbers">Numbers</h2>
<p>None of its own; the record carries the detectors' numbers.</p>
<h2 id="limits">Known limits</h2>
<p>The verdict label ignores <code>--flip</code> and uses the pre-map probability; the ranking the metric scores is <code>probability_synthetic</code>. The frontend contract's line reference into the gate module is stale.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_pipeline.py</code>: the response shape and required version keys, the routing-log order and wording, the fusion line lists every weighted term, the verdict rule.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/handoffs/2026-09-26_frontend-contract.md|The frontend contract}}; {{src:docs/reports/2026-09-26_worked-examples.md|the worked examples}}; {{src:docs/reports/2026-09-26_runner-docker.md|the runner report}}.</p>
