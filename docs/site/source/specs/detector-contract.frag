<!-- {"title": "Detector contract and safe runner", "group": "Ingest", "order": 11, "lead": "Every detector takes the same read-only clip and returns the same kind of result: a score that rises with synthetic likelihood, one sentence of evidence, named features and a status. A wrapper guarantees that a crashing detector becomes a neutral, flagged result and never costs a file its row.", "sources": ["src/hearsay/detectors/base.py", "CLAUDE.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Ten detectors from four people had to plug into one pipeline in a weekend. The contract is the agreement that made that possible: what a detector gets, what it must return, and what happens when it fails. The rules are enforced, not assumed: a score outside [0, 1] or a missing evidence sentence raises an error rather than being quietly fixed, so a detector cannot slip a bad value into fusion.</p>
<h2 id="io">What goes in and comes out</h2>
<ul>
<li><strong>In:</strong> a <code>ClipContext</code>: the path, the decoded audio (read-only, decoded once), the header probe (computed once), and a per-clip memo cache that detectors share expensive intermediates through.</li>
<li><strong>Out:</strong> a <code>DetectorResult(name, score, evidence, features, status, error)</code>.</li>
<li><strong>Through:</strong> <code>safe_run(detector, ctx)</code>, which never raises.</li>
</ul>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/base.py|src/hearsay/detectors/base.py}}: <code>ClipContext</code>, <code>DetectorResult</code>, the <code>Detector</code> protocol, <code>safe_run</code>, and the <code>Registry</code>. The engineered detectors register themselves through {{src:src/hearsay/detectors/engineered.py|engineered.py}}.</p>
<h2 id="params">Rules and where each is set</h2>
<ul>
<li><code>score</code> must be a real number, not a bool, finite, in [0, 1]; violations raise <code>ValueError</code> and values are never clipped ({{src:src/hearsay/detectors/base.py#L125|base.py, lines 125–127}}).</li>
<li><code>evidence</code> must be a non-empty string; <code>features</code> a mapping of non-empty names to finite floats (copied, so a detector cannot mutate a result after the fact); <code>status</code> one of <code>ok</code>, <code>skipped</code>, <code>error</code>, and only an error carries an error string ({{src:src/hearsay/detectors/base.py#L128|base.py, lines 128–143}}).</li>
<li><strong>The neutral score</strong> is {{num:0.5|src/hearsay/detectors/base.py#L40}}. <code>safe_run</code> turns a raised exception, a wrong return type, a name mismatch or a non-ok status from <code>run</code> into <code>status="error"</code> at that score with the evidence "detector failed: …"; a detector whose <code>applies()</code> is false is <code>skipped</code> without running ({{src:src/hearsay/detectors/base.py#L163|base.py, lines 163–192}}). Keyboard interrupts still propagate.</li>
<li><strong>The audio is decoded once</strong> and marked read-only; a decode failure is cached and re-raised without a second FFmpeg call ({{src:src/hearsay/detectors/base.py#L70|base.py, lines 70–84}}).</li>
<li><strong>The registry</strong> rejects duplicate or empty names and iterates in sorted name order, so fusion columns are stable ({{src:src/hearsay/detectors/base.py#L195|base.py, lines 195–217}}).</li>
<li><strong>Detectors never read filenames or filesystem timestamps</strong>, because git, zip and Docker rewrite them (a project rule in {{src:CLAUDE.md#detector-contract|CLAUDE.md}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Built and frozen. A change to <code>DetectorResult</code> fields must name the affected detector owners; none was made after the D-track began.</p>
<h2 id="numbers">Numbers</h2>
<p>Every learned detector exports one file per detector with columns path, fold, split, score and logit over the same {{num:21,671|README.md#the-detector-contract}} rows (the training clips and the test files); fusion reads only those exports, so any detector can be swapped or ablated without touching the others.</p>
<h2 id="limits">Known limits</h2>
<p>The neutral {{num:0.5|src/hearsay/detectors/base.py#L40}} on a skipped or errored result is a placeholder, not evidence: the fusion code imputes such a column at its training centre instead (see <a href="fusion-rule.html">the fusion rule</a>). The role labels in the runner's code tag Spectra-AASIST as "fused" although the rule only lets it suppress; the reports and this site say "suppressor".</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_detector_contract.py</code>, thirty-odd cases named B1–B9: invalid scores rejected and never clipped, valid scores coerced to float, score direction, exceptions become error results, bad return values become errors, evidence required, bad features rejected, registry rejects duplicates and sorts, the same clip scored twice is identical, audio is read-only, decoded once, decode failure cached, memo computes once, not-applicable is skipped, results unhashable but comparable, status and error coherence. <code>tests/test_engineered_registry.py</code> pins the seven registered engineered names.</p>
<h2 id="sources">Sources</h2>
<p>{{src:CLAUDE.md#detector-contract|CLAUDE.md, "Detector contract"}}; {{src:docs/architecture.md#5-the-detector-contract|Architecture §5}}; {{src:README.md#the-detector-contract|the README}}.</p>
