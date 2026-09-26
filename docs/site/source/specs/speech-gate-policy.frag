<!-- {"title": "The speech gate and the default-answer policy", "group": "Orchestration", "order": 23, "lead": "Five cheap cues decide whether a file contains speech to judge. A file that fails skips fusion and is placed in a block strictly below every judged file, at the real end of the ranking, where an uncertain file belongs under the sponsor's cost.", "sources": ["src/hearsay/detectors/speech_gate.py", "docs/reports/2026-09-26_gate-and-drift.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Our first deep probe scored pure silence at {{num:0.99|README.md#orchestration-what-routing-changes}} synthetic. Under a cost where a false alarm counts {{num:9.33|README.md#how-it-decides}} misses, a handful of silent or musical files at the top of the ranking would cost more than the entire fusion decision. The gate catches such files; the policy puts them where a "we don't know" belongs. The thresholds were set beyond the extreme value of every cue in the full test set, so the gate touches {{num:0 of 1,671|README.md#the-eight-forensic-techniques}} test files and only ever catches silence, tones, static and noise.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the prepared clip. Out: a <code>speech_gate</code> result whose score is always the neutral value (fusion never sees it), with features for the five cues, the reasons that fired and <code>is_speech</code>; and, from the policy, the final placement of every file's probability.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/speech_gate.py|src/hearsay/detectors/speech_gate.py}}: <code>analyze</code>, the detector, and <code>apply_default_answer</code>. The runner applies the policy exactly once in <code>final_score</code> ({{src:src/hearsay/pipeline.py#L656|pipeline.py, lines 656–668}}).</p>
<h2 id="params">The five cues and where each is set</h2>
<ul>
<li><strong>Silence:</strong> RMS below {{num:−70.0|src/hearsay/detectors/speech_gate.py#L47}} dBFS.</li>
<li><strong>Unvoiced:</strong> fewer than {{num:0.05|src/hearsay/detectors/speech_gate.py#L48}} of frames voiced (a frame is voiced when it is loud enough and its pitch is in the speech range).</li>
<li><strong>Tone:</strong> voiced, but the spread of log pitch is below {{num:0.02|src/hearsay/detectors/speech_gate.py#L49}}.</li>
<li><strong>Static:</strong> loudness standard deviation below {{num:3.0|src/hearsay/detectors/speech_gate.py#L50}} dB.</li>
<li><strong>Noise:</strong> long-term spectral flatness at or above {{num:0.5|src/hearsay/detectors/speech_gate.py#L51}}.</li>
</ul>
<p>Any one cue firing gates the file. The test set's extremes that sit behind the thresholds: quietest file {{num:−34.6|src/hearsay/detectors/speech_gate.py#L15}} dBFS, lowest voiced fraction {{num:0.099|src/hearsay/detectors/speech_gate.py#L15}}, lowest loudness spread {{num:6.0|src/hearsay/detectors/speech_gate.py#L15}} dB, highest flatness {{num:0.39|src/hearsay/detectors/speech_gate.py#L15}} ({{src:src/hearsay/detectors/speech_gate.py#L14|speech_gate.py, lines 14–16}}). Clips shorter than half a second are zero-padded; pitch comes from YIN between 60 and {{num:400|src/hearsay/detectors/speech_gate.py#L76}} Hz.</p>
<h3 id="policy">The default-answer policy</h3>
<ul>
<li>Judged files are mapped into [{{num:0.001|src/hearsay/detectors/speech_gate.py#L52}}, 1] (the determinate map, applied once).</li>
<li>Gated files that decoded land in [1e-4, 1e-3): ordered by a weak signal (the probe's own probability when it exists), with a {{num:0.01|src/hearsay/detectors/speech_gate.py#L54}} share of hash jitter from the filename so no two files tie.</li>
<li>Files that failed to decode, or are missing, land in [0, 1e-4), at the very bottom ({{src:src/hearsay/detectors/speech_gate.py#L132|speech_gate.py, lines 132–162}}).</li>
<li>Only the gate's own flag gates a file; a gate error does not.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built about 06:10 on Saturday; the block's formula was corrected at 08:30 when the pipeline turned out to score some real files as low as {{num:0.0009|docs/reports/2026-09-26_gate-and-drift.md}}, which the first version would have overlapped. The architecture document still shows the first formula.</p>
<h2 id="numbers">Numbers</h2>
<p>Gated: {{num:0 of 1,671|README.md#the-eight-forensic-techniques}} test files; 31 holdout rows and 9 In-the-Wild rows in the ablation, costing {{num:0.001|README.md#orchestration-what-routing-changes}} under the brief's cost on In-the-Wild. Placement is optimal under the brief's cost when the block's fake rate is below {{num:80%|README.md#orchestration-what-routing-changes}} and under the sponsor code's reading when it is above {{num:9.7%|README.md#orchestration-what-routing-changes}}; see <a href="../metric.html#default-answer">the metric page</a>.</p>
<h2 id="limits">Known limits</h2>
<p>Rhythmic polyphonic music can pass the cues; the preflight and the high-score flag remain the backstop. The module's docstring says "a tenth" of frames voiced where the constant is five percent.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_speech_gate.py</code>: silence, a chord, a tone and white noise are gated with the right reason; speech-like input passes; quiet real-level speech is not silence; contract and determinism; the policy pins gated files strictly below every scored file; ordering uses the weak signal and keys and puts failures at the bottom; only the gate flag gates; a data-gated test that real test clips pass. <code>tests/test_pipeline.py</code> pins that the map is applied once and that a decode failure sits below the block.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/reports/2026-09-26_gate-and-drift.md|The gate and drift report}}; {{src:README.md#orchestration-what-routing-changes|README, "The abstention path"}}; {{src:docs/consults/2026-09-26_fusion-strategy_RESPONSE.md#what-to-do-adopted|consult item 5}}.</p>
