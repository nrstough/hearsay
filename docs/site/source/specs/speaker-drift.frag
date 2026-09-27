<!-- {"title": "Speaker drift", "group": "Engineered detectors", "order": 45, "lead": "Speaker embeddings on one-second windows, compared pairwise, to ask whether the voice stays the same within a clip. It works as a detector of edits and second voices, and it points the wrong way as a detector of fakes: cloned voices are the most self-consistent voices in the data.", "sources": ["src/hearsay/detectors/speaker_drift.py", "docs/reports/2026-09-26_gate-and-drift.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The rubric's speaker-embedding consistency technique. A pretrained speaker-recognition network ({{term:ECAPA}}) turns each second of speech into a vector describing the voice; if two windows of the same clip disagree, the voice changed. The surprise in the data: real people vary from second to second and clones do not, so the mean similarity separates classes <em>toward fake</em>. Fused, it would push the hard real recordings toward synthetic, which is the expensive error, so it stays evidence.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the clip, trimmed and capped. Out: the number of windows, the minimum, mean and standard deviation of the pairwise cosine similarities, the adjacent-window minimum, a drift flag, and a sentence: <code>voice drifts across the clip: minimum window-to-window speaker similarity −0.10 (mean 0.37) over 14 windows; consistent with a second voice or an unstable cloned identity</code>, or <code>one consistent voice: …</code>.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/speaker_drift.py|detectors/speaker_drift.py}}; the weights are the SpeechBrain ECAPA model loaded from a local directory with the hub path overridden, so it runs offline.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>Windows of {{num:1.0|src/hearsay/detectors/speaker_drift.py#L34}} s every {{num:0.5|src/hearsay/detectors/speaker_drift.py#L34}} s over at most {{num:8.0|src/hearsay/detectors/speaker_drift.py#L34}} s, the last window flush with the end; unit-norm embeddings; fewer than two windows means no drift ({{src:src/hearsay/detectors/speaker_drift.py#L75|speaker_drift.py, lines 75–101}}).</li>
<li>Drift when the minimum pairwise cosine is below {{num:0.05|src/hearsay/detectors/speaker_drift.py#L35}}; scores drift {{num:0.6|src/hearsay/detectors/speaker_drift.py#L36}}, none {{num:0.5|src/hearsay/detectors/speaker_drift.py#L36}}.</li>
<li>Calibrated on synthetic two-speaker splices: {{num:94%|docs/reports/2026-09-26_gate-and-drift.md}} of splices flagged against {{num:6.2%|docs/reports/2026-09-26_gate-and-drift.md}} of single voices.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built about 06:10 on Saturday; evidence only, by the finding below.</p>
<h2 id="numbers">Numbers</h2>
<p>Mean window-to-window similarity separates classes toward fake at AUC {{num:0.73|README.md#what-did-not-work}}. Drift flags {{num:16%|README.md#what-did-not-work}} of test files, mostly the hard real recordings; in the ablation {{num:180|README.md#orchestration-what-routing-changes}} holdout rows and {{num:269|README.md#orchestration-what-routing-changes}} test files, without moving a score.</p>
<h2 id="limits">Known limits</h2>
<p>After trimming, a mostly silent clip yields too few windows to say anything (worked example 7). The module's docstring quotes older flag rates than the report.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_speaker_drift.py</code>: windows cover the clip flush with the end, one voice is consistent and two voices drift (with a stub encoder), a single window is neutral, registration, offline loading with an empty cache and a real two-speaker splice (both weights-gated).</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/reports/2026-09-26_gate-and-drift.md|The gate and drift report}}; {{src:README.md#what-did-not-work|README, "Speaker drift points the wrong way"}}; {{src:docs/reports/2026-09-26_worked-examples.md#4-hgt1046947wav-the-deep-detectors-disagree-drift-flagged-p--0003-z-mean-0686|worked example 4}}.</p>
