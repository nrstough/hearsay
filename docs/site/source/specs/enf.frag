<!-- {"title": "ENF: mains hum", "group": "Engineered detectors", "order": 43, "lead": "Tracks the 50 or 60 Hz hum a mains-powered recording chain leaves behind, and whether it is stable across the clip. Rule-based and mild on purpose: on our training data hum means a home studio, which means one real corpus, so a learned version would learn the corpus.", "sources": ["src/hearsay/detectors/enf.py", "docs/reports/2026-09-26_cpu-detectors.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The electrical network frequency ({{term:ENF}}) is the rubric's acoustic-environment technique: a genuine recording made near mains power carries a faint hum at the grid frequency, and a spliced or fabricated background may carry a hum that jumps. Stable hum is in about a third of LJ Speech clips and almost no generated audio, so it is corpus evidence, not class evidence, and it stays out of the score.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the clip, silence-trimmed, capped and RMS-normalized. Out: the better of the 50 and 60 Hz candidates with its signal-to-noise ratio, the fraction of frames where it is present, its frequency, its spread and range; and one of three sentences: no hum, stable hum "consistent with a genuine electrical recording environment", or discontinuous hum "consistent with a fabricated or spliced background".</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/enf.py|detectors/enf.py}}.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>Windows of {{num:2.0|src/hearsay/detectors/enf.py#L29}} s every {{num:0.5|src/hearsay/detectors/enf.py#L29}} s over at most {{num:8.0|src/hearsay/detectors/enf.py#L29}} s, with a large FFT so the bins are about half a hertz; for each candidate, the peak within ±1 Hz per frame with quadratic interpolation, and its SNR against the neighbouring band ({{src:src/hearsay/detectors/enf.py#L29|enf.py, lines 29–70}}).</li>
<li>Present when the median SNR is at least {{num:12.0|src/hearsay/detectors/enf.py#L31}} dB and the hum is found in at least {{num:0.6|src/hearsay/detectors/enf.py#L31}} of frames; stable when, in addition, the frequency's standard deviation is at most {{num:0.25|src/hearsay/detectors/enf.py#L31}} Hz and its range at most {{num:0.5|src/hearsay/detectors/enf.py#L31}} Hz ({{src:src/hearsay/detectors/enf.py#L85|enf.py, lines 85–92}}).</li>
<li>Scores: stable hum {{num:0.4|src/hearsay/detectors/enf.py#L32}}, jumpy hum {{num:0.6|src/hearsay/detectors/enf.py#L32}}, none {{num:0.5|src/hearsay/detectors/enf.py#L32}}; mild by design, and never fused.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built; evidence only.</p>
<h2 id="numbers">Numbers</h2>
<p>Stable hum in {{num:36%|README.md#what-did-not-work}} of LJ Speech clips and {{num:21%|README.md#what-did-not-work}} of LibriSpeech; under {{num:0.5%|docs/architecture.md#11-shortcut-ledger}} of nine generators. Only {{num:24|README.md#what-did-not-work}} test files carry hum. In the ablation it flagged {{num:591|README.md#orchestration-what-routing-changes}} holdout rows without moving a score.</p>
<h2 id="limits">Known limits</h2>
<p>It cannot be validated as a class detector on this data. A test clip with a stable hum can be a fake: worked example 5 is one.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_enf_detector.py</code>: a stable 60 Hz hum, 50 Hz, none, a hum that jumps by 1 Hz mid-clip is discontinuous, digital silence, a short clip, contract and determinism, the analysis keys, registration.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/reports/2026-09-26_cpu-detectors.md|The CPU detectors report, "ENF (mains hum) detector"}}; {{src:docs/reports/2026-09-26_worked-examples.md#5-hgt3237868wav-a-fake-with-a-stable-mains-hum-p--0935|worked example 5}}.</p>
