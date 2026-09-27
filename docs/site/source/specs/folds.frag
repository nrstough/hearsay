<!-- {"title": "The fold file", "group": "Validation", "order": 70, "lead": "One tracked CSV assigns every training clip to an inner fold or the outer holdout, grouping fakes by generator and real speech by speaker, so that no model is ever tested on a voice or a synthesis system it trained on.", "sources": ["scripts/make_folds.py", "scripts/extend_folds.py", "docs/architecture.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Splitting clips at random would let a model see every generator and every speaker in training and then be "tested" on more of the same. The fold file prevents that by moving whole groups. Two generators and about a fifth of the real-speech groups form a holdout that nothing is ever fitted or selected on and each model reads once; the rest forms five inner folds that each hold out whole generators. Extra training corpora are added to the inner folds only, so holdout numbers stay comparable across every model.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the training manifest (path, label, generator, speaker, source). Out: <code>splits/nsa_folds.csv</code> with a <code>group</code> and a <code>fold</code> per row (0–4 or <code>holdout</code>); <code>splits/nsa_folds_plus_asv19.csv</code> with the ASVspoof 2019 rows added to the inner folds.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:scripts/make_folds.py|scripts/make_folds.py}} builds the base file; {{src:scripts/extend_folds.py|scripts/extend_folds.py}} adds extra rows; the M5 data module pins both files' hashes.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>Grouping: fakes by generator; LJ Speech by chapter (it is one speaker); other real speech by speaker ({{src:scripts/make_folds.py#L28|make_folds.py, lines 28–33}}).</li>
<li>Holdout generators <code>wavegrad2</code> and <code>playht</code>; a {{num:0.2|scripts/make_folds.py#L41}} share of real-speech groups; five inner folds by stratified group k-fold; at least 30 clips per class in every fold; seed 0 ({{src:scripts/make_folds.py#L39|make_folds.py, lines 39–44}}). Two assertions: every fold has both classes, and no group straddles folds.</li>
<li>Extra rows (M1b, M5) go to the inner folds only by the same grouped split, never to the holdout; path overlap with the base file is refused; the base file is never modified ({{src:scripts/extend_folds.py#L28|extend_folds.py, lines 28–48}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Built and frozen on Friday night; hash-pinned by the M5 tests since Saturday.</p>
<h2 id="numbers">Numbers</h2>
<ul>
<li>{{num:20,000|README.md#validation}} rows: 10 {{term:DiffSSD}} generators, {{term:LJ Speech}}, {{term:LibriSpeech}}. Inner folds {{num:16,142|README.md#numbers}} rows; the holdout {{num:3,858|README.md#validation}} rows (PlayHT, WaveGrad2 and 26 real-speech groups).</li>
<li>Inner folds hold out, in turn: grad_tts + unit_speech; diffgan_tts + openvoicev2; pro_diff + xtts_v2; your_tts; ElevenLabs ({{src:README.md#validation|README}}).</li>
<li>M1b's extension adds {{num:5,128|docs/architecture.md#62-xls-r-backbone-with-the-m1-probe-and-the-m5-head}} ASVspoof 2019 real clips (40 speakers) and {{num:2,520|docs/architecture.md#62-xls-r-backbone-with-the-m1-probe-and-the-m5-head}} spoof anchors to the inner folds.</li>
<li>The holdout's resolution floor is about ±{{num:0.07–0.10|README.md#validation}}; a gap under {{num:0.15|README.md#validation}} is treated as noise.</li>
</ul>
<h2 id="limits">Known limits</h2>
<p>Known leakage that grouping cannot fix is printed, not hidden: several DiffSSD generators clone the same voices, and LJ is one voice on both sides of every fold. grad_tts is invisible to the handcrafted features under this scheme because no other generator shows its phase cue.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_m5_manifest.py</code>: the fold files' hashes, the extended file agrees with the shared one on all {{num:20,000|README.md#validation}} core rows and never puts an extra row in the holdout, no holdout row in any training set. <code>tests/test_cloud_scripts.py</code>: the M5 code never writes the fold files.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/architecture.md#8-validation-and-selection|Architecture §8}}; {{src:README.md#validation|README, "Validation"}}; {{src:docs/reports/2026-09-26_m1b-asv19-bonafide.md|the M1b report}}.</p>
