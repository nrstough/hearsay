<!-- {"title": "Splice and discontinuity detection", "group": "Engineered detectors", "order": 44, "lead": "Looks for the seams an edit leaves: a single-sample click, or a jump in the DC offset between short windows. On this data it finds vocoder artifacts more often than edits, and the sponsor said no test clip is partially synthetic, so it is evidence.", "sources": ["src/hearsay/detectors/splice.py", "docs/reports/2026-09-26_cpu-detectors.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Inserting or replacing words in a recording leaves a {{term:splice}}, a discontinuity where the cut is, unless the editor is careful. Two cheap signals catch careless cuts: a sample step far larger than its neighbours, and a shift in the waveform's mean between one window and the next. On the training data those signals fire on several generators' outputs and on some real LibriSpeech clips alike, so they describe artifacts, not the class.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the clip normalized to peak 1. Out: the counts and sizes of clicks and DC jumps, the time of the first seam, an informational background-floor range, and a sentence such as <code>editing seam at 1.59 s: 1 click(s) (largest step 17x the surrounding waveform); consistent with a cut or inserted segment (background-floor range 26 dB, informational)</code>, or <code>no clicks or DC-offset jumps: no sign of a cut or inserted segment</code>.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/splice.py|detectors/splice.py}}.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>A click is a sample step more than {{num:15.0|src/hearsay/detectors/splice.py#L30}} times the RMS of the steps in the surrounding 20 ms (the step itself excluded) and larger than {{num:0.15|src/hearsay/detectors/splice.py#L30}} of full scale; hits within 5 ms are merged ({{src:src/hearsay/detectors/splice.py#L30|splice.py, line 30}} and {{src:src/hearsay/detectors/splice.py#L52|lines 52–56}}).</li>
<li>A DC jump is a change in the mean between consecutive windows a tenth of a second long, larger than both {{num:0.02|src/hearsay/detectors/splice.py#L31}} and {{num:6.0|src/hearsay/detectors/splice.py#L31}} times the median window-to-window change ({{src:src/hearsay/detectors/splice.py#L57|splice.py, lines 57–61}}).</li>
<li>Scores: a seam {{num:0.6|src/hearsay/detectors/splice.py#L33}}, none {{num:0.5|src/hearsay/detectors/splice.py#L33}}; never fused.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built; evidence only.</p>
<h2 id="numbers">Numbers</h2>
<p>On training data the seams are single-sample clicks in {{num:27%|README.md#what-did-not-work}} of WaveGrad2 and DC jumps in 9–{{num:13%|README.md#what-did-not-work}} of several generators and {{num:9%|README.md#what-did-not-work}} of LibriSpeech. It flags {{num:58|README.md#what-did-not-work}} test files; in the ablation {{num:504|README.md#orchestration-what-routing-changes}} holdout rows, without moving a score.</p>
<h2 id="limits">Known limits</h2>
<p>A plosive can look like a click; the local-RMS rule and the absolute floor are what keep it from firing on speech, and a test pins that a burst is not a click. The rubric's phase breaks and background seams beyond DC are not measured.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_splice_detector.py</code>: a continuous signal is neutral, a click is a seam at the right time, a DC jump is a seam, a plosive burst is not a click, short and silent clips, the contract, the analysis keys, registration.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/reports/2026-09-26_cpu-detectors.md|The CPU detectors report, "Splice / discontinuity detector"}}; {{src:docs/reports/2026-09-26_worked-examples.md#8-hgt2305393wav-genuinely-uncertain-and-it-says-so-p--0471-z-mean-0995|worked example 8}}.</p>
