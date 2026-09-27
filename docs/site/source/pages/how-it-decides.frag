<!-- {"title": "How it decides", "order": 10, "sources": ["README.md", "docs/reports/2026-09-26_worked-examples.md"], "lead": "Every file takes the same road: decoded once, handed to ten detectors, routed by fixed rules, fused by rank, checked by one model that may only lower a score, mapped to a probability, and written to the TSV with a report that says why. This page walks that road with one real test file.", "description": "The HEARSAY pipeline step by step: decode, route, ten detectors, rank fusion, Spectra suppression, Platt map, pinned block, on a real test file."} -->
<h2 id="six-steps">The six steps</h2>
<ol>
<li><strong>Decode once.</strong> Every file goes through one FFmpeg path to 16 kHz mono float32, and the header facts are read before decoding. Nothing downstream ever sees the original {{term:container}}, sample rate or filename.</li>
<li><strong>Run every detector.</strong> Each detector gets the same read-only clip and returns a score in [0, 1] (higher = more synthetic), a one-sentence reason, named features and a status. A detector that crashes becomes an error at {{num:0.5|README.md#how-it-decides}} and is treated as missing, never as {{term:evidence}}, and never costs a row.</li>
<li><strong>Route.</strong> The {{term:router}} is a set of fixed rules over measured file properties. It does not choose which detectors run (the whole test set is about 95 minutes of audio, so everything runs); it decides what each result is <em>allowed to do</em>: be fused, only lower the score, or stand as evidence. Every decision is written to the file's routing log.</li>
<li><strong>Fuse by rank, and let Spectra only pull scores down.</strong> The shipped rule is {{src:README.md#how-it-decides|0.6 × rank(M1b) + 0.2 × rank(handcrafted) + 0.2 × rank(M5)}}, each rank taken against that detector's own {{term:out-of-fold}} training scores. {{term:Spectra-AASIST}} can halve a high score when it is confident the voice is real. It can never raise one.</li>
<li><strong>Abstain at the bottom.</strong> A file with no speech to judge, or one that fails to decode, is pinned below every scored file. With a {{term:false alarm}} costing {{num:9.33|README.md#how-it-decides}} misses, "we don't know" belongs at the real end of the ranking.</li>
<li><strong>Write the TSV and one JSON per file.</strong> Header <code>filename&lt;TAB&gt;cm-score</code>, rows in the template's order, scores in [{{num:0.001|README.md#how-it-decides}}, 1] for every file we could judge, and an {{term:explanation JSON}} beside each.</li>
</ol>
{{diagram:pipeline|The shipped pipeline from audio file to TSV and explanation JSON.}}
<h2 id="router">The router: which fact decides what</h2>
<p>Orchestration here is not a learned model and not a language model. It is a short list of rules over facts the detectors measure, and the point of each rule is written next to it. The per-file routing log records every decision, so a reader can check any file after the fact.</p>
{{diagram:router|The router's rules: container facts, effective bandwidth, the speech gate, the Spectra step, evidence-only detectors, and imputation of a missing fused column.}}
<h3 id="router-measured">Router on versus off, measured</h3>
<p>The orchestration ablation re-fuses the exported detector scores with each rule switched on and off. "Router off" is the plain rank blend with no Spectra suppression, no gate and no {{term:pinned block}}; "fuse everything equally" is a four-way equal rank mean of M1b, the handcrafted model, M5 and Spectra.</p>
<div class="table-wrap"><table>
<thead><tr><th>Split</th><th>Router on (shipped)</th><th>No Spectra suppression</th><th>Router off</th><th>Fuse everything equally</th><th>Files Spectra suppression touched</th><th>Files the gate touched</th></tr></thead>
<tbody>
<tr><td>Holdout, {{num:3,858|README.md#orchestration-what-routing-changes}} rows</td><td>{{num:0.0065|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.0200|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.0200|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.0045|README.md#orchestration-what-routing-changes}}</td><td>83 (all real)</td><td>31</td></tr>
<tr><td>In-the-Wild, brief cost</td><td>{{num:0.229|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.274|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.274|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.214|README.md#orchestration-what-routing-changes}}</td><td>24 (all real)</td><td>9</td></tr>
<tr><td>In-the-Wild, sponsor-code cost</td><td>{{num:0.2415|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.2535|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.2505|README.md#orchestration-what-routing-changes}}</td><td>{{num:0.1955|README.md#orchestration-what-routing-changes}}</td><td></td><td></td></tr>
<tr><td>NSA test, share above {{num:0.5|README.md#orchestration-what-routing-changes}}</td><td>{{num:27.5%|README.md#orchestration-what-routing-changes}}</td><td>{{num:30.3%|README.md#orchestration-what-routing-changes}}</td><td>{{num:30.3%|README.md#orchestration-what-routing-changes}}</td><td>{{num:28.8%|README.md#orchestration-what-routing-changes}}</td><td>47</td><td>0</td></tr>
</tbody></table></div>
<ul>
<li><strong>Spectra suppression is the rule that changes decisions.</strong> It cuts the holdout cost to a third and takes {{num:0.045|README.md#orchestration-what-routing-changes}} off In-the-Wild under the brief's cost, and every file it touched where a label exists was real.</li>
<li><strong>The gate is a measured null on the test set.</strong> It touched holdout and In-the-Wild rows and 0 test files; its thresholds were set beyond the test set's extremes on purpose, so it only catches silence, tones, static and noise.</li>
<li><strong>The evidence-only detectors never move a score, by design.</strong></li>
<li><strong>Our rule does not win every readout.</strong> With M5 in the blend, fusing everything equally beats the shipped rule on every labeled readout. It does so by giving Spectra a full vote, which we ruled out before seeing these numbers because Spectra's training data is undisclosed. The choice rests on that argument, and the numbers are reported as measured.</li>
</ul>
<h2 id="worked-file">One file, end to end</h2>
<p>Test file <code>HGT1046947.wav</code> is one where the deep detectors disagree. What follows is the shipped pipeline's own output, run live from audio; its score matches the submitted TSV to seven millionths. {{src:README.md#how-it-decides|Source: the README's trace.}}</p>
<pre class="evidence-block"><code>fused    m1b_v3          XLS-R layer-7 probe: calibrated log-likelihood ratio -5.78 (real-like, P=0.00)
fused    m5_xlsr_ft      XLS-R fine-tuned head (M5, 12 layers, attentive pooling): logit +5.64 (synthetic-like, P=1.00)
fused    handcrafted     real-like (P=0.16): LFCC 16 frame-to-frame change 3.0 SD below real speech (toward synthetic); ...
fused    spectra_aasist  spoof-minus-bonafide margin +9.50 (synthetic-like, P=1.00)
evidence speaker_drift   voice drifts across the clip: minimum window-to-window speaker similarity -0.10 over 14 windows
evidence compression     compression-trace features synthetic-like (P=0.91): ...
routing  container       wav/pcm_s16le 16000 Hz mono, [encoder=Lavf58.29.100]: no class evidence in the container
gate     speech_gate     speech present: voiced 28% of frames, pitch spread 0.34, loudness std 18.5 dB

routing_log:
  container: PCM WAV, not lossy, FFmpeg-written; compression forensics run for evidence, not score
  compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
  enf: no mains hum; no environment evidence either way
  splice: no editing seams
  speaker_drift: voice drifts (min window similarity -0.10); evidence only, never fused
  speech_gate: is_speech=true (voiced 28% of frames); default-answer policy not applied
  fusion: e_on_a (A3_w0.2_E): 0.6 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) + 0.2 x rank(m5_xlsr_ft) = 0.319;
          M3 margin +9.50, no suppression (M3 never promotes)

probability_synthetic = 0.0348  -> real</code></pre>
{{diagram:fusion-arith|The fusion rule as arithmetic on HGT1046947.wav: three ranks blended, the Spectra step not applied, the Platt map, and what the rejected equal-weight rule would have said.}}
<p>Spectra-AASIST and M5 both say this voice is fake; the probe and the handcrafted model say real. Under the equal-weight rule we first tried, Spectra's vote carried the file to {{num:0.686|README.md#how-it-decides}}, above the midpoint. Under the shipped rule, M5's {{num:20%|README.md#how-it-decides}} share lifts it from {{num:0.0027|README.md#how-it-decides}} (the previous rule, without M5) to {{num:0.0348|README.md#how-it-decides}}, still firmly real. Spectra gets no say in that direction: we do not let a pretrained model with undisclosed training data push a file toward "synthetic", because being wrong that way is the expensive error. The drift and compression lines stay in the report as evidence and never touch the score.</p>
<h3 id="eight-files">Seven more files</h3>
<p>The worked-examples report walks eight real test files through the pipeline with their full routing logs and every evidence sentence: a confident real, a confident fake, the file where Spectra's suppression fired, a fake with a stable mains hum, a real file with editing seams, a {{num:21%|README.md#how-it-decides}}-voiced file near the gate, and a genuinely uncertain one. They were written on the previous rule and re-run under the shipped rule; no file changes side of {{num:0.5|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}. {{src:docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e|Source: the addendum.}}</p>
<div class="table-wrap"><table>
<thead><tr><th>File</th><th>Situation</th><th>p under the previous rule</th><th>p under the shipped rule</th><th>Spectra step</th></tr></thead>
<tbody>
<tr><td>HGT1013455.wav</td><td>confident real</td><td>{{num:0.0024|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.0028|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no</td></tr>
<tr><td>HGT1310023.wav</td><td>confident fake</td><td>{{num:0.9995|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.9994|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no</td></tr>
<tr><td>HGT3739624.wav</td><td>Spectra suppressed a likely false alarm</td><td>{{num:0.0312|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.0305|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td><strong>yes</strong></td></tr>
<tr><td>HGT1046947.wav</td><td>deep detectors disagree; drift flagged</td><td>{{num:0.0027|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.0348|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no (Spectra never promotes)</td></tr>
<tr><td>HGT3237868.wav</td><td>fake with a stable mains hum</td><td>{{num:0.9347|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.8973|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no</td></tr>
<tr><td>HGT1794158.wav</td><td>real with editing seams</td><td>{{num:0.0049|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.0097|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no</td></tr>
<tr><td>HGT2080120.wav</td><td>{{num:21%|README.md#how-it-decides}} voiced, near the gate</td><td>{{num:0.9640|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.9384|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no</td></tr>
<tr><td>HGT2305393.wav</td><td>genuinely uncertain; click flagged</td><td>{{num:0.4714|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>{{num:0.4656|docs/reports/2026-09-26_worked-examples.md#addendum-sat-1230-the-same-eight-files-under-fusion_v2-a3_w02_e}}</td><td>no (Spectra never promotes)</td></tr>
</tbody></table></div>
<p>Two rules can change a score, and each is visible in the log when it does: Spectra suppression fired on one of the eight files and on no other; the gate fired on none. Six detectors contributed evidence without touching the score, and the log says so each time. The rejected equal-weight rule would have moved four of the eight files across {{num:0.5|docs/reports/2026-09-26_worked-examples.md#what-the-eight-files-show-in-one-place}}, all by letting a detector with undisclosed training data vote in full.</p>
