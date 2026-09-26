<!-- {"title": "The metric module", "group": "Outputs", "order": 61, "lead": "The normalized detection cost used for every selection and every reported number, plus an exact re-implementation of the sponsor's shipped scoring code, so each result can be read both ways.", "sources": ["src/hearsay/metrics.py", "CLAUDE.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Everything on this site is measured with one function that mirrors the sponsor's rule: a false alarm costs {{num:4×|CLAUDE.md#conventions}} a miss and about {{num:70%|CLAUDE.md#conventions}} of files are real, scaled so that {{num:1.0|src/hearsay/metrics.py#L7}} is the best constant answer. The sponsor's actual script reads scores the other way round and moves the heavy weight to the missed fake, so a second function reproduces that script exactly. The long explanation is on <a href="../metric.html">the metric page</a>.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: labels (1 = synthetic) and scores. Out: normalized minDCF at the best threshold, the cost at a given threshold, EER, the Bayes threshold on a log-likelihood ratio, and the sponsor-code reading with or without a flip.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/metrics.py|src/hearsay/metrics.py}}.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li>Constants: {{src:src/hearsay/metrics.py#L29|C_FA = 4, C_MISS = 1, PI_SYNTH = 0.3}}. The normalizer is the best constant decision, <code>min(C_FA·(1−π), C_MISS·π)</code> ({{src:src/hearsay/metrics.py#L41|metrics.py, lines 41–42}}), which makes the normalized cost {{num:9.33|src/hearsay/metrics.py#L10}}·P_FA + P_miss.</li>
<li><code>min_cost</code> sweeps every threshold from the ROC curve and caps the result at the constant decision, so it is never above 1 ({{src:src/hearsay/metrics.py#L54|metrics.py, lines 54–59}}). Only the ranking of scores can change it.</li>
<li><code>bayes_llr_threshold</code> = ln(C_FA/C_MISS) − logit(π): the log-likelihood ratio above which calling a file synthetic pays ({{src:src/hearsay/metrics.py#L62|metrics.py, lines 62–64}}); <code>decision_logit</code> shifts a calibrated LLR so that a fixed {{num:50%|src/hearsay/metrics.py#L69}} cut makes that decision, which is a posterior above {{num:0.8|src/hearsay/metrics.py#L69}} at the four-to-one cost.</li>
<li><code>sponsor_min_dcf</code>: the sponsor's constants (bona fide prior {{num:0.5|docs/reports/2026-09-26_fusion-sweep-predeclared.md#readouts-for-every-candidate}}, the 4× on an accepted spoof) with a higher score read as bona fide, and a <code>flip</code> switch ({{src:src/hearsay/metrics.py#L78|metrics.py, lines 78–96}}). The sweeps call this reading the "miss-averse" cost.</li>
<li><code>report</code> returns EER, minDCF, the actual cost at the Bayes threshold, the same at an even prior, and both sponsor-code readings; <code>by_group</code> gives per-generator and per-source readouts.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built in the first hours; the sponsor-code twin was verified against the shipped script within 1e-9 for both polarities (a data-gated test).</p>
<h2 id="numbers">Numbers</h2>
<p>Every table on this site.</p>
<h2 id="limits">Known limits</h2>
<p>The docstring records an open question about whether the kickoff slide's formula weights the false-alarm term by the attack prior; the brief's wording is what the constants implement, and the sponsor's script is reproduced separately.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_audio_and_submission.py</code>: constant decisions normalize to one, a perfect scorer gives zero and the Bayes threshold at an even prior is ln 4, the sponsor-code twin matches the shipped script. <code>tests/test_docs_consistency.py</code> pins the constants stated in the documents to the module.</p>
<h2 id="sources">Sources</h2>
<p>{{src:CLAUDE.md#conventions|CLAUDE.md, "NSA metric and format"}}; {{src:docs/plan.md#scoring-and-deliverables|the plan, "Scoring and deliverables"}}; {{src:docs/architecture.md#1-what-shapes-the-design|Architecture §1}}.</p>
