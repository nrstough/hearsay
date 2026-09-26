<!-- {"title": "The stress sets", "group": "Validation", "order": 71, "lead": "Two public sets that nothing was selected on: In-the-Wild, web-sourced real and fake speech that no model ever trained on, for false alarms on unfamiliar voices; and MLAAD, synthetic speech from many generators, as training-only data for M5 and a miss-rate readout for the others.", "sources": ["README.md", "CLAUDE.md", "docs/reports/2026-09-26_channel-robustness.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Our training real speech is all clean read speech, so a model can look excellent on the holdout and still call ordinary phone or web audio synthetic. In-the-Wild is the check for that, read for every model and never trained on. It is also where the equal-weight fusion fell apart and where the pre-declared rules did their selecting. MLAAD answers a different question: how often a detector misses a generator it has never seen.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the sets' clips, prepared exactly like training rows (band match, trim, a seeded test-length crop). Out: minDCF under both cost readings and the false-alarm rate at each model's own inner-fold threshold; for MLAAD, the miss rate at that threshold.</p>
<h2 id="where">Where it lives</h2>
<p>Each detector's export carries its stress rows; the sweeps read them ({{src:scripts/fuse_sweep.py|fuse_sweep.py}}, {{src:scripts/fuse_sweep_m5.py|fuse_sweep_m5.py}}); the M3 probes score MLAAD ({{src:scripts/m3_probes.py|m3_probes.py}}). The M5 data module refuses In-the-Wild paths in any training manifest.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li><strong>In-the-Wild:</strong> {{num:3,000|README.md#validation}} clips, {{num:2,000|README.md#validation}} real from 54 speakers and {{num:1,000|README.md#validation}} fake; evaluation only ({{src:README.md#validation|README}}). Crops use a fixed stress seed, and the M5 addendum was re-run "crop-fair" after its first run scored whole clips.</li>
<li><strong>{{term:MLAAD}}:</strong> {{num:6,180|README.md#datasets}} English clips from {{num:143|README.md#datasets}} models pulled during the event; for M5 a training-only spoof source capped at {{num:12%|CLAUDE.md#ai-use-disclosure}} of the spoof side per batch and {{num:120|CLAUDE.md#ai-use-disclosure}} clips per model, with the generator families that validate on a fold excluded from that fold's model; for the leakage probes, {{num:572|README.md#what-did-not-work}} clips, four per model.</li>
</ul>
<h2 id="status">Status</h2>
<p>In use since M1; both cost readings reported from the fusion sweep on.</p>
<h2 id="numbers">Numbers</h2>
<p>Every XLS-R probe sits at {{num:0.34–0.41|README.md#what-did-not-work}} on In-the-Wild against {{num:0.07–0.16|README.md#what-did-not-work}} on the holdout, and the handcrafted model reaches {{num:1.00|README.md#what-did-not-work}} there. The shipped rule: {{num:0.228|README.md#results-at-a-glance}} under the brief's cost, {{num:0.239|README.md#numbers}} under the sponsor code's. On MLAAD, the miss rates: Spectra {{num:14.9%|README.md#what-did-not-work}}, M1b {{num:22.7%|README.md#what-did-not-work}}, M5 {{num:28.7%|README.md#what-did-not-work}} (it trained on MLAAD), handcrafted {{num:56%|README.md#what-did-not-work}}.</p>
<h2 id="limits">Known limits</h2>
<p>In-the-Wild stands in for "wild"; telephony and physical replay are not represented. Spectra-AASIST's authors evaluated on In-the-Wild, so its number there cannot be trusted. The λ̂ estimate says the test set is partly wild ({{num:0.51|README.md#what-did-not-work}}, with a wide interval), so neither validation set represents it alone.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_m5_manifest.py</code> refuses In-the-Wild paths in a training manifest; <code>tests/test_channel_robustness.py</code> pins the MLAAD miss-rate readout's threshold search and the perturbation probes; <code>tests/test_spectra.py</code> pins the stress-seed crops.</p>
<h2 id="sources">Sources</h2>
<p>{{src:README.md#validation|README, "Validation"}}; {{src:README.md#datasets|README, "Datasets"}}; {{src:docs/reports/2026-09-26_channel-robustness.md|the channel-robustness report}}.</p>
