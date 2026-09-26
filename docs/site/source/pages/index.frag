<!-- {"title": "Home", "heading": "HEARSAY: is this voice real?", "order": 0, "sources": ["README.md"], "lead": "HEARSAY is a software-only audio authentication system built in 36 hours at HackGT 13 for the NSA HEARSAY challenge. Give it an audio file in any format and it returns the probability that the voice is synthetic ({{num:0.0|README.md#hearsay}} real, {{num:1.0|README.md#hearsay}} synthetic) with a report that says which detectors ran, what each one found, and how the number was put together. Ten forensic detectors run on every file; a rule-based router decides what each result may do; a fusion rule chosen by a selection rule written down before any result combines the three that vote.", "description": "HEARSAY, the Cross Exam entry to the NSA HEARSAY audio authentication challenge at HackGT 13: what it is, how well it does, and how it decides."} -->
<h2 id="verdict">The numbers</h2>
<p>All figures are normalized {{term:minDCF}} (lower is better). The rule was chosen by a {{term:pre-declared}} selection rule, applied once; the {{term:out-of-fold}} figure is the training-side readout, the {{term:holdout}} and {{term:In-the-Wild}} figures were each read once after the choice.</p>
<div class="verdict-row">
<div class="stat"><span class="label">Outer holdout</span><span class="value">{{num:0.0065|README.md#results-at-a-glance}}</span><span class="note">two unseen generators and 26 unseen real-speech groups</span></div>
<div class="stat"><span class="label">In-the-Wild, brief cost</span><span class="value">{{num:0.228|README.md#results-at-a-glance}}</span><span class="note">web-sourced speech we never trained on; {{num:0.239|README.md#numbers}} under the {{term:sponsor code|sponsor code's}} cost</span></div>
<div class="stat"><span class="label">Inner out-of-fold</span><span class="value">{{num:0.135|README.md#results-at-a-glance}}</span><span class="note">pooled over the five inner folds</span></div>
<div class="stat pending"><span class="label">NSA test set, draft review</span><span class="value">pending</span><span class="note">the returned minDCF, once NSA sends it</span></div>
</div>
{{cost-panel}}
<h2 id="how-it-decides">How it decides, in one picture</h2>
<p>One decode path, ten detectors behind one contract, a router that tells each result what it may do, and a fusion rule that reads only ranks: {{src:README.md#how-it-decides|0.6 × rank(M1b) + 0.2 × rank(handcrafted) + 0.2 × rank(M5)}}, then {{term:Spectra-AASIST}} allowed only to pull a file toward real, then a {{term:Platt map}}, with undecidable files in the {{term:pinned block}} at the real end. The full walk-through, with a real test file, is on <a href="how-it-decides.html">How it decides</a>.</p>
{{diagram:pipeline|The shipped pipeline: decode once, route, ten detectors, three fused by rank, Spectra as a one-way vote, Platt map, pinned block, TSV and explanation JSON.}}
<h2 id="results">Results at a glance</h2>
<div class="table-wrap"><table>
<thead><tr><th></th><th>Inner OOF</th><th>Holdout</th><th>In-the-Wild, brief cost</th><th>In-the-Wild, sponsor-code cost</th></tr></thead>
<tbody>
<tr><td><strong>Shipped rule</strong> ({{src:README.md#numbers|A3 w 0.2 + E}}: M1b, handcrafted and M5 by rank, Spectra suppression)</td><td>{{num:0.135|README.md#results-at-a-glance}}</td><td>{{num:0.0065|README.md#results-at-a-glance}}</td><td>{{num:0.228|README.md#results-at-a-glance}}</td><td>{{num:0.239|README.md#numbers}}</td></tr>
<tr><td>Previous rule (E on α 0.2, no M5), the fallback</td><td>{{num:0.140|README.md#results-at-a-glance}}</td><td>{{num:0.014|README.md#results-at-a-glance}}</td><td>{{num:0.260|README.md#results-at-a-glance}}</td><td>{{num:0.267|README.md#numbers}}</td></tr>
<tr><td>{{term:M1b}} alone, the best single detector we trained</td><td>{{num:0.301|README.md#results-at-a-glance}}</td><td>{{num:0.072|README.md#results-at-a-glance}}</td><td>{{num:0.343|README.md#results-at-a-glance}}</td><td>{{num:0.296|README.md#numbers}}</td></tr>
<tr><td>{{term:Spectra-AASIST}} off the shelf, a {{term:suppression|suppressor}} only</td><td colspan="2">not fused: its training data is undisclosed, so no row of ours is provably out-of-sample for it</td><td colspan="2">{{num:0.065|README.md#numbers}}, possibly in-sample for the same reason</td></tr>
</tbody></table></div>
<p>The test set has {{num:1,671|README.md#numbers}} files. The shipped rule scores {{num:27.4%|README.md#numbers}} of them above {{num:0.5|README.md#numbers}} against the roughly {{num:30%|README.md#numbers}} synthetic the brief states (the orchestration ablation, a different readout, reports {{num:27.5%|README.md#orchestration-what-routing-changes}}); the {{term:speech gate}} abstains on {{num:0 of 1,671|README.md#the-eight-forensic-techniques}}. Turning the {{term:router}} off moves the holdout from {{num:0.0065|README.md#orchestration-what-routing-changes}} to {{num:0.0200|README.md#orchestration-what-routing-changes}} and In-the-Wild from {{num:0.229|README.md#orchestration-what-routing-changes}} to {{num:0.274|README.md#orchestration-what-routing-changes}}.</p>
<h2 id="read-next">Where to go next</h2>
<ul class="cards">
<li class="card"><h3><a href="how-it-decides.html">How it decides</a></h3><p>The pipeline, the router and the fusion arithmetic, on a real test file with its evidence sentences.</p></li>
<li class="card"><h3><a href="metric.html">The metric and the default answer</a></h3><p>What minDCF is, why a false alarm costs so much, why only ranking matters, and where the undecidable files go.</p></li>
<li class="card"><h3><a href="what-worked.html">What worked and what did not</a></h3><p>The honest log: every shortcut found, every idea rejected, with its number and its mechanism.</p></li>
<li class="card"><h3><a href="detectors.html">The ten detectors</a></h3><p>One card each, mapped to the sponsor's rubric, with its role, its numbers and one real evidence sentence.</p></li>
<li class="card"><h3><a href="validation.html">Validation</a></h3><p>Folds by generator and speaker, the pre-declared sweeps, and the decoding table for the draft review.</p></li>
<li class="card"><h3><a href="specs/index.html">Systems and subsystems</a></h3><p>One spec page per part of the system, every parameter cited to the line that sets it.</p></li>
<li class="card"><h3><a href="library/index.html">The library</a></h3><p>Every document in the repository, rendered: reports, run specs, audits, handoffs, consult records.</p></li>
<li class="card"><h3><a href="reproduce.html">Reproduce it</a></h3><p>The commands, verbatim, and where every artifact lives.</p></li>
</ul>
