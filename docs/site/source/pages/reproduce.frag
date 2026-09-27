<!-- {"title": "Reproduce it", "order": 60, "sources": ["README.md"], "lead": "The commands below are the README's, verbatim. Data, weights, model directories and submission TSVs are not in the repository; the layout they are expected in is stated, and the parity checks say how closely a live run reproduces the submitted file.", "description": "How to set up and run HEARSAY: the runner command that reproduces the submitted TSV, the Docker evidence image, and where every artifact lives."} -->
<h2 id="setup">Setup</h2>
<p>Requires uv and FFmpeg. {{src:README.md#setup|Source: the README's setup section.}}</p>
<pre><code class="language-bash">uv sync
cp .env.example .env
uv run pytest -q
uv run ruff check .</code></pre>
<p>Data, weights, model directories and submission TSVs are gitignored; only the experiment log is tracked. Expected layout: the NSA test set at <code>data/nsa/HackGTHearsayTesting/</code>, the template at <code>data/nsa/HearsayScoreKey4TeamX.tsv</code>, pretrained weights under <code>weights/</code>, trained bundles under <code>models/</code>.</p>
<h2 id="score">Score a directory</h2>
<pre><code class="language-bash">uv run python scripts/run_pipeline.py --in data/nsa/HackGTHearsayTesting --out outputs/runner/final --template data/nsa/HearsayScoreKey4TeamX.tsv --team CrossExam --fusion models/fusion_v2/constants.json</code></pre>
<p><code>--fusion models/fusion_v2/constants.json</code> selects the shipped rule. It writes <code>CrossExam_predictions.tsv</code>, one explanation JSON per file under <code>results/</code>, a resumable <code>results.jsonl</code>, <code>timings.csv</code> and <code>run_meta.json</code>. Rerun the same command to resume after a crash. <code>--flip</code> also writes the pre-flipped twin; <code>--rule</code>, <code>--policy</code> and <code>--flip</code> re-fuse the cached logits without reloading a model. Speed with M5 loaded: {{num:1.34 s|README.md#score-a-directory-on-the-mac}} per file on the full test set ({{num:38 minutes|README.md#score-a-directory-on-the-mac}} wall on a Mac shared with other jobs); peak memory {{num:4.0 GB|README.md#score-a-directory-on-the-mac}}. {{src:README.md#score-a-directory-on-the-mac|Source: the README.}}</p>
<h3 id="parity">Parity</h3>
<div class="table-wrap"><table>
<thead><tr><th>Check</th><th>Result</th></tr></thead>
<tbody>
<tr><td>Exported logits → shipped constants → policy, vs the submitted TSV, all {{num:1,671|README.md#score-a-directory-on-the-mac}} rows</td><td>max abs diff {{num:6.7e-16|README.md#score-a-directory-on-the-mac}}</td></tr>
<tr><td>Same, <code>--flip</code>, vs the flipped twin</td><td>max abs diff {{num:7.2e-16|README.md#score-a-directory-on-the-mac}}</td></tr>
<tr><td>Live from audio, all files, vs the submitted TSV</td><td>Spearman {{num:1.000000|README.md#score-a-directory-on-the-mac}}, max {{num:9.3e-4|README.md#score-a-directory-on-the-mac}}, mean {{num:1.8e-5|README.md#score-a-directory-on-the-mac}}; 0 verdict flips, 0 scorer errors</td></tr>
<tr><td>M5 logit, live CPU on WAV vs the A100 export on FLAC</td><td>max {{num:0.035|README.md#score-a-directory-on-the-mac}}, median {{num:7.0e-4|README.md#score-a-directory-on-the-mac}}</td></tr>
<tr><td>Handcrafted logit, live vs export</td><td>max {{num:4.4e-16|README.md#score-a-directory-on-the-mac}} (numpy trees, identical)</td></tr>
<tr><td>M1b logit, live CPU vs the MPS-extracted export</td><td>max {{num:5.5e-4|README.md#score-a-directory-on-the-mac}}</td></tr>
<tr><td>Spectra-AASIST logit, live CPU vs MPS export</td><td>max {{num:1.1e-4|README.md#score-a-directory-on-the-mac}}</td></tr>
<tr><td>Previous rule: exported logits → previous constants vs the 08:13 TSV, all rows</td><td>max abs diff {{num:1.1e-16|README.md#score-a-directory-on-the-mac}}</td></tr>
</tbody></table></div>
<h2 id="docker">Reproduce in Docker (offline, CPU; not a graded deliverable)</h2>
<p>NSA said on Saturday that the image is not required; the deliverable is the TSV and the README. The image is kept as evidence that the pipeline runs offline from a clean build. <strong>The recorded image runs the previous rule, not the shipped one:</strong> it does not contain the M5 checkpoint. {{src:README.md#reproduce-in-docker-offline-cpu-linuxamd64-not-a-graded-deliverable|Source: the README.}}</p>
<pre><code class="language-bash">bash docker/build.sh
docker run --network none -v &lt;test_dir&gt;:/data:ro -v &lt;out_dir&gt;:/out -v &lt;path&gt;/HearsayScoreKey4TeamX.tsv:/tmpl/key.tsv:ro -e HEARSAY_TEMPLATE=/tmpl/key.tsv -e HEARSAY_TEAM=CrossExam hearsay:20260926-0916</code></pre>
<p>The image contains every engineered detector, the M1b probe, Spectra-AASIST, the ECAPA speaker model, the handcrafted bundle and the fusion constants, with a sha manifest of the shipped files checked at start. It refuses to start unless it is offline, and nothing is downloaded at run time. Always pass <code>HEARSAY_TEAM</code>: the runner's default team name is not ours. Recorded image <code>hearsay:20260926-0916</code>: {{num:50|README.md#reproduce-in-docker-offline-cpu-linuxamd64-not-a-graded-deliverable}} test files scored inside the image against the previous rule's TSV, Spearman {{num:1.0|README.md#reproduce-in-docker-offline-cpu-linuxamd64-not-a-graded-deliverable}}, max abs diff {{num:2.02e-4|README.md#reproduce-in-docker-offline-cpu-linuxamd64-not-a-graded-deliverable}}; decoded audio identical to the Mac on 50 of 50 files; the smoke test passed, including the refusal to run online.</p>
<h2 id="artifacts">Where each artifact lives</h2>
<div class="table-wrap"><table>
<thead><tr><th>Artifact</th><th>Path</th></tr></thead>
<tbody>
<tr><td>Experiment log (every TSV with its holdout score)</td><td><code>submissions/log.csv</code></td></tr>
<tr><td>Shipped fusion constants</td><td><code>models/fusion_v2/constants.json</code>; previous rule <code>models/fusion_v1/constants.json</code></td></tr>
<tr><td>Per-detector score exports</td><td><code>outputs/detector_scores/&lt;name&gt;.csv</code> (regenerable; commands in each report)</td></tr>
<tr><td>Model bundles and their readouts</td><td><code>models/&lt;rung&gt;_&lt;stamp&gt;/meta.json</code></td></tr>
<tr><td>Submitted TSV</td><td><code>submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv</code> (sent as <code>CrossExam_predictions.tsv</code>)</td></tr>
<tr><td>Fold file</td><td><code>splits/nsa_folds.csv</code> (+ <code>nsa_folds_plus_asv19.csv</code> for M1b)</td></tr>
<tr><td>Experiment write-ups</td><td><a href="library/index.html#reports">docs/reports/</a></td></tr>
<tr><td>Outside reviews we asked for, and their answers</td><td><a href="library/index.html#consults">docs/consults/</a></td></tr>
<tr><td>Run specs and audits</td><td><a href="library/index.html#specs">docs/specs/</a></td></tr>
</tbody></table></div>
<h2 id="layout">Repository layout</h2>
<pre><code>src/hearsay/            application code; detectors/ holds the contract and every detector
tests/                  pytest suite (hermetic; data- and weight-dependent tests are marked)
scripts/                extraction, training, fusion, runner, cloud and ops scripts
docker/                 build, entrypoint, smoke, parity and asset-manifest scripts
docs/                   plan, architecture, code map, reports/, specs/, consults/, handoffs/, and this site
submissions/            log.csv (tracked) and submission TSVs (ignored)
data/ weights/ models/  local only
.claude/                Claude Code skills and review scripts shared by the team</code></pre>
