<!-- {"title": "Inventories and the shortcut ledger", "group": "Validation", "order": 72, "lead": "Before any model trained, scripts measured how the training and test data differ in ways that have nothing to do with the voice. Every difference found is a shortcut a model would learn instead of the task, and each one is closed somewhere in the pipeline. The ledger is the list.", "sources": ["docs/architecture.md", "scripts/inventory.py", "scripts/inventory_container.py", "scripts/hash_audio.py"]} -->
<h2 id="plain">In plain words</h2>
<p>A detector with a perfect holdout score can be measuring the microphone, the file format or the loudness meter. The inventory scripts ask, for each cheap property of a file, how well it predicts the label on its own; anything that predicts too well is a trap. The ledger records the trap, the evidence and the fix, and every new detector or dataset is checked against it.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: a directory of audio (and labels, when known). Out: per-file probe and decode statistics, per-class format and duration tables, shortcut AUCs with the suspicious ones flagged, container facts, and hash indexes for leakage checks.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:scripts/inventory.py|inventory.py}} (probe and decode statistics, shortcut AUCs), {{src:scripts/inventory_container.py|inventory_container.py}} (header facts, the container crosstab), {{src:scripts/hash_audio.py|hash_audio.py}} (byte and PCM hashes, duplicate groups, cross-set matches). The ledger is section 11 of the architecture document.</p>
<h2 id="params">What each measures</h2>
<ul>
<li>Decode statistics per file: length, peak, leading and trailing silence, voiced fraction; shortcut AUCs for duration, silence, voiced fraction, peak, sample rate and bit rate, flagged beyond a fixed margin from chance ({{src:scripts/inventory.py#L108|inventory.py, lines 108–118}}).</li>
<li>Container facts through the container detector's own probe: format, codec, rate, channels, bits, tags; embedded fields only, never filesystem timestamps or names ({{src:scripts/inventory_container.py|inventory_container.py}}).</li>
<li>Hashes of the file bytes and of the decoded PCM, so exact and re-muxed copies are found across sets; lossy re-encodes are not caught ({{src:scripts/hash_audio.py|hash_audio.py}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Run at every data drop. The leakage check across LJ Speech, the sponsor's LJ subset and In-the-Wild found no exact copy of any test file.</p>
<h2 id="numbers">The ledger</h2>
<p>From {{src:docs/architecture.md#11-shortcut-ledger|the shortcut ledger}}:</p>
<div class="table-wrap"><table>
<thead><tr><th>Shortcut</th><th>Evidence</th><th>Closed by</th></tr></thead>
<tbody>
<tr><td>Peak level</td><td>test peaks near {{num:1.0|docs/architecture.md#11-shortcut-ledger}}, training real speech about {{num:0.5|docs/architecture.md#11-shortcut-ledger}}; AUC {{num:0.66|docs/architecture.md#11-shortcut-ledger}} alone</td><td>per-input normalization in both paths; level is never a feature</td></tr>
<tr><td>Leading silence</td><td>LibriSpeech {{num:0.37|docs/architecture.md#11-shortcut-ledger}} s vs test {{num:0.06|docs/architecture.md#11-shortcut-ledger}} s</td><td>the same silence trim on train and test</td></tr>
<tr><td>Duration and tiling</td><td>training 5–9 s vs test about {{num:3.4|docs/architecture.md#11-shortcut-ledger}} s; repeat-padding put a seam only in test data</td><td>segment mode: no tiling, training crops drawn from the test-duration distribution</td></tr>
<tr><td>Container and codec</td><td>real = FLAC or 22 kHz WAV, fakes include MP3, test = 16 kHz PCM with one FFmpeg tag (also on PlayHT)</td><td>one decode path; the container detector is never learned; laundering-equalized compression training</td></tr>
<tr><td>The {{num:7.2 kHz|docs/architecture.md#11-shortcut-ledger}} low-pass</td><td>{{num:1,602|docs/architecture.md#11-shortcut-ledger}} of {{num:1,671|docs/architecture.md#11-shortcut-ledger}} test files drop {{num:44|docs/architecture.md#11-shortcut-ledger}} dB between {{num:6.5|docs/architecture.md#11-shortcut-ledger}} and {{num:7.5|docs/architecture.md#11-shortcut-ledger}} kHz; no training corpus does</td><td>the band match in both paths</td></tr>
<tr><td>Mains hum</td><td>stable hum in {{num:36%|docs/architecture.md#11-shortcut-ledger}} of LJ, {{num:21%|docs/architecture.md#11-shortcut-ledger}} of LibriSpeech, under {{num:0.5%|docs/architecture.md#11-shortcut-ledger}} of nine generators</td><td>ENF is never learned</td></tr>
<tr><td>Crop edges</td><td>training crops start mid-word, test clips are whole utterances</td><td>no onset or offset features</td></tr>
<tr><td>LJ single speaker</td><td>one voice on both sides of every fold</td><td>folds group LJ by chapter; the stacker was fit on non-LJ real rows</td></tr>
<tr><td>Non-speech</td><td>silence and a chord score near {{num:1.0|docs/architecture.md#11-shortcut-ledger}} with the probe</td><td>the speech gate and the default-answer policy</td></tr>
</tbody></table></div>
<p>The band inventory of the test set: {{num:981|README.md#orchestration-what-routing-changes}} files read {{num:7,250|README.md#orchestration-what-routing-changes}} Hz, {{num:670|README.md#orchestration-what-routing-changes}} read {{num:7,500|README.md#orchestration-what-routing-changes}} Hz, 12 read full band and 8 sit at 5–7 kHz. The later λ̂ correction found the same class of shortcut again, in a noise-floor feature that read the stopband.</p>
<h2 id="limits">Known limits</h2>
<p>Inventories find the differences someone thought to measure. The test-domain offset in the handcrafted features (a darker envelope even after the band match) was found by a domain classifier, not by the inventory.</p>
<h2 id="tests">Tests that pin it</h2>
<p>The inventory scripts have no direct tests; the PCM hash is pinned through <code>tests/test_docker_image.py</code>, and the shortcut gate in the M5 trainer refuses a bundle whose duration, peak or silence AUC exceeds a limit (<code>tests/test_m5_manifest.py</code>).</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/architecture.md#11-shortcut-ledger|Architecture §11}}; {{src:docs/reports/2026-09-26_cpu-detectors.md|the CPU detectors report ("Trap 1", "Trap 2")}}; {{src:docs/STATUS.md#findings-so-far-the-what-worked--what-didnt-log-starts-here|STATUS, "Findings so far"}}.</p>
