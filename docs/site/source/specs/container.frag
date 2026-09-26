<!-- {"title": "Container and metadata", "group": "Engineered detectors", "order": 42, "lead": "Reads the header facts of a file (format, codec, rate, bit depth, tags) and applies a few rules. On this test set it returns a constant score by design, because on our training data the container is the label; its facts route the compression detector instead.", "sources": ["src/hearsay/detectors/container.py", "docs/reports/2026-09-26_cpu-detectors.md"]} -->
<h2 id="plain">In plain words</h2>
<p>We measured the container before using it: LibriSpeech is FLAC, LJ Speech is 22 kHz WAV, two commercial generators are MP3, and PlayHT's MP3s carry the exact encoder tag every one of the {{num:1,671|README.md#what-did-not-work}} test files carries. A learned metadata model would ace our holdout for the wrong reason and then call the whole test set PlayHT. So the detector is rule-based, never learned, scores every test file at the neutral value, and hands its facts to the router. It never reads filenames or filesystem timestamps, which git, zip and Docker rewrite.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the file path (it runs ffprobe itself). Out: format, codec, sample rate, channels, bits, duration, tag count and selected tag values as features; routing facts <code>lossy</code>, <code>is_pcm_wav</code>, <code>ffmpeg_written</code>; a rule score; and a sentence such as <code>wav/pcm_s16le 16000 Hz mono 16-bit, 1 tag(s) [encoder=Lavf58.29.100]: no class evidence in the container; written by FFmpeg (libavformat), so the file was re-muxed or transcoded at least once</code>.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/detectors/container.py|detectors/container.py}}: <code>probe_container</code>, <code>classify_tags</code>, the detector; {{src:scripts/inventory_container.py|inventory_container.py}} for the corpus crosstab.</p>
<h2 id="params">Rules and where each is set</h2>
<ul>
<li>Tag values (never keys) are lower-cased and joined; a hit on a list of synthesis-tool names scores {{num:0.9|src/hearsay/detectors/container.py#L93}}, a hit on a list of synthesis words ("synthetic", "cloned", "ai voice", …) scores {{num:0.75|src/hearsay/detectors/container.py#L95}}, otherwise {{num:0.5|src/hearsay/detectors/container.py#L97}} with "no class evidence in the container" ({{src:src/hearsay/detectors/container.py#L88|container.py, lines 88–97}}). "tts" must match as a whole word.</li>
<li>A codec in the lossy list sets <code>lossy</code>; a WAV with a PCM codec sets <code>is_pcm_wav</code>; an encoder tag starting with "lavf" sets <code>ffmpeg_written</code> ({{src:src/hearsay/detectors/container.py#L117|container.py, lines 117–128}}).</li>
<li>A probe failure is an error result, not a crash.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built; routing only. The rubric's timestamps and MAC times are deliberately not read.</p>
<h2 id="numbers">Numbers</h2>
<p>A constant {{num:0.5|README.md#what-did-not-work}} on all {{num:21,671|README.md#the-detector-contract}} rows of the training and test exports. Every test file is 16 kHz mono 16-bit PCM WAV with one FFmpeg encoder tag.</p>
<h2 id="limits">Known limits</h2>
<p>By design it cannot score this test set. A tag naming a synthesis tool would score a file toward synthetic, but no test file carries one.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_container_detector.py</code>: a plain PCM WAV is neutral and carries the routing facts, a lossy codec is flagged but stays neutral, a tool tag scores {{num:0.9|src/hearsay/detectors/container.py#L93}}, a synthesis claim {{num:0.75|src/hearsay/detectors/container.py#L96}}, a table of tag rules (the FFmpeg tag, an empty tag, "Watts per channel", "Coqui TTS", "ai-generated", a key named tts), an unreadable file is an error, deterministic and filename-blind, no filesystem fields, registration.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/reports/2026-09-26_cpu-detectors.md|The CPU detectors report, "Trap 1"}}; {{src:README.md#what-did-not-work|README, "The container is the label"}}.</p>
