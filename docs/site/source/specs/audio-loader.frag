<!-- {"title": "Audio loader and probe", "group": "Ingest", "order": 10, "lead": "One decode path for every file: FFmpeg turns any container or codec into 16 kHz mono float32, once, and the header is read separately. Nothing downstream ever sees the original format.", "sources": ["src/hearsay/audio.py", "docs/architecture.md"]} -->
<h2 id="plain">In plain words</h2>
<p>Every audio file, whatever it is, becomes the same thing before any detector looks at it: a single channel of samples at {{num:16 kHz|README.md#how-it-decides}}, as floating-point numbers, with silence trimmed where a detector asks for it. The header (codec, rate, tags) is read on its own and never mixed into the audio. This is what closes the {{term:container}} {{term:shortcut}}: training real speech is FLAC or 22 kHz WAV and some training fakes are MP3, while every test file is 16 kHz PCM, so a model that could see the format would learn the format.</p>
<h2 id="io">What goes in and comes out</h2>
<ul>
<li><code>load_audio(path)</code> → a 1-D float32 array at 16 kHz mono.</li>
<li><code>probe_audio(path)</code> → a dict of header facts: format, codec, sample rate, channels, duration, bit rate; on failure an <code>error</code> entry, never an exception.</li>
<li><code>trim_silence(x)</code> → the array with quiet edges removed, relative to its loudest frame.</li>
<li><code>windows(x, win, hop)</code> → fixed-length windows, tiling clips shorter than one window.</li>
</ul>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/audio.py|src/hearsay/audio.py}}: <code>DecodeError</code>, <code>load_audio</code>, <code>probe_audio</code>, <code>windows</code>, <code>trim_silence</code>. The sample rate constant is in {{src:src/hearsay/__init__.py#L3|the package's __init__}}.</p>
<h2 id="params">Parameters and where each is set</h2>
<ul>
<li><strong>The FFmpeg command</strong> takes the first audio stream, downmixes to mono and resamples to 16 kHz, emitting little-endian float32 ({{src:src/hearsay/audio.py#L31|audio.py, lines 31–36}}). Empty output raises <code>DecodeError</code> with FFmpeg's last line; a partial decode is kept; non-finite samples are zeroed.</li>
<li><strong>No denoising and no loudness normalization</strong> anywhere on the path; a test pins it. Level is normalized per detector, later.</li>
<li><strong>Silence trim:</strong> threshold {{num:35|src/hearsay/audio.py#L93}} dB below the loudest 20 ms frame, and the result is never shorter than {{num:1.0|src/hearsay/audio.py#L94}} s; with fewer than two frames or nothing above the threshold the clip is returned unchanged ({{src:src/hearsay/audio.py#L93|audio.py, lines 93–110}}).</li>
<li><strong>Windows:</strong> the hop defaults to half a window; a clip shorter than one window is tiled; the last window is flush with the end ({{src:src/hearsay/audio.py#L76|audio.py, lines 76–90}}). Windows are the v1 path; the deep detectors now use segment mode (see <a href="segment-preparation.html">segment preparation</a>).</li>
<li><strong>Probe:</strong> <code>ffprobe</code> with <code>-show_format -show_streams -select_streams a:0</code>; returns the fields above and never raises ({{src:src/hearsay/audio.py#L47|audio.py, lines 47–73}}).</li>
</ul>
<h2 id="status">Status</h2>
<p>Built and tested; unchanged since the M0 rung. Invariant 1 of the architecture document ("one audio path").</p>
<h2 id="numbers">Numbers</h2>
<p>None of its own. Its effect shows up as the closed container shortcut on <a href="../what-worked.html#did-not-work">What did not work</a>.</p>
<h2 id="limits">Known limits</h2>
<p>FFmpeg's channel downmix averages channels; a stereo file with out-of-phase channels would lose signal. The loader keeps a partial decode of a truncated file rather than failing it, so a damaged file scores on what could be read (and the runner flags decode failures separately).</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_audio_and_submission.py</code>: any format to 16 kHz mono float32 (stereo 44.1k WAV, 8k mono, 22 kHz, CBR and VBR MP3, AAC in m4a and mp4, FLAC, Opus), determinism, empty and garbage files raise, a truncated WAV decodes what it can, NaN samples are sanitized, silence loads as zeros, no loudness normalization, windows cover the clip, silence trim is level-independent. <code>tests/test_detector_contract.py</code> pins that a clip is decoded once and that a decode failure is cached.</p>
<h2 id="sources">Sources</h2>
<p>{{src:docs/architecture.md#4-invariants-the-rules-the-code-enforces|Architecture, invariant 1}}; {{src:docs/architecture.md#11-shortcut-ledger|the shortcut ledger}}; {{src:docs/code-map.md|the code map}}.</p>
