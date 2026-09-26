<!-- {"title": "Credits and AI use", "order": 70, "sources": ["README.md", "CLAUDE.md"], "lead": "We used AI coding assistants heavily, and we say so plainly. The team decided what to build and why; Claude Code built it under our direction; Codex reviewed plans and audited finished rungs; two strategy questions went to an outside multi-model review. Nothing on the scoring path calls a hosted model.", "description": "Who built what in HEARSAY: the team's decisions, what the AI assistants wrote, the pretrained models and datasets with their licences, and the frameworks used."} -->
<h2 id="ai-use">AI use, in short</h2>
<ul>
<li><strong>Claude Code (Anthropic)</strong> wrote most of the code and documents under our direction, in parallel sessions per rung: loader, metrics, fold file, the M1 probe, the engineered detectors, M3 scoring, the M5 cloud pipeline, the fusion sweep, the runner, the API, the Docker image, the documentation, and this site and its generator. We chose the approach, set the gates and selection rules, reviewed results and made every ship decision.</li>
<li><strong>OpenAI Codex (Codex CLI)</strong> reviewed plans and audited finished rungs through our plan-review workflow.</li>
<li><strong>Outside consults.</strong> Two strategy questions, one on M5's extra data and one on fusion, went to an outside multi-model review (VeriLM; the fusion memo was answered by Claude and Gemini with a synthesis). The prompts and answers are saved in the repository, and we record which advice we adopted and which we ignored: see the <a href="library/index.html#consults">consult records</a>.</li>
<li><strong>No AI on the scoring path.</strong> No LLM or hosted API runs at inference; the runner loads every model from local files, and the Docker image runs with networking off.</li>
</ul>
<p>The full disclosure, including which components each tool touched and what we built versus what the AI wrote, component by component, is kept current in {{src:CLAUDE.md#ai-use-disclosure|CLAUDE.md, "AI use disclosure"}}; the README's summary is at {{src:README.md#ai-use-and-credits|"AI use and credits"}}.</p>
<h2 id="models">Pretrained models</h2>
<div class="table-wrap"><table>
<thead><tr><th>Model</th><th>Source</th><th>Licence</th><th>How we used it</th></tr></thead>
<tbody>
<tr><td>XLS-R 300M, <code>facebook/wav2vec2-xls-r-300m</code></td><td>Hugging Face (Meta)</td><td>Apache-2.0</td><td>Frozen backbone for M1/M1b (layer 7) and M5 (first 12 layers); both are fused</td></tr>
<tr><td>Spectra-AASIST, <code>lab260/Spectra-AASIST</code></td><td>Hugging Face</td><td>unclear: repo header says Apache-2.0, model card says MIT</td><td>M3, scored off the shelf, false-alarm suppression only</td></tr>
<tr><td>ECAPA-TDNN, <code>speechbrain/spkrec-ecapa-voxceleb</code></td><td>Hugging Face (SpeechBrain)</td><td>Apache-2.0</td><td>Speaker-drift evidence</td></tr>
<tr><td>WavLM Large / Base, <code>microsoft/wavlm-large</code>, <code>microsoft/wavlm-base</code></td><td>Hugging Face (Microsoft)</td><td>no licence on the card; released through Microsoft's unilm repo (MIT)</td><td>Downloaded as bake-off challengers; never run</td></tr>
</tbody></table></div>
<h2 id="datasets">Datasets</h2>
<div class="table-wrap"><table>
<thead><tr><th>Dataset</th><th>Source</th><th>Licence</th><th>How we used it</th></tr></thead>
<tbody>
<tr><td>NSA HEARSAY data: test set, resampled LJ subset, DiffSSD, ASVspoof5 scoring code</td><td>NSA, provided during the event</td><td>event terms</td><td>Test set; training sample and fold file; metric reference</td></tr>
<tr><td>LJ Speech {{num:1.1|README.md#datasets}}</td><td>keithito.com</td><td>public domain</td><td>Real speech (one speaker); training</td></tr>
<tr><td>LibriSpeech dev-clean / test-clean</td><td>openslr.org/12</td><td>CC BY {{num:4.0|README.md#datasets}}</td><td>Real speech (multi-speaker); training</td></tr>
<tr><td>ASVspoof 2019 LA</td><td>University of Edinburgh DataShare</td><td>ODC Attribution</td><td>Public shakedown of M1; 40 VCTK real speakers and an A01–A06 anchor as training-only rows for M1b and M5</td></tr>
<tr><td>In-the-Wild (Müller et al. 2022), <code>mueller91/In-The-Wild</code></td><td>Hugging Face</td><td>CC-BY-SA-4.0 on Hugging Face; Apache-2.0 on deepfake-total.com</td><td>Stress test only; never trained on</td></tr>
<tr><td>MLAAD, <code>mueller91/MLAAD</code></td><td>Hugging Face (gated)</td><td>non-commercial notice</td><td>{{num:6,180|README.md#datasets}} fake clips from {{num:143|README.md#datasets}} TTS models, training-only for M5</td></tr>
</tbody></table></div>
<p>{{src:README.md#pretrained-models|Source: the README's models table}} and {{src:README.md#datasets|datasets table}}.</p>
<h2 id="frameworks">Frameworks and tools</h2>
<p>PyTorch and torchaudio (BSD), Hugging Face transformers and huggingface_hub (Apache-2.0), SpeechBrain (Apache-2.0), librosa (ISC), soundfile (BSD-3-Clause), NumPy, SciPy, scikit-learn and pandas (BSD-3-Clause), LightGBM (MIT, training only), pydub (MIT), ffmpeg-python (Apache-2.0), FFmpeg (LGPL/GPL), FastAPI (MIT) and uvicorn (BSD-3-Clause) for the local API, pytest and ruff for tests and lint, uv for environments, Docker with Colima for the image, vast.ai A100s and Cloudflare R2 for the M5 runs. This site: a standard-library Python generator, Mermaid (MIT) rendered once to SVG, and no runtime dependency. {{src:README.md#frameworks-and-tools|Source: the README.}}</p>
<h2 id="team">The team</h2>
<p>Team Cross Exam at HackGT 13. Nathan (@nrstough) owned the deep detectors, the fold file, fusion, the orchestrator and the TSV; teammates owned the engineered detectors and the Docker image; Hrushi built the demo frontend against the pipeline's JSON contract. The component-by-component table of who decided what and what the AI wrote is in {{src:CLAUDE.md#ai-use-disclosure|CLAUDE.md}}.</p>
