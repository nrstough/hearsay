# Demo clips

Three short clips, each scored by the shipped pipeline (`scripts/run_pipeline.py --fusion models/fusion_v2/constants.json`, rule A3 w0.2 + E, Sat Sep 26, 2026, ~21:45). The explanation JSON the runner wrote for each is in `results/`; the frontend's demo presets read these files, so what the UI shows is exactly what the pipeline said. Copies of the audio live in `public/demo/` for playback in the app.

| File | What it is | Pipeline score (1.0 = synthetic) |
|---|---|---|
| `real_in_the_wild_28886.wav` | Real speech: In-the-Wild clip 28886 (a public-figure recording), never trained on; converted to 16 kHz mono PCM like the challenge test set | 0.0012 |
| `synthetic_apple_tts_samantha.wav` | Synthetic speech we generated with macOS text-to-speech (`say -v Samantha`), a generator absent from every training corpus; 16 kHz mono PCM | 0.9979 |
| `synthetic_in_the_wild_5518.wav` | A voice-cloned fake: In-the-Wild clip 5518, never trained on; 16 kHz mono PCM | 0.9979 |

Scores are from the runner's TSV for this directory; per-detector evidence and the routing log are in the matching `results/<file>.json`.

**Sources and licenses.** In-the-Wild: Müller et al., "Does Audio Deepfake Detection Generalize?" (2022), distributed on Hugging Face as `mueller91/In-The-Wild` under CC-BY-SA-4.0 (Apache-2.0 on deepfake-total.com); the two clips are redistributed here for demonstration under that license. The Apple TTS clip was synthesized by us; the sentence is our own.

Regenerate the results (from the repo root, models present):

```bash
uv run python scripts/run_pipeline.py --in demo --out outputs/runner/demo --team CrossExam --fusion models/fusion_v2/constants.json --no-preflight
```

then copy `outputs/runner/demo/results/*.json` into `demo/results/`.

## Run the app against the pipeline

Two processes, from the repo root. The API holds the models; the Next.js app forwards uploads to it and never computes a score itself.

```bash
HEARSAY_RESULTS=outputs/runner/v2_full uv run uvicorn hearsay.api:app --port 8000
```

```bash
npm ci && npm run dev
```

Then open http://localhost:3000. The demo presets load from `demo/results/`; "Load the submitted 1,671-file run" reads `submissions/CrossExam_predictions.tsv` and the per-file JSON under `HEARSAY_RESULTS`; "Export .TSV" serves that submitted file byte for byte. Set `HEARSAY_API_URL` if the API is not on port 8000.
