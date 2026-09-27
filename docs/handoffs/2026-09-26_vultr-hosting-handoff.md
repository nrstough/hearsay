# Handoff — host the demo app on a Vultr VM for Hrushi's video (Sat Sep 26, 2026, ~22:20)

**Purpose of this chat:** put the wired frontend and the pipeline API on one public Ubuntu VM on Vultr so Hrushi can record the demo video against real scores, without changing the model, the TSV or the UI. Target: a URL that serves the app on port 80, with uploads scored by the shipped rule, the three demo presets, the submitted-run batch view and the TSV export. Tear the VM down after the video.

## Context

- The frontend (Next.js at the repo root) is wired to the real pipeline as of `7f870ca`: its route handlers call the FastAPI service `hearsay.api` server-side (`HEARSAY_API_URL`, default `http://127.0.0.1:8000`), read `submissions/CrossExam_predictions.tsv` and the runner's per-file JSON (`HEARSAY_RESULTS`), and serve demo presets from `demo/results/`. No CORS is needed: the browser only talks to the Next app. Run instructions: `demo/README.md`.
- The model is frozen and verified: a full 1,671-file re-run from audio matched the submitted file (Spearman 1.0, max 9.3e-4, 0 flips; `submissions/log.csv`, 22:52 row). Nothing in this task may change `models/`, `weights/`, `submissions/` or the scoring code.
- The repo is public: https://github.com/nrstough/hearsay (main at `4258d33` or later; includes Hrushi's guided tour `7d89a75` and the docs site).
- The Docker image (`hearsay:20260926-0916`) runs the batch runner with the fallback rule and has no M5 checkpoint or API server; it is not the hosting path. Host from a checkout instead.
- **Gitignored assets the API needs, copied from the Mac** (about 3.3 GB; the WavLM weights are not needed):

| Path | Size | Why |
|---|---|---|
| `weights/wav2vec2-xls-r-300m/` | 1.2 GB | XLS-R backbone for the probe and the M5 head |
| `weights/Spectra-AASIST/` | 1.3 GB | Spectra-AASIST (suppression step) |
| `weights/spkrec-ecapa-voxceleb/` | 85 MB | speaker-drift evidence |
| `models/m1_wav2vec2-xls-r-300m_L7_20260926-0521/` | 36 KB | the shipped probe |
| `models/hc_lgbm_20260926-055451/` + symlink `models/hc_selected -> hc_lgbm_20260926-055451` | 4.4 MB | handcrafted bundle |
| `models/m5_xlsr_ft_20260926-0741/` | 631 MB | M5 checkpoint (hash-checked by the runner) |
| `models/fusion_v2/`, `models/fusion_v1/` | 1.6 MB | the shipped rule and the fallback |
| `outputs/runner/v2_full/results/` | 33 MB | per-file JSON for the batch view (optional; without it the batch view shows scores only) |
| `submissions/CrossExam_predictions.tsv` | 56 KB | the submitted file, for the batch view and the export |

- Resource needs measured on the Mac: API peak memory about 4 GB with M5 loaded; 1.3 s per file with 6 threads on an M3 Pro, so expect 3–6 s per upload on a 4-vCPU cloud VM after a 30–60 s model load on the first request. Next production build about 300 MB of `node_modules` plus `.next`.
- **Cost and approval:** a Vultr Cloud Compute instance with 4 vCPU / 8 GB RAM / 80 GB or more disk (about $0.07/h, hourly billed). CPU only; no GPU. Nathan approved hosting for the video; destroy the instance afterwards and record the spend in `docs/reports/cloud-expense-ledger.md`.
- **Security:** the API has no authentication. Bind it to 127.0.0.1 and expose only the Next app through nginx on port 80. Anyone with the URL can upload audio for scoring during the demo window; that is acceptable for a short-lived demo box, not for a permanent host. No secrets go in the repo; if an HF token is needed for anything, it lives in `/etc/hearsay.env` on the box only.

## Working branch / worktree

Local: `main` in `~/Projects/hearsay` on the Mac, in sync with origin at `4258d33` plus `bd94e39` (log row). On the VM: a fresh clone of `main`. This task adds only `docs/reports/2026-09-26_vultr-hosting.md` (what was deployed, the URL, the spend) and, if useful, `deploy/vultr/` with the nginx and systemd units used. Stage only those; never `git add -A`; the oversight chat pushes.

## Environment / setup

On the Mac (source of the assets), from the repo root:

```bash
cd ~/Projects/hearsay && ls weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb models/m5_xlsr_ft_20260926-0741/model models/fusion_v2/constants.json >/dev/null && echo assets present
```

On the VM (Ubuntu 24.04), after `ssh root@<ip>`:

```bash
apt-get update && apt-get install -y ffmpeg nginx git curl build-essential && curl -LsSf https://astral.sh/uv/install.sh | sh && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
```

## What to do next

1. **Create the VM** in the Vultr console or CLI: Cloud Compute, Ubuntu 24.04, 4 vCPU / 8 GB / 80+ GB (a 2 vCPU / 4 GB box will swap when M5 loads), region near Hrushi, SSH key added. Note the IP. Ask Nathan for the account if this chat does not have it; do not create a second instance.
2. **Clone and install** as a non-root user `hearsay`: `git clone https://github.com/nrstough/hearsay && cd hearsay && uv sync && cp .env.example .env && npm ci && npm run build`. `uv sync` pulls CPU torch (about 2 GB of wheels); `npm run build` was verified on the Mac (see the build log note below).
3. **Copy the assets from the Mac** (one rsync per row of the table, or one command with `--relative`):
   ```bash
   cd ~/Projects/hearsay && rsync -avz --relative --copy-links weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb models/m1_wav2vec2-xls-r-300m_L7_20260926-0521 models/hc_lgbm_20260926-055451 models/hc_selected models/m5_xlsr_ft_20260926-0741 models/fusion_v2 models/fusion_v1 outputs/runner/v2_full/results submissions/CrossExam_predictions.tsv hearsay@<ip>:~/hearsay/
   ```
   About 3.3 GB; 10–30 minutes depending on the uplink. `--copy-links` turns the `hc_selected` symlink into a directory copy, which the runner accepts. Check `models/hc_selected/meta.json` exists on the box afterwards.
4. **Smoke the API on the box** before wiring nginx: `HEARSAY_RESULTS=outputs/runner/v2_full uv run uvicorn hearsay.api:app --host 127.0.0.1 --port 8000` in one shell, then `curl -s -F file=@demo/synthetic_apple_tts_samantha.wav localhost:8000/analyze | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['probability_synthetic'], d['verdict'], d['fusion']['detail']['final'])"` → `0.9979 synthetic A3_w0.2_E` (the Mac gave 0.9979; a CPU-vs-CPU difference under 1e-3 is expected, a different verdict is not). `curl localhost:8000/health` must show `fusion: fusion_v2/constants.json` and four scorers.
5. **Run both as systemd services** (units under `deploy/vultr/`, commit them): `hearsay-api.service` (`ExecStart=/home/hearsay/.local/bin/uv run uvicorn hearsay.api:app --host 127.0.0.1 --port 8000`, `Environment=HEARSAY_RESULTS=outputs/runner/v2_full OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1`, `WorkingDirectory=/home/hearsay/hearsay`, `Restart=always`) and `hearsay-web.service` (`ExecStart=/usr/bin/npm start -- --port 3000`, `Environment=HEARSAY_API_URL=http://127.0.0.1:8000 NODE_ENV=production`, same working directory). Warm the API once with a demo upload so the first video take does not wait 60 s.
6. **nginx** on port 80 proxying to 127.0.0.1:3000 with `client_max_body_size 100m` and `proxy_read_timeout 300s` (a first upload can take a minute while models load). No route to :8000 from outside. Optional: `certbot` if a domain is pointed at the box; the demo works over plain http.
7. **Verify from outside**, exactly what the video will show: open `http://<ip>/`, confirm the three presets load with 0.001 / 0.998 / 0.998, click "Load the submitted 1,671-file run" (457 above 0.5, minDCF 0.0733 in the benchmark card), upload `demo/real_in_the_wild_28886.wav` from a laptop and get 0.001 real, export the TSV and check `shasum -a 256` starts `fb783076`. Take screenshots for the report.
8. **Hand the URL to Hrushi** with three notes: pull main before any UI change; the tour copy still carries unmeasured numbers (minDCF 0.1983, EER 4.82%, 48 kHz, 16 kHz roll-off, 60.014 Hz ENF, "Intercept #0042", a 2-hour timeline, "8 sensors", an "AI specialist") that must be fixed in `src/components/GuidedTour.tsx` before filming, then rebuilt on the box (`git pull && npm run build && systemctl restart hearsay-web`); the API scores in 3–6 s per upload, so leave the spinner in the take.
9. **Write** `docs/reports/2026-09-26_vultr-hosting.md`: instance spec and region, URL, the smoke results, the spend, the teardown time. Add the spend line to `docs/reports/cloud-expense-ledger.md`. **Destroy the instance** when Hrushi confirms the footage is in, and record that.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test on the box: `uv run pytest -q tests/test_api.py tests/test_pipeline.py` (hermetic parts pass without data; data-marked tests skip); `npm run build` exit 0.
- Parity on the box (the one number that matters): the demo upload in step 4 must return `0.9979 synthetic A3_w0.2_E`, and the real demo clip `0.0012 real`. If either verdict differs, stop: an asset is missing or wrong (check `models/hc_selected`, the M5 hashes in `models/fusion_v2/constants.json` against `models/m5_xlsr_ft_20260926-0741/model/hashes.json`, and `HF_HUB_OFFLINE`).
- At-risk: nothing new is created on the Mac. On the box everything is a copy; destroying the VM loses nothing. The only records are the report and the ledger line, both tracked.
- Do not copy `data/`, `outputs/` beyond the results dump, or `weights/wavlm-*` (unused, 1.6 GB).
- In flight, other lanes: README polish until Sunday ~05:00; the Sunday 05:00 preflight and direction check on the final file (not this lane's); Hrushi's tour-copy fix.

## Analytical notes

- The app never computes a score: `src/lib/hearsay.ts` adapts the API's JSON; `src/app/api/forensic/*` are thin. If the API is down, uploads show an error and the presets still work.
- `HEARSAY_RESULTS` is optional; without it the batch view lists TSV scores with "per-file explanation not on this machine".
- Ports: 80 public (nginx) → 3000 (Next, localhost) → 8000 (API, localhost). Keep 3000 and 8000 closed in the Vultr firewall.
- Memory: API about 4 GB resident with M5; Node about 0.5 GB; leave 2 GB headroom → 8 GB box.
- Threads: set `OMP_NUM_THREADS` to the vCPU count; the runner caps at 6.
- Score direction 1.0 = synthetic everywhere; the UI labels "Authentic" below 0.5.

## Pointers

- `demo/README.md` (run commands, presets, sources), `docs/handoffs/2026-09-26_frontend-contract.md` (contract v0 and the rewire note), `src/hearsay/api.py` (env vars, endpoints), `src/app/api/forensic/*/route.ts`, `docs/reports/2026-09-26_runner-docker.md` (timings, memory), `docs/reports/cloud-expense-ledger.md`, `CLAUDE.md` (rules: ask before renting, secrets in `.env`, never push without approval).
