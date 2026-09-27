# Hosting the demo app on a Vultr VM for the video (Sat Sep 26, 2026, 23:03–)

**What:** the wired frontend and the pipeline API on one public Ubuntu box, so Hrushi can record the demo against real scores. The model, the TSV and the UI are unchanged; the box is a copy of `main` plus the gitignored assets, and is destroyed after the video. Handoff: `docs/handoffs/2026-09-26_vultr-hosting-handoff.md`. Deploy files: `deploy/vultr/` (README there gives the order of operations).

## Instance

| | |
|---|---|
| Provider / label | Vultr Cloud Compute, `hearsay-demo` (id `d8c2e85c-d73e-4ebe-82eb-2b10776ea617`) |
| Plan | `vc2-4c-8gb`: 4 vCPU, 8 GB RAM, 160 GB SSD, shared CPU (the High Performance AMD tier was not taken; the regular tier was fast enough, see timings) |
| Region | Atlanta (AMER) |
| Image | Ubuntu 24.04.5 LTS x64 |
| Network | public IPv4 only, `64.177.49.133`; ufw allows 22 and 80; 3000 and 8000 bind to 127.0.0.1 and are unreachable from outside (checked from the Mac) |
| Price | $40/mo, $0.055/h, no backups, no DDoS add-on |
| Created | 2026-09-27 03:03 UTC (Sat Sep 26, 23:03 EDT), by Nathan in the console with the `nathan-laptop` SSH key |

**URL for the video:** `http://64.177.49.133/` (plain http; no domain, no TLS).

## What runs on it

nginx :80 → `next start` on 127.0.0.1:3000 → `uvicorn hearsay.api:app` on 127.0.0.1:8000, all as systemd units under `deploy/vultr/` (`hearsay-web`, `hearsay-api`, and `hearsay-warm`, a oneshot that scores one demo clip after every API start so the models are resident before the first take). Checkout: `main` at `bd94e39`, user `hearsay`, `/home/hearsay/hearsay`. Assets rsynced from the Mac (2.44 GB sent, 3.47 GB on disk with the symlinked bundle copied): the three weight sets, the M1b probe, the handcrafted bundle, the M5 checkpoint, `fusion_v1` and `fusion_v2`, the 1,671 per-file JSONs and the submitted TSV. No secrets on the box; `/etc/hearsay.env` is not present (nothing needs it).

**Python environment.** The handoff's `uv sync` would have been wrong on Linux: `uv.lock` resolves torch to `2.14.0` from PyPI with the CUDA tree as Linux-only dependencies (`cuda-toolkit`, `nvidia-*`), several GB that a CPU box cannot use. `deploy/vultr/venv.sh` uses the Dockerfile's recipe instead: export the lock without torch, torchaudio and lightgbm, install with `--no-deps`, then torch/torchaudio from the PyTorch CPU index. Result on the box: `torch 2.14.0+cpu`. The units call `.venv/bin/uvicorn` directly rather than `uv run`, which would try to re-sync the CUDA lock. Worth a line in the README's setup section for anyone reproducing on Linux outside Docker.

## Timings (regular-tier 4 vCPU)

| Step | Time |
|---|---|
| bootstrap (apt, Node 22, swap, user, uv, clone, venv with CPU torch, `npm ci` + `npm run build`, units, nginx, ufw) | under 2 min wall clock (datacenter link; 395 npm packages in 9 s, Next compiled clean) |
| asset rsync from the Mac | 5.5 min at 7.5 MB/s average (10 MB/s steady) |
| model load (health `model_load_seconds`) | 8.9 s |
| first upload including the load (warm-up unit) | 23.9 s |
| upload after warm-up, through nginx and Next | 3.6–3.7 s per clip |
| API resident memory with M5 loaded | 3.1 GB (box: 7.7 GB RAM, 8 GB swap, 0 used) |

Tests on the box: `pytest tests/test_api.py tests/test_pipeline.py` → 85 passed, 9 skipped (the data-marked ones), 15 s.

## Parity smoke (`deploy/vultr/smoke.sh`, on the box, 23:20 EDT)

Health: `fusion: fusion_v2/constants.json`, scorers `['m1b_v3', 'handcrafted_v5', 'm5_xlsr_ft', 'spectra_aasist']`, `n_results: 1671`, git `bd94e39`.

| Clip | Box | Mac (`demo/README.md`) |
|---|---|---|
| `synthetic_apple_tts_samantha.wav` | 0.9979 synthetic, `A3_w0.2_E` | 0.9979 synthetic |
| `real_in_the_wild_28886.wav` | 0.0012 real, `A3_w0.2_E` | 0.0012 real |
| `synthetic_in_the_wild_5518.wav` | 0.9979 synthetic, `A3_w0.2_E` | 0.9979 synthetic |

All three match to four decimals under the shipped rule. Through nginx: `GET /` 200; `/api/forensic/presets` returns 3; `/api/forensic/export-tsv` sha256 starts `fb783076`; `/api/forensic/batch` reports 1,671 / 1,671, minDCF 0.0733, EER 3.53%, source "NSA draft review of the submitted file".

## Verified from outside (the Mac, 23:22 EDT)

- `http://64.177.49.133/` loads in 0.26 s; the first preset shows 0.001 Authentic, the other two `p=0.998`; the benchmark card shows 0.0733 / 3.53% / 0.0065 / 0.228 and "Shipped rule: A3 w0.2 + E".
- "Load the submitted 1,671-file run" fills the queue ("View All (1671)") with HGT files and their scores.
- Uploading `demo/real_in_the_wild_28886.wav` from the Mac through the public URL: `overallScore 0.0012`, `decision BONA_FIDE`, 3.6 s.
- "Export .TSV" from the public URL is byte-identical to `submissions/CrossExam_predictions.tsv` (`cmp` clean; sha256 `fb783076…`).
- Ports 3000 and 8000 time out from outside.
- Screenshot: `docs/img/vultr-hosting/2026-09-26_demo-landing.png` (headless Chrome, 1440×1000).

## Notes handed to Hrushi with the URL

1. Pull `main` before any UI change. Redeploy: `ssh hearsay@64.177.49.133 'cd hearsay && git pull --ff-only && npm run build'` then `ssh root@64.177.49.133 systemctl restart hearsay-web`.
2. The guided-tour copy still carries unmeasured numbers that must be fixed in `src/components/GuidedTour.tsx` before filming and rebuilt on the box: minDCF 0.1983 (the modal shows it on load), EER 4.82%, 48 kHz, 16 kHz roll-off, 60.014 Hz ENF, "Intercept #0042", a 2-hour timeline, "8 sensors", an "AI specialist". The dashboard proper shows the measured numbers.
3. Uploads take 3–4 s on this box; leave the spinner in the take. The first upload after a restart takes about 25 s (the warm-up unit normally absorbs this).

## Spend and teardown

Hourly at $0.055; the console showed $0.06 accrued at 23:20 EDT. Ledger row added at start (`docs/reports/cloud-expense-ledger.md`); the destroy row with the total is added at teardown. **Teardown:** Destroy in the Vultr console when Hrushi confirms the footage is in, then append the row. Nothing on the box is unique: the deploy files are committed, the assets live on the Mac.

_Teardown: pending._
