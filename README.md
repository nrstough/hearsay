# HEARSAY

Audio authentication for the NSA HEARSAY challenge at HackGT 13. Give it any audio file and it returns the probability that the voice is synthetic (0.0 real, 1.0 synthetic), along with the forensic evidence behind that call.

A rule-based orchestrator chooses which detectors to run on each file. The detectors cover:
- container and metadata
- spectral
- prosody
- compression
- speaker-embedding drift
- splice
- ENF
- deep SSL anti-spoofing

A logistic stacker fuses their scores into the final probability.

_The team will rewrite this README in our own words before submission: approach, architecture, what worked and what didn't._

**Joining now? Start with [docs/STATUS.md](docs/STATUS.md)** (current state, data, how to help).

Project context, the detector contract, the model ladder, and working rules are in [CLAUDE.md](CLAUDE.md). The plan is in [docs/plan.md](docs/plan.md) and the NSA brief is in [docs/nsa-challenge.md](docs/nsa-challenge.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```bash
uv sync                  # creates .venv (Python 3.12) with all dependencies
cp .env.example .env     # fill in any secrets locally; never commit .env
uv run pytest            # tests
uv run ruff check .      # lint
```

Datasets go in `data/`, model weights in `weights/`, and submission TSVs in `submissions/`. All of these are gitignored except `submissions/log.csv`.

## Docker

The image runs the whole pipeline offline on CPU (linux/amd64): every engineered detector, the XLS-R probe and Spectra-AASIST, the persisted fusion rule, the non-speech gate, and the TSV writer. Nothing is downloaded at run time; all weights are baked in.

```bash
bash docker/build.sh                      # builds hearsay:<stamp> and hearsay:latest for linux/amd64
docker run --network none \
  -v <test_dir>:/data:ro -v <out_dir>:/out \
  -e HEARSAY_TEAM=<teamName> hearsay
```

Outputs under `<out_dir>`: `<teamName>_predictions.tsv` (header `filename<TAB>cm-score`, probability that the file is synthetic, 1.0 = synthetic), one JSON per file under `results/` with every detector's score, evidence and the routing log, a resumable `results.jsonl` (rerun the same command to continue after a crash), `timings.csv` and `run_meta.json`. A run never overwrites an earlier TSV.

Row order comes from NSA's template when one is available: mount it and set `HEARSAY_TEMPLATE=/tmpl/key.tsv` (`-v <key.tsv>:/tmpl/key.tsv:ro`), or drop the `.tsv` beside the audio and the entrypoint finds it. Without a template the rows are the sorted filenames. Other settings: `HEARSAY_RULE` (`zmean`, the default, or `stack_nonlj`), `OMP_NUM_THREADS` (default: the container's CPU quota, at most six). Extra arguments go to `scripts/run_pipeline.py`, for example `--limit 50 --compare-tsv /ref/logged.tsv` for a parity check.

Speed: about 4 s per file inside the image on an Apple Silicon Mac through Rosetta (the whole test set in about two hours), and about 0.8 s per file natively on the Mac's CPU; a native amd64 box should land in between. All models stay loaded; peak memory is under 3 GB.

Checks: `bash docker/smoke.sh` runs three files (WAV, MP3, FLAC) with a reversed template through the image with `--network none` and asserts the row order, byte-identical repeat runs and the in-image self-checks; `docker/parity.py pcm-hash` compares decoded audio between the Mac and the image; `uv run pytest -q tests/test_docker_image.py` checks the build files without Docker. Building needs Docker with buildx (Colima with Rosetta works on Apple Silicon: `colima start --vm-type vz --vz-rosetta`). Design and results: `docs/specs/2026-09-26_k-docker-image.md`.

## Layout

```
src/hearsay/            application code (detectors/ holds the detector contract)
tests/                  pytest suite
scripts/                one-off and ops scripts
docs/                   plan.md, nsa-challenge.md, specs/, reports/, handoffs/, consults/
submissions/            log.csv (tracked) + submission TSVs (ignored)
data/ weights/ models/  local only
.claude/                Claude Code skills and Codex review scripts (shared with the team)
```
