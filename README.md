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

Planned; see the Docker section of [docs/plan.md](docs/plan.md). The target interface is to run offline on a directory of test audio and write `teamName_predictions.tsv`:

```bash
docker run --network none -v <test_dir>:/data:ro -v <out_dir>:/out hearsay
```

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
