# HEARSAY

Audio deepfake detector for HackGT 13 (NSA challenge): any audio clip in, a 0–100% likelihood that the voice is AI-generated out. It also drives a live demo, a pan-tilt listening head that turns toward whichever of two phones is talking and turns red on the cloned voice.

Project context, data contract, model ladder, and working rules are in [CLAUDE.md](CLAUDE.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```bash
uv sync                  # creates .venv (Python 3.12) with all dependencies
cp .env.example .env     # fill in any secrets locally; never commit .env
uv run pytest            # tests (none yet)
uv run ruff check .      # lint
```

Datasets go in `data/`, model weights in `weights/`, and submission CSVs in `submissions/`. All of these are gitignored except `submissions/log.csv`.

## Layout

```
src/hearsay/     application code
tests/           pytest suite
scripts/         one-off and ops scripts
docs/            scoping.md, master-doc.md, specs/, reports/, handoffs/, consults/
submissions/     log.csv (tracked) + submission CSVs (ignored)
data/ weights/   local only
.claude/         Claude Code skills and Codex review scripts (shared with the team)
```
