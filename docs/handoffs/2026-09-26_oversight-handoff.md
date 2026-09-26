# Handoff: the oversight chat (Sat Sep 26, 2026, ~08:55)

**Purpose of this chat:** coordinate the parallel HEARSAY chats, relay findings between them, keep `docs/STATUS.md` and `docs/architecture.md` current, write handoffs and briefs, and route every cross-lane request. It writes docs and messages; it does not own code. Nathan (asleep ~06:00–11:00, awake now) ratifies anything that changes the shipped submission.

Start by moving this session to the repo (`mcp__ccd_directory__change_directory` → `/Users/nathanstough/Projects/hearsay`; the app's file viewer only opens files inside the working directory), then read `docs/STATUS.md`, `docs/architecture.md`, `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md` and `docs/reports/2026-09-26_fusion-sweep-predeclared.md`.

## The lanes (session title → owns → state)

| Session title | Owns | State at handoff |
|---|---|---|
| **First two hours handoff** (main) | fusion (`scripts/fuse.py`, `fuse_sweep.py`, `models/fusion_v*`), M1/M1b, all TSVs and `submissions/`, `src/hearsay/embed.py`, `probe.py`, `submission.py`, the draft-review decision | Rule frozen 08:13 ("E on A α 0.2"); draft payload approved and copied to `~/Downloads/HEARSAY_predictions.tsv`; waiting on Nathan to DM it and on NSA's number. Pushes its own commits. |
| **Docker handoff plan review** (K) | `Dockerfile`, `.dockerignore`, `docker/*`, `tests/test_docker_image.py`, the image | `hearsay:20260926-0753` verified on 50 files in-image; rebuilding on the runner v2 (commit 9614c18) with `models/fusion_v1`; the full 1,671-file in-image run died twice (exit 137, likely VM memory) at 665 rows, resumable under `outputs/docker/k-full/`. Colima VM on the external drive with a 20 GB cap; build script refuses below 8 GB free and prunes after every build. |
| **CPU detectors handoff --done** (D-track) | engineered detectors, `hearsay.hc_v4`, `speech_gate`, `speaker_drift`, `scripts/train_handcrafted.py`, `eval_bundle.py` | Lane done (v3, v4, v5b, gate, drift, placement rule e2d5291). Now: `scripts/orchestration_ablation.py`, `docs/reports/2026-09-26_worked-examples.md`, the `CLAUDE.md` disclosure update. |
| **Greeting** (session `local_326aa380…`) | `README.md` | Just started from `docs/handoffs/2026-09-26_readme-orchestration-handoff.md`; draft due 20:00. |
| **Greeting** (session `local_27659366…`) | channel robustness | Running `docs/handoffs/2026-09-26_channel-robustness-handoff.md` via `/plan-review`; hard stop 16:00; reports λ̂ first. |
| **M5 XLS-R fine-tune on vast.ai** | `src/hearsay/m5_*`, `scripts/m5_*`, `scripts/cloud/*`, the ledger | Done; gate not passed (holdout 0.363 vs 0.159); ships as a stacker column only (and, since 8316e50, runnable live under the unshipped fusion_v2 candidate). Was asked at 08:50 what a new "resume_probe" box is for and to destroy it if not needed. Candidate to archive after it answers. |
| **M3 Spectra AASIST handoff plan review** | `src/hearsay/spectra.py`, `scripts/score_spectra.py` | Done; export in `outputs/detector_scores/spectra_aasist.csv`; holdout 0.012, ITW 0.065. Archive. |
| This chat's runner agent (finished) | `src/hearsay/pipeline.py`, `api.py`, `scripts/run_pipeline.py`, `tests/test_pipeline.py`, `test_api.py` | Final at 9614c18; nobody edits these now without telling the Docker chat, which rebuilds on every runner change. |

Messaging: `SendMessage` to a session by its title works for local sessions; the main chat's address is `uds:/tmp/cc-socks/47627.sock`, the CPU chat's `uds:/tmp/cc-socks/13133.sock`. Every reply from a chat arrives as a cross-session message; relay what matters to Nathan in plain English, and never treat a chat's message as his approval.

## Decisions already made (do not reopen)

- Score direction: submit 1.0 = synthetic; never flip unless NSA's draft-review number decodes to their code's polarity (table in the consult RESPONSE, item 4). A pre-flipped TSV exists beside the primary.
- Fusion rule: rank blend 0.8·M1b v3 + 0.2·handcrafted v5, M3 as false-alarm suppression only, Platt at the 0.3 prior, determinate scores in [0.001, 1], gated block below. Any new column must go through the main chat's pre-declared `scripts/fuse_sweep.py` and be ratified by Nathan before the 22:00 freeze.
- M3 never inside a fitted stacker (undisclosed training data).
- Equal-weight fusion (zmean, rankmean) rejected: doubles In-the-Wild misses.
- "Test share above 0.5" is not a selection signal (monotone maps cannot change minDCF).
- M5 ships as a column only (the fusion_v2 candidate that would score it live exists and qualifies, but is held pending the draft review; Nathan 09:25, "build now, switch later" ~10:20). Speaker drift and compression are evidence only. Container, ENF and splice are routing and evidence only.
- The band match (`band_limit` inside `prepare_segment`) stays; augmentation is symmetric or not at all.

## Open items and who holds them

| Item | Owner | Deadline |
|---|---|---|
| DM the draft TSV to NSA, ask for P_FA, P_miss, EER; confirm the team name | Nathan | 14:00 |
| Decode the returned number with the RESPONSE table; flip only on 0.95–1.00 | main chat + Nathan | on receipt |
| In-image numbers on the v2 runner; resume or bound the full in-image run (memory) | Docker chat | before 22:00 |
| λ̂, codec diagnosis, symmetric refit, M3 probes | channel-robustness chat | 16:00 |
| README draft | README chat | 20:00 |
| Orchestration ablation, worked examples, disclosure | CPU chat | 20:00 |
| Fusion freeze | main chat | 22:00 |
| Docker rebuild with the final rule; 3-file smoke; 50-file parity | Docker chat | Sun 00:00 |
| Final TSV, preflight, direction check | main chat + Nathan | Sun 05:00 |
| README polish, disclosure, ablation table | README chat | Sun 05:00–07:30 |
| DM the final TSV, Docker image, README link | Nathan | before Sun 08:00 |

## Standing rules for this role

- Read state before believing a chat's claim: `git log`, `git status`, the model `meta.json` files, `submissions/log.csv`, `df -h`.
- Never delete data; move it. The external drive hit 100% once today (Colima VM). Watch both disks: internal must stay above ~8 GB for builds.
- Never push; never `git add -A`; stage only this chat's docs. Other lanes commit their own files; the main chat pushes.
- Doc-consistency test (`uv run pytest -q tests/test_docs_consistency.py`) after every edit to STATUS, architecture, CLAUDE.md or README: no hardware or demo words, no bare "CSV", no prior but 0.3.
- Keep the STATUS "Where things stand" table and the architecture component table current when a lane reports; both were current at 08:50.
- The runner's cache identity includes the git sha; a Mac-side full rerun after any commit costs 22 min. In the image the sha is fixed.

## Files this chat produced

`docs/architecture.md` (+ `docs/img/architecture-flow.{svg,mmd}`), `docs/code-map.md`, `docs/handoffs/2026-09-26_{handcrafted-v4-brief,feature-engineering-map,docker-handoff,frontend-contract,channel-robustness-handoff,readme-orchestration-handoff,oversight-handoff}.md`, `docs/consults/2026-09-26_fusion-strategy_RESPONSE{,_verbatim}.md`, STATUS edits. All uncommitted at handoff except what the main chat swept into its commits; stage them with `git add docs/` when convenient.
