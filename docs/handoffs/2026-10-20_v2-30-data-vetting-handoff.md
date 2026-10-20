# Handoff: V2-30 vetting, the data pull aimed at the test misses

**Date:** 2026-10-20. **Chunk:** `docs/prod-chunks.md` → V2-30 (Phase 3), the vetting half only. **Branch to start from:** `v2` (`origin/v2`). **Stakes:** low (research and manifests; nothing trained). **Model:** a cheaper one is fine; this is reading, licences and bookkeeping. **No GPU, no MPS, no conflict with other lanes.**

## Why now

The V2-02 atlas (`docs/reports/2026-10-20_v2-02-error-atlas.md`) found v1's 24 test misses are a *style*: expressive, multi-sentence, pause-rich synthesis (voiced fraction ~0.54 vs 0.71 on the test set; speaker-drift min-cosine ~0.16 vs 0.27), wide-band and clean. No training population has that style, which is why re-weighting (V2-33) and re-fusing (V2-60) could not reach it. V2-33's order is now "pool first, then mine". This lane finds and vets the pool; a later lane downloads and folds it.

## What to produce

A vetting table, one row per candidate corpus, with these columns filled from primary sources (the dataset's own page, paper or licence file; cite the URL and the date read):

| Column | What goes in it |
|---|---|
| name, source, size (hours, clips, speakers, generators) | |
| licence, exact terms | commercial use? share-alike? redistribution? attribution? MLAAD is non-commercial and already a constraint (CLAUDE.md disclosure) |
| access | open download, gated form, request to authors; estimated turnaround |
| **style fit** | does it contain the target style? evidence: sample durations, utterance structure (sentences vs turns), delivery (read vs conversational vs expressive) |
| **overlap** | with anything we already hold or evaluate on: LJ Speech, LibriSpeech, VCTK, ASVspoof 2019, DiffSSD, MLAAD, In-the-Wild; and with the **AntiDeepfake training table** (`outputs/v2/backbone/cards/*.card.md`, 29 corpora), since those models may become columns and anything in their list is in-sample for them |
| generator / speaker grouping | can the fold file group by generator and speaker? are speaker ids available? |
| format | sample rate, codec, channel; anything that would become a container shortcut (v1's "the container is the label" finding, `README.md` null results) |
| cost | download size, where it would live (the external Crucial drive; the internal volume is tight, see `scripts/v2_backbone_fetch.py` A3) |

Two sides, both required (the model must not learn "pauses and variation = fake"):

- **Fakes in the target style:** MLAAD releases newer than our Apr 18 pull (models and release dates), long-form / conversational / expressive TTS sets, codec-LM voices, recent commercial voices with expressive modes, ASVspoof 5 (the chunk already names it), CodecFake-style sets. For each: which generators, and are they in the AntiDeepfake list.
- **Reals in the target style:** conversational and spontaneous speech (podcast, interview, meeting), e.g. Common Voice (read, so weak fit; say so), VoxCeleb (request form; interview speech, good fit), AMI or similar meeting corpora, podcast corpora with clear licences, telephone speech (Switchboard-type, licence check). Note recording-chain differences that would become a shortcut.

Then a **recommendation**: at most one real corpus and one or two spoof corpora to pull first, with the reason, the licence risk stated, and the shortcut risk stated. And a **profile plan**: how the two atlas features (voiced fraction, drift min-cosine) will be measured on a sample of each corpus before anything is folded (`hearsay.detectors.speech_gate` and `hearsay.detectors.speaker_drift` compute them; `scripts/v2_error_atlas.py` shows the join).

## Rules

- Research only: nothing downloaded in this lane beyond small samples needed to check format and style (a few files, into a scratch dir, deleted after). The pull is a separate, approved step.
- Every claim cites its source and date. A licence read from a mirror or a blog does not count; find the licence file or the dataset's own terms page.
- Do not open the sealed key, the holdout ledger or the sealed ledger. Do not run torch jobs.
- Report at the end; no time box.

## Deliverables

- `docs/reports/2026-10-20_v2-30-data-vetting.md` with the table and the recommendation.
- `docs/prod-chunks.md` V2-30 status line.
- Commit on a branch cut from `v2`; no push unless Nathan says so.

## Pointers

- Existing data notes: CLAUDE.md "Public datasets" and "Planned, not yet downloaded"; `docs/prod-chunks.md` V2-10 (corpus atlas) for the profile-card idea; `docs/reports/2026-10-10_v2-fusion-research.md` "What other HEARSAY teams did" (SideQuests used 11,900 copy-synthesis fakes; `outputs/v2/research/other-teams/`).
- Where data lives: `data/` (gitignored), the Crucial drive at `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`.
