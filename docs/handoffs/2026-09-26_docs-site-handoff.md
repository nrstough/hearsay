# Handoff: the documentation site (Sat Sep 26, 2026, ~13:10)

**Purpose of this chat:** turn HEARSAY's documentation into a beautiful, self-contained HTML site inside the repo, with flowcharts and plain-language explainers, without changing a single number or claim. Documentation is 20% of the score ("in your own words: approach, architecture, what worked, what didn't"), and diversity/depth is another 20% that the docs have to make visible. The markdown stays the source of truth; the site is a faithful, better-looking copy of it plus diagrams and explainers.

Read first: `README.md` (397 lines; the judge-facing story), `docs/architecture.md` (five Mermaid diagrams already), `docs/STATUS.md` (the running record with today's decisions), then this file. Skim `docs/reports/2026-09-26_worked-examples.md` (eight test files walked through) and `docs/reports/2026-09-26_fusion-sweep-predeclared.md` (how the rule was chosen).

## Context

- The system is done and the draft went to NSA at 12:30. The shipped rule is A3 w0.2 + E (`models/fusion_v2/constants.json`): ranks 0.6 XLS-R probe (M1b v3) + 0.2 handcrafted spectral/prosody (v5) + 0.2 M5 (fine-tuned-head XLS-R), then Spectra-AASIST used only to pull files toward real, Platt map, undetermined files pinned at the bottom. Ten detectors run per file; four are fused, the rest are evidence and routing. Numbers: holdout 0.0065, In-the-Wild 0.228, inner 0.135.
- The README already tells the story well and is being polished by its own chat until ~05:00 Sunday. `docs/architecture.md` has the component table and the diagrams. `docs/STATUS.md` has the day's findings, including the honest negatives (equal-weight fusion rejected, M5 failed its gate and ships only as a minority vote, four detectors dead by mechanism, the codec hunt negative, a λ̂ estimate that was withdrawn and corrected). All of that is material for the site.
- **The repo is private, on a free GitHub plan.** That means: no GitHub Pages, and no judge can see anything until Nathan makes the repo public (or adds the judges as collaborators). So the site must work as plain files opened from the repo (`docs/site/index.html` double-clicked, or `python3 -m http.server` in `docs/site`), and become a Pages site with zero changes once the repo is public (Settings → Pages → deploy from `main`, folder `/docs`; URL `https://nrstough.github.io/hearsay/site/`). Making the repo public before Sunday 08:00 is on Nathan's list; the README chat will add the URL once it exists.
- A separate frontend (Hrushi's Next.js app at the repo root and a static site in `web/`) exists and is being rewired to the real pipeline. **The docs site is not that.** Do not touch `web/`, `src/app/`, `src/components/` or `package.json`.
- GitHub renders Mermaid inside markdown natively, so the markdown diagrams already display on GitHub. The site's job is presentation, navigation, explainers and new diagrams, not replacing what GitHub already renders.

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared by several chats. This lane owns **`docs/site/`** only (plus, if you write a generator, `scripts/build_site.py`). Never edit `README.md`, `docs/STATUS.md`, `docs/architecture.md`, `CLAUDE.md` or anything under `src/`, `scripts/` (other than your own), `web/`; if the site needs a change in a source document, message the oversight chat ("Oversight handoff execution") and it routes it. Stage only your files (`git add docs/site scripts/build_site.py`), never `git add -A` (the repo root now carries a JS tree that would come along). Ask Nathan before pushing; the oversight chat pushes on his standing instruction, so tell it the commit hash.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
```

No new Python or Node dependencies in the repo, please: `uv.lock` and `package.json` are shared with lanes that are mid-work. Tooling you may use without touching the repo's dependency files: hand-written HTML/CSS/JS; Mermaid from the jsDelivr CDN (`https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js`) for live diagrams, with the committed `docs/img/architecture-flow.svg` as the offline fallback for the main flow; `npx -y @mermaid-js/mermaid-cli` in your scratchpad if you want static SVGs of the other diagrams (it downloads a headless Chromium once, ~300 MB, fine on the internal disk with 30 GB free); the built-in browser to check rendering. If you decide a generator is faster than hand HTML, write it as a single dependency-free Python script (`scripts/build_site.py`, standard library plus what `uv sync` already installed, e.g. `markdown` is **not** installed; check with `uv run python -c "import markdown"` before assuming) and commit its output under `docs/site/`, so the site works with no build step.

Preview:

```bash
cd ~/Projects/hearsay/docs/site && python3 -m http.server 8080
```

## What to do next

1. **Design first, ten minutes, on paper.** One palette, one type scale, light and dark, phone width. Every page: a left nav (or top nav on phone), a title, a one-paragraph "in plain words" lead, then the detail. Every number carries a small source link to the markdown file and section it came from. Every page footer says "Generated from `<file>` at commit `<sha>`; the markdown is the record."
2. **Build these pages, in this order** (the first four carry the grade):
   1. **Home**: what HEARSAY is in three sentences; the verdict numbers (holdout 0.0065, In-the-Wild 0.228, the sponsor-cost twin 0.239; the draft's returned minDCF once NSA sends it, blank until then); one big pipeline flowchart; links.
   2. **How it decides**: audio → decode/band-match/trim → router → ten detectors → four fused by rank → Spectra false-alarm step → Platt → pinned block → TSV. A flowchart for the pipeline, a second one for the router (what facts route what: container, speech gate, disagreement), a third for the fusion rule as arithmetic on a worked file. Use one of the eight worked examples with its real evidence sentences from `docs/reports/2026-09-26_worked-examples.md`.
   3. **The metric and the default answer**: what minDCF is, why false alarms cost 4× (NSA's "analyst scenario"), why only ranking matters, why the operating threshold is near 0.9 and not 0.5, and the game-theory hint from NSA's brief answered by the pinned block, with the little table showing the block is right under both readings of the score (consult item 5). A diagram of the score line with the block at the bottom.
   4. **What worked / what did not**: the honest log, one card each with the number and the mechanism: shortcuts found and removed (7.2 kHz wall, loudness, leading silence, tiling, container = label); the domain shift in the handcrafted features and the augmented v5 fix; equal-weight fusion rejected (doubles wild misses); M5 gate not passed, ships as a minority vote; Spectra kept out of the vote because its training data is undisclosed, even though a full vote scores better on our proxies; the four dead detectors (container, ENF, splice, compression) with the reason each cannot work on this test set; the codec hunt (negative); λ̂ estimated, withdrawn, corrected to 0.51. Judges reward this page.
   5. **Detectors**: ten cards, each: technique (rubric name), what it measures, role (fused / evidence / routing / gate), its numbers, one real evidence sentence, source link. Map each to the rubric's list so the diversity score is easy to award.
   6. **Validation**: the fold scheme as a diagram (generator- and speaker-grouped, inner folds, outer holdout, In-the-Wild never trained on, the test set unlabeled), pre-declared sweeps as a timeline (08:13 rule, 09:14 M5 candidate, 12:03 switch, the acceptance rules written before results), the draft-review decoding table as a graphic.
   7. **Reproduce**: the commands from the README verbatim (runner with `--team CrossExam`, the Docker evidence line), the repository layout.
   8. **Credits and AI use**: the disclosure from `CLAUDE.md` and the README, unchanged in substance.
3. **Diagrams**: reuse the five Mermaid blocks in `docs/architecture.md` where they fit; add the router tree, the fusion arithmetic, the fold scheme, the score line with the pinned block, the decision timeline. Keep every diagram legible on a phone (split rather than shrink).
4. **Sync pass** at the end and again late tonight: re-read `README.md` and `docs/STATUS.md` and copy any changed number. Expected changes before Sunday: the returned draft minDCF, the channel lane's Spectra-probe verdict (~14:00), possibly a final-rule note. Do the last sync after 04:00 Sunday, before the README's final polish window (05:00–07:30), and tell the README chat what the site says so the two agree.
5. **Check**: open every page in the built-in browser at desktop and phone widths, light and dark; click every internal link; confirm every number against its source. Then commit under `docs/site/` and message the oversight chat with the hash.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q tests/test_docs_consistency.py` → 52 passed. It pins `README.md`, `CLAUDE.md`, `docs/plan.md` (and STATUS/architecture wording): it does not look at `docs/site/`, but run it anyway after any sync so you notice if a source document changed under you.
- Link check (no dependency): `cd docs/site && uv run python -c "import re,pathlib,sys; bad=[(p,l) for p in pathlib.Path('.').rglob('*.html') for l in re.findall(r'href=\"([^\"#:]+)', p.read_text()) if not (p.parent/l).exists()]; print(bad or 'links OK'); sys.exit(bool(bad))"`.
- Numbers to reproduce on the Home page, all from `README.md` "Results at a glance" and `docs/STATUS.md`: shipped rule inner 0.135 / holdout 0.0065 / In-the-Wild 0.228 brief, 0.239 miss-averse; previous rule 0.140 / 0.014 / 0.260 / 0.267; M1b v3 alone holdout 0.072, ITW 0.343; Spectra ITW 0.065; test share above 0.5 27.3%; 1,671 files; 0 gated. If your copy disagrees with the README, the README wins and you tell the oversight chat.
- At-risk: nothing regenerable belongs to this lane; everything you make is committed under `docs/site/`. Do not put the site's images under `outputs/` (gitignored) or `web/`.
- In flight, other lanes: the README chat (polish until ~05:00 Sunday; address `local_326aa380-ab14-4eaa-bb59-5ee0cbbbb1ae` for SendMessage); the channel lane's Spectra probes (~14:00; their verdict goes into STATUS and the "what did not work" story); NSA's returned draft minDCF (unknown time; placeholder until then); Hrushi's frontend rewire (separate; not yours).

## Analytical notes

- Score direction: 1.0 = synthetic everywhere. If any diagram shows a score line, synthetic is at the top and the pinned block sits below 0.001.
- Costs: NSA's brief says false alarm ×4, ~70% real; normalized minDCF = 9.33·P_FA + P_miss. The sponsor's shipped script uses a 50/50 prior and reads higher as bona fide; we report both. Don't explain this at length on the Home page; the metric page carries it.
- The ten detectors and their roles: fused: `m1b_v3` (deep SSL probe), `handcrafted_v5` (spectral + prosody), `m5_xlsr_ft` (fine-tuned-head SSL), `spectra_aasist` (deep anti-spoofing, suppression only, weight 0). Evidence: compression, ENF, splice, speaker_drift. Routing: container. Gate: speech_gate.
- Rubric mapping (for the Detectors page): container/metadata → container; spectral → handcrafted, Spectra; prosody → handcrafted; ENF → enf; compression → compression; speaker-embedding consistency → speaker_drift; deep anti-spoofing → M1b, M5, Spectra; splice → splice; orchestration → router + gate + the ablation table (router on vs off: holdout 0.0065 vs 0.020, In-the-Wild 0.229 vs 0.274).
- Tone: plain English, short sentences, no marketing. The README's voice is the model. Never invent a number, a claim or a result; if a page needs a number that does not exist in the markdown, leave it out.
- Time: it is Saturday ~13:10. The final TSV freezes Sunday ~05:00; nothing in this lane touches it. A first complete version of the site by ~19:00 lets the README chat and Nathan review it in the evening; the sync pass is the only work after that.

## Pointers

- `README.md`: the story, the tables, the run lines, the disclosure.
- `docs/architecture.md` and `docs/img/architecture-flow.{mmd,svg}`: component table and the five diagrams.
- `docs/STATUS.md`: today's findings and decisions, with times.
- `docs/reports/2026-09-26_worked-examples.md`: eight real files walked through (best material for "how it decides").
- `docs/reports/2026-09-26_fusion-sweep-predeclared.md`: the pre-declared sweeps, the M5 addendum, the acceptance rules.
- `docs/reports/2026-09-26_draft-review-branches.md`: the decoding table.
- `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`: items 4 and 5, the decoding table's origin and the pinned-block argument.
- `docs/reports/2026-09-26_cpu-detectors.md`, `2026-09-26_gate-and-drift.md`, `2026-09-26_handcrafted-v4.md`, `2026-09-26_m3-spectra.md`, `2026-09-26_m5-xlsr-finetune.md`, `2026-09-26_channel-robustness.md`: per-detector detail and the negatives.
- `docs/nsa-challenge.md`: the brief; `docs/reports/2026-09-25_sponsor-questions.md`: NSA's answers, including today's.
- `docs/code-map.md`: where each module lives, if a page needs a code link.
