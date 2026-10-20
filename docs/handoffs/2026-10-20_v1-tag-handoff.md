# Handoff — v1 tag (2026-10-20)

**Purpose of this chat:** create the annotated tag `v1` on commit `c8260b4` (the HackGT 13 state at the 8 AM Sun Sep 27, 2026 judging cutoff) and push it to `origin`. A local session is needed because the cloud session that wrote this could push only to its own branch: every tag push was cut off with "remote end hung up".

## Context
- The remote had no tags and no releases (checked with `git ls-remote --tags origin`, 2026-10-20).
- `c8260b4` (Sun Sep 27, 00:54 EDT, "fix(tour): align guided tour copy…") is the tip of `main` and the last commit before the 8 AM cutoff. Nathan ruled that the submitted state runs up to 8 AM on judging day, so `v1` goes on `c8260b4`, not on the 00:09 FINAL commit `7856a18`.
- Branches: `origin/hrushi` (tip `e98ac2a`) and `origin/claude/elegant-fermi-uqdz2u` have nothing that isn't on `main`. `origin/tharakesh` has 1 commit and `origin/claude/hearsay-docs-site-p1-rw35db` has 8 that never reached `main`; they are **not** part of v1.

## Working branch / worktree
Any local clone works. No code changes; the tag goes straight onto the existing commit.

## Environment / setup
None beyond a clone with push access to `nrstough/hearsay`.

## What to do next
1. Run `git fetch origin`, then check that `git rev-parse origin/main` still starts with `c8260b4`. If `main` has moved, tag `c8260b4` anyway; that's the commit wanted, not the branch tip.
2. Tag it and push the tag:
   ```
   git tag -a v1 c8260b4 -m "HEARSAY v1: HackGT 13 state at the 8 AM Sun Sep 27, 2026 judging cutoff (submitted CrossExam_predictions.tsv, sha fb783076)"
   git push origin v1
   ```
3. Verify: `git ls-remote --tags origin` lists `refs/tags/v1`, and `git rev-list -n1 v1` prints `c8260b4…`.
4. Optional: on GitHub, go to Releases → "Draft a new release" and choose the existing tag `v1`, to get a release page you can link from Devpost.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)
- No tests are relevant (tag only). The verification is step 3.
- At-risk: `submissions/CrossExam_predictions.tsv` (sha256 `fb7830762691d04d…`) is gitignored, so the tag does **not** contain it. Keep it on the Mac and the external drive (`/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`). Status unknown from the cloud session; check it. The same applies to `data/`, `weights/`, `models/` and `outputs/`.
- In flight: nothing.

## Analytical notes
Final submission: Sun 00:09, rule A3_w0.2_E (fusion_v2). NSA scored the byte-identical draft at minDCF 0.0733 / EER 3.534%. Outer-holdout validation minDCF was 0.0065 (clean-only 0.0065). Sources: the last rows of `submissions/log.csv`.

## HEARSAY status
Rung: done (frozen after FINAL) · Latest valid TSV: `submissions/CrossExam_predictions.tsv` (fb783076) · Val minDCF (outer holdout): clean 0.0065 / augmented 0.0065
Time left to 8 AM Sun: the event is over · Detector-contract changes pending: none

## Pointers
`submissions/log.csv` (FINAL rows), `docs/STATUS.md` (freeze header), commit `7856a18` (FINAL submission record).
