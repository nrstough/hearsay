---
name: validate-training-data
description: "Validate an ML training dataset or manifest before training — label distribution and imbalance, feature/tensor integrity (shape, range, NaNs, all-zero rows), sequence-length stats, split leakage, and metadata/asset consistency. Use when the user says 'validate the training data', 'check the dataset', 'is this manifest clean', 'why is the model degenerate', or a new dataset/manifest has just been produced."
user_invocable: true
---

# /validate-training-data — Dataset / manifest validation

The job is to catch the data problems that produce a confidently broken model, **before**
the training run spends money. Every check reports numbers, not adjectives.

Any arguments passed to this skill name the dataset, manifest, or split to validate. With
no arguments, resolve the newest dataset artifact and say which one you picked.

## Step 0 — Resolve the artifact and its schema (inspect, don't assume)

1. Read `CLAUDE.md` / `AGENTS.md` and the training docs for where datasets live and which
   manifest is canonical. Regenerable data often lives **outside the repo** (an external
   drive, object storage) — resolve the real path rather than reading a stale local copy.
2. Find the artifact: manifest JSON/JSONL, parquet or .csv tables, a TFRecord/webdataset shard set,
   or a label directory.
3. **Derive the schema from the data**, then state it back: record count, fields per
   record, feature shape/dtype, label vocabulary, group/split keys. Confirm it matches
   what the training code actually reads (`git grep` the loader and the field names) —
   a manifest field the loader ignores is a silent no-op, and a field the loader needs
   but the manifest lacks is a crash-at-epoch-0.
4. Note which records are **hand-curated ground truth** vs **draft / auto-labeled**.
   Draft labels are not ground truth and must never be counted as such in the report.

**HEARSAY specifics:**
- The split key is the **generator** (TTS / voice-conversion system). No generator may
  appear in both train and validation — clip-level splits are invalid for this project.
  Also check speaker and source-recording overlap across splits.
- Everything the model sees must be **16 kHz mono**. Report the sample-rate / channel /
  codec histogram **per class** — if fakes and reals differ in native format, the model
  can learn the codec instead of the voice (a leak, not a signal). Same for duration and
  leading/trailing silence per class.
- Compare our real/fake ratio with whatever is known about the NSA set, and report it.
- Augmented data (laundering: codec/telephony round-trips, noise, simulated replay/RIR,
  band-limiting) is tracked separately; the clean-only validation set must stay clean.
  Also compare duration and peak level per class against the NSA test inventory.
- The validation split is never trained on. Flag any path where it could leak into
  training (feature scalers, calibration, stacking folds).

## Checks

### 1. Label distribution and imbalance
- Count each label per target/head (multi-output datasets need a per-head table, not a global one).
- Flag any head where one class exceeds ~80% — that's the classic degenerate-model cause
  (the model predicts the majority class and scores well on accuracy).
- Report the effective positive count, not just the ratio: 3% positives out of 200 rows is
  a different problem from 3% out of 200,000.
- Flag labels absent entirely, and labels outside the declared vocabulary.

### 2. Feature / tensor integrity
- Every record's feature shape matches the declared shape exactly. Report the histogram of
  shapes found, not just "mismatch".
- Value ranges: normalized features inside their stated range; confidences in `[0,1]`;
  any padding region actually zero.
- **NaN / Inf** count, and which fields.
- **All-zero or constant rows/frames** — usually an upstream detection or extraction
  failure that silently became training data. Report the rate; a few percent is normal,
  a fifth is a bug.
- Duplicate records (same content hash) — inflates apparent dataset size and leaks across
  splits.

### 3. Sequence / shape statistics (for sequence models)
- min / max / mean / median / p5 / p95 length.
- Count below the model's minimum useful length, and count at the upper clamp (records
  sitting exactly at the max are usually untrimmed, not long).
- Length distribution **per class** — if the classes differ in length, the model can
  learn length instead of the signal. This is a real leak; call it out.

### 4. Segment / boundary quality (when records are trimmed from longer sources)
- Trimmed vs fallback-to-full counts, and the length distribution of each.
- Fallback rate per source/venue/session — a rate concentrated in one source is a bug in
  that source's handling, not a data-quality fact.

### 5. Split integrity and leakage
- Identify the grouping key that must not straddle splits (event, session, subject,
  source video, patient). **Check it doesn't.** Group leakage is the most common reason a
  validation number is unreproducible in the field.
- Report per-split counts and per-split label distribution — a split whose class balance
  differs from train invalidates the comparison.
- Flag any record appearing in two splits.

### 6. Metadata and asset consistency
- Every referenced asset path exists and is readable (report the missing count and a few
  examples). Sample a handful and confirm they're non-empty and the expected format.
- Cross-field consistency: declared type/class vs the label, declared resolution/duration
  vs the actual file, timestamps in range.
- Distribution across the categorical metadata (type, source, quality tier, camera/angle)
  — a category with a handful of records cannot support a per-category claim.

## Output format

A summary table first:
```
| Check | Result | Verdict |
|-------|--------|---------|
| Records | 4,092 | — |
| Label balance (head X) | 63% / 37% | OK |
| Shape integrity | 4,090 ok / 2 wrong shape | FIX |
| Group leakage | 0 straddling groups | OK |
```
Then a section per failed check with the specific offending record ids (up to ~20) and
what to do about them.

Close with a **train / don't-train verdict** and the blocking items, ranked.

## Rules

1. **Numbers, always.** "Some clips are short" is not a finding; "412/4,092 (10.1%) under
   10 frames" is.
2. Compute from the data. Never restate a count from a doc or a manifest header — those
   drift, and reconciling them is itself a finding.
3. **Draft labels ≠ ground truth.** Report them separately and never fold them into a
   ground-truth count.
4. Name record ids for every flagged item so they can be quarantined or re-derived.
5. Don't modify the dataset. Propose the quarantine/fix list; the user decides.
6. If the artifact is large, sample — and state the sample size and method, so nobody
   mistakes a sampled rate for an exact count.
