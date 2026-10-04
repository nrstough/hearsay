# Layer bake-off: XLS-R L7 + L20 does not confirm

Sat Oct 3, 2026.
- **Spec, fixed before the run:** `docs/specs/2026-10-03_layer-bakeoff.md`, commit `f96109e`.
- **Script:** `scripts/bakeoff_layers.py`, with three stages: `extract` on the GPU, `probes` on the CPU (the two ran in parallel), then `report`.
- **Outputs:** `outputs/v2/bakeoff/report.json` and logs.
- **Writes:** nothing under `models/`, `submissions/` or `splits/`.

## Verdict

**P1 (L7 + L20) does not confirm, and neither does any other candidate.**

The pair is a large, real gain on both wild audio and the holdout, with bootstrap 5th percentiles above zero. It fails two of the pre-declared rules:
- **R1:** it makes the ElevenLabs fold worse.
- **R4:** it makes 20 dB noise worse.

That trade is real, not noise, and it is the kind of trade a rule chosen on wild audio alone would have hidden.

## Sanity

- **The L7 probe reproduces shipped M1b exactly:** inner 0.2649, holdout 0.0717, In-the-Wild 0.342. Test logits match the export to 2e-15.
- **Re-extracted "none" cohort vs the stored In-the-Wild scores:** max |Δlogit| is 0.0066. That is GPU nondeterminism and is negligible against the logit spread.
- **F(P0) reproduces the shipped fusion:** inner 0.1351, holdout 0.0065, In-the-Wild 0.228.

## Probe level

minDCF uses the brief's cost. "Averse" is the sponsor code's cost. P_FA and P_miss are measured at each candidate's own inner threshold.

| | Inner | Fold 4 (ElevenLabs) | Holdout | ITW | ITW gain, boot p5 | ITW averse | ITW P_FA / P_miss | Noise cohort | MLAAD miss | Test share |
|---|---|---|---|---|---|---|---|---|---|---|
| P0 L7 | 0.265 | 0.464 | 0.072 | 0.342 | | 0.296 | 0.8% / 27.9% | 0.485 | 17.3% | 24.4% |
| **P1 L7+L20** | **0.255** | **0.536** ✗ | **0.030** | **0.267** | +0.024 | 0.265 | 0.5% / 26.3% | **0.547** ✗ | 18.2% | 24.2% |
| P2 L7+L21 | 0.258 | 0.560 ✗ | 0.019 | 0.268 | +0.026 | 0.241 | 0.45% / 23.9% | 0.540 ✗ | 18.2% | 24.5% |
| P3 L7+L18 | 0.254 | 0.582 ✗ | 0.030 | 0.282 | +0.026 | 0.245 | 0.85% / 24.2% | 0.525 ✗ | 19.9% | 24.2% |
| P4 L6+L20 | 0.263 | 0.566 ✗ | 0.022 | 0.286 | +0.004 | 0.281 | 0.45% / 27.2% | 0.447 | 18.4% | 24.0% |
| P5 L8+L20 | 0.262 | 0.584 ✗ | **0.0075** | **0.258** | +0.010 | 0.297 | 0.5% / 24.7% | **0.425** | 22.0% ✗ | 23.2% |
| P6 L7‖L20 concat | 0.275 | 0.596 ✗ | 0.048 | 0.347 ✗ | −0.072 | 0.257 | 0.9% / 29.2% | 0.720 ✗ | 19.6% | 23.3% |
| P7 all-layer stack | 0.281 | 0.559 ✗ | 0.055 | 0.393 ✗ | −0.101 | 0.299 | 1.05% / 31.8% | 0.557 ✗ | 21.5% ✗ | 22.7% |

**Perturbation cohort** (250 real + 250 fake In-the-Wild clips):
- mp3, speed and shift moved every candidate by little.
- Noise is the axis that separates them. Pairs anchored on L7 get worse; L6 and L8 anchors get better.

**Other readouts:**
- **Short In-the-Wild clips (≤ 4 s):** improve for every pair, 0.400 → 0.29–0.33.
- **Test-set flips at the inner threshold vs P0:** 32–56 of 1,671 files.

**Rules failed:**

| Candidate | Rules failed |
|---|---|
| P1, P2, P3 | R1 (fold 4), R4 (noise) |
| P4 | R1 (fold 4) |
| P5 | R1 (fold 4), R5 (MLAAD 22.0% vs 17.3% + 3) |
| P6, P7 | R1, R3, R4; P7 also R5 |

**Neighbour check:** 0 of 4 neighbours confirm.

## Fusion level

The candidate replaces M1b in the shipped rule; it is reported, not adopted.

| | Inner | Holdout | ITW | ITW gain, boot p5 | Inner averse |
|---|---|---|---|---|---|
| F0 (shipped) | 0.135 | 0.0065 | 0.228 | | 0.300 |
| F(P1) | 0.162 | 0.0010 | 0.206 | −0.003 | 0.369 |
| F(P3) | 0.162 | 0.0020 | 0.184 | +0.025 | 0.361 |
| F(P5) | 0.162 | 0.0010 | 0.185 | +0.003 | 0.400 |
| F(P7) | 0.149 | 0.0075 | 0.276 | −0.075 | 0.281 |

Inside the full fusion, every top-layer pair improves holdout and In-the-Wild but costs about 0.026 on inner, under both costs. That cost is the ElevenLabs fold again.

## What it means

1. **There is a trade-off, not a free lunch.**
   - Adding a top XLS-R layer to L7 helps on wild audio (real speech on unseen channels) and on short clips.
   - It hurts on the held-out commercial generator (ElevenLabs: 0.464 → 0.54–0.58) and, for L7-anchored pairs, under noise.
   - Commercial generators are the ones the NSA kickoff named, so R1 caught a cost that matters.
   - **Hypothesis, not tested:** the top layers encode content and speaking style. Wild real speech is distinctive there, but ElevenLabs' prosody is good enough to pass. The mid layers keep the acoustic artifacts that catch it.
2. **How you combine matters as much as what you combine.**
   - The simple equal-weight rank blend generalized.
   - Learned combinations overfit the inner folds and lost on wild audio: concatenating the layers into one classifier (P6), and a stacker over all 25 layers (P7).
   - This matches v1's fusion experience.
3. **The L6 and L8 anchors behaved differently from L7 under noise** (0.447 and 0.425 vs 0.485), with P5 the best on holdout and wild. One cohort of 500 clips is too small to build on, but it is a lead for V2-20.
4. **The holdout gains are large but untrustworthy as a guide.** P5 reaches 0.0075 vs 0.072. The holdout keeps over-rewarding whatever helps on playht and wavegrad2 (see `docs/learn/v1-explained.md`, lesson 4).

## Next

These are candidates for a new pre-declared run; none is a re-test of this one.
- **ElevenLabs is the binding constraint.** More commercial-generator fakes for evaluation (V2-30) would tell whether the fold-4 loss is general. MLAAD has only 12 ElevenLabs-family clips in this cohort, too few to read.
- **A gated or weighted use of top layers, belonging to V2-60.** For example, the top layer contributes only where the clip looks wild (noise floor, bandwidth), so that clean commercial TTS is judged by the mid layers.
- **Noise-augmented probe training (V2-31).** Every candidate's noise cost is 0.43–0.72, which dwarfs the differences between candidates.
