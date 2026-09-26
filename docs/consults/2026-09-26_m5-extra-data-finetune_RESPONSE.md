# Response: M5 extra-data mix and fine-tune recipe

Prompt: `2026-09-26_m5-extra-data-finetune_CONSULTATION.md`. Target: a frontier-LLM panel (VeriLM memo synthesizing two experts, "Claude" and "Gemini"). Verbatim answer: `2026-09-26_m5-extra-data-finetune_RESPONSE_verbatim.md` (765 lines, received ~03:20 EDT).

## Round 1: clarifying questions (~02:40 EDT)

The source first asked five clarifying questions. Our answers, from measured facts:
- **Ablation budget:** ~6 GPU-hours total; 6 final models fixed; ~2 h of ablation across all four questions; rank by EV, give the top 2–3.
- **M1 bar:** known in time (it is: holdout minDCF 0.1458). M5 replaces M1 only if it beats it; its scores feed the stacker either way.
- **Ratios:** first guesses, open to revision.
- **M-AILABS bona fide:** off the table (both hosts unreachable at 02:36 EDT).
- **Diagnostics:** the handcrafted-feature + LightGBM tooling is available on the Mac's CPU; frozen XLS-R embeddings only on the rented box.

## Round 1: the answer, in brief

**Verdict:** "Over-engineered on data, proportionate on recipe, and under-invested in the quantity the metric actually measures." Both experts derived the same cost weight (normalized DCF = P_miss + 9.33·P_FA; at the 70/30 split, one false alarm costs four misses) and inverted our plan: **real-side diversity beats spoof volume.**

| Question | Recommendation |
|---|---|
| **MLAAD (spoof only)** | Split: Gemini drops it; Claude keeps it **gated** at ~28% of the spoof side, ≤ 120 clips per model, behind a CPU corpus probe with an explicit drop rule. The memo sides with the gated position: MLAAD clips are vocoder output and carry a resample/rolloff cue and an audiobook register, not a microphone chain; LibriSpeech (LibriVox, labelled bona fide in the same batches) contradicts the register cue; band-limit augmentation attacks the rolloff cue. Probe A: LightGBM on handcrafted features, MLAAD spoof vs DiffSSD spoof (label held constant), on augmented copies; project the real pools onto that axis. AUC > 0.95 with LibriSpeech landing on the MLAAD side → fall back to ~12%; < 0.85 → no exploitable fingerprint. Probe C: score both arms on **In-the-Wild bona fide only** (read-out is allowed; training is not) at the inner-fold threshold; drop MLAAD if ΔP_FA > +1 point. |
| **ASVspoof 2019** | **All ~5.1k bona fide: the highest-EV item in the plan** (a third real recording chain pushes the model toward chain-invariant bona fide; the trade-off risk is a miss, cost 1). Correction: train+dev is ~40 VCTK speakers, not ~100. **Spoof: 45k is wrong; ~2.5k stratified across A01–A06 as a channel anchor** (so VCTK isn't real-only); zero is defensible. |
| **Mixing** | Label-balanced 50/50. Real side 25/35/40 LJ / LibriSpeech / ASV19 (LJ capped below its data share: one speaker). Spoof side 62/28/10 DiffSSD / MLAAD / ASV19. **Kill the +50k DiffSSD spoof** (no new generators on a generator-grouped validation); cap DiffSSD at ~2k per inner generator. Pool ≈ 48k at 49% real vs our add-everything 145k at 16% real. **No curriculum, no taper**, constant mix. Sampling ratios instead of loss weights; a bona fide class weight of ~2.0 is a "maybe". |
| **Recipe** | Truncate to **L = 12** and fully fine-tune the retained layers (CNN frozen); fallback L = 14–16 if learned layer weights concentrate at the top. **Learned softmax-weighted layer sum** + **attentive statistics pooling** (mean + std, 128-d bottleneck). Head LR 5e-4 (no weight decay); backbone 1e-5 at the top layer with layer-wise decay γ = 0.85, weight decay 0.01 (not biases/LayerNorm); 8% warmup then cosine; bf16; grad clip 1.0. ≤ 4 epochs, validate every half epoch, **select on minDCF at π = 0.3, never on loss or EER.** Augmentation p = 0.65 with the existing codec/noise/band-limit/RIR set, band-limit upweighted, plus only RawBoost's convolutive (linear + nonlinear) algorithm at p = 0.25. **Identical augmentation distribution across classes and pools; log realized rates per source × class.** BCE with label smoothing 0.05. **No OC-softmax** (a narrow bona fide class makes it manufacture weight-4 errors), no AASIST. |
| **Unnamed failure modes** | (1) **Score–duration coupling** under a single global threshold: check Spearman(score, log duration) on out-of-fold bona fide; correct with a linear trend fit on OOF bona fide if needed. (2) Augmentation–label leakage (the highest-probability silent bug). (3) **Stacker miscalibration from LJ's single speaker**: LJ's OOF bona fide scores are optimistic; fit the stacker on LibriSpeech/VCTK OOF bona fide or treat LJ as weakly grouped. (4) Trim-boundary and crest-factor signatures: jitter the trim threshold (30–40 dB) and retain 0–200 ms at random. |
| **The gate itself** | On a holdout with 2 generators and 26 bona fide groups, σ(minDCF) ≈ 0.07–0.10 (97% from the P_FA term). **A holdout gap under ~0.15 is not evidence.** Decide on 5-fold OOF minDCF and the holdout together; if they disagree by < 0.15, keep M1 primary and route M5 through the stacker. |
| **Cuts** | +50k DiffSSD spoof; ASV19 spoof 45k → ~2.5k; curriculum/taper; OC-softmax; AASIST; non-convolutive RawBoost. Reallocate to the MLAAD probe, one mix A/B, and the CPU audits. "Run few arms, pre-declare the criterion, and stop." |

## Synthesis: what we act on

**Adopt (config or small code):**
1. Extra DiffSSD spoof capped so each inner generator has ~2k total (1k sample + 1k extra), not 4k.
2. All extra NSA bona fide kept (LJ + LibriSpeech, minus holdout groups).
3. ASV19: all bona fide; spoof 2.5k stratified across A01–A06 (then the ≥ 2.0 s cut and duration matching from the plan, applied to both classes).
4. MLAAD: 28% of the spoof side, ≤ 120 clips per model; **Probe A** runs on the Mac's CPU before the finals (handcrafted features already exist for the pipeline); Probe C runs on In-the-Wild bona fide at assembly. Drop rule as stated.
5. Batch mix: real 25/35/40, spoof 62/28/10, constant.
6. Model: L = 12 (fallback 16), all retained layers trainable, learned layer-weighted sum, attentive statistics pooling (mean + std), BCE + label smoothing 0.05, head 5e-4 / backbone 1e-5 with LLRD 0.85, weight decay 0.01, 8% warmup + cosine, bf16, clip 1.0.
7. Augmentation p = 0.65; add convolutive RawBoost at p = 0.25; add trim-threshold jitter (30–40 dB) and 0–200 ms retained edges as a training-only op; log realized augmentation rates per source × class.
8. Readouts: Spearman(score, log duration) on OOF bona fide; pooled 5-fold OOF minDCF next to the holdout; gate stated as "both".

**Keep from the reviewed plan despite the memo:**
- Selection stays wall-clock-fixed for the fold models (the memo's "select on minDCF every half epoch" would select on single-generator folds 3 and 4; Codex/critique flagged that). The pilot and ablation curves are reported.
- Codec ops stay precomputed for a seeded 30% subset of **all** training clips regardless of class (that satisfies the symmetric-augmentation rule; test D6 measures it).
- "Random peak normalization before ZMUV" is a no-op after ZMUV (pure gain); our gain + hard-clip op covers the crest-factor axis instead.
- Frozen-backbone control arm: not run; M1 v2 already is that control (holdout 0.1458 with OOF scores).

**Pass to the main chat:** the stacker caveat about LJ's single-speaker OOF bias, and the "holdout gap under 0.15 is noise; decide on OOF + holdout" rule for the M5-vs-M1 decision.

## Follow-ups

None planned; the ablation (NSA-only vs the recommended mix) is the empirical check.
