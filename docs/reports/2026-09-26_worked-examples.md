# Worked examples: eight real test files through the shipped pipeline

Sat Sep 26, 2026, README chat. Brief: `docs/handoffs/2026-09-26_readme-orchestration-handoff.md`. Purpose: show that per-file routing and explanation are real, on real NSA test files, with the exact records the runner writes.

_Update, Sat ~12:30: the shipped fusion rule moved at ~12:05 from `E_on_A_alpha0.2` (fusion_v1) to `A3_w0.2_E` (fusion_v2: 0.6 M1b v3 + 0.2 handcrafted v5 + 0.2 M5 in rank space, same M3 suppression; the default switch is pending Nathan's word). The eight records below were produced under fusion_v1 and are kept as written. The addendum at the end re-runs the same eight files under fusion_v2; no file changes side of 0.5._

**Where the records come from.** The eight files were chosen from `outputs/inventory/test_worked_candidates.csv` (a join of the shipped TSV, the 06:02 z-mean TSV and the runner's per-file features over all 1,671 test files) to cover the situations the brief asks for: a confident real, a confident fake, a file where M3 suppressed a false alarm, a file with a hum and one with a seam flagged as evidence, a low-voiced-fraction file near the gate, and two disagreement cases. The records below were produced by running the shipped pipeline on these eight files with the frozen rule:

```bash
uv run python scripts/run_pipeline.py --in <dir with the eight files> --out <out> --rule e_on_a --team HEARSAY --threads 4 --no-preflight
```

Runner probabilities agree with the submitted TSV (`submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`) to four decimals on all eight files. The full-run records under `outputs/runner/full_20260926/results/` carry identical detector sentences; only their fusion line differs, because that run predates the frozen rule.

**How the fused probability is computed** (`models/fusion_v1/constants.json`). Each detector's logit is placed against its own inner out-of-fold distribution: rank = share of inner rows scoring below it. Base = 0.8 · rank(M1b v3) + 0.2 · rank(handcrafted v5). If Spectra-AASIST's margin is below −3 and base is above 0.5, base is halved (M3 suppression; M3 never raises a score). Then p = 0.001 + 0.999 · sigmoid(17.816 · base − 8.302 − 0.847), the Platt map at the 0.3 prior followed by the determinate map into [0.001, 1]. "z-mean p" is what the same file scored under the rejected 06:02 equal-weight rule (`submissions/20260926-0602_M4_fusion_zmean_m1bv3_hcv5_m3.tsv`), shown for contrast.

Score conventions: 1.0 = synthetic. M1b reports a calibrated log-likelihood ratio (LLR; positive = synthetic). Spectra reports its spoof-minus-bonafide margin. The handcrafted detector reports a probability from its tree model. "SD" in an evidence sentence is standard deviations from the real-speech mean on the inner rows; each sentence names that detector's three largest contributions.

| File | Duration | Shipped p | z-mean p | M1b LLR (rank) | Handcrafted P (rank) | M3 margin | Base | M3 suppression | Situation |
|---|---|---|---|---|---|---|---|---|---|
| HGT1013455.wav | 4.74 s | 0.002 | 0.001 | −6.30 (0.044) | 0.21 (0.556) | −7.07 | 0.146 | no | confident real |
| HGT1310023.wav | 3.41 s | 1.000 | 1.000 | +13.85 (0.940) | 1.00 (0.958) | +10.91 | 0.944 | no | confident fake |
| HGT3739624.wav | 3.92 s | 0.031 | 0.817 | +1.91 (0.609) | 0.96 (0.752) | −6.63 | 0.638 → 0.319 | **yes** | M3 suppressed a likely false alarm |
| HGT1046947.wav | 7.15 s | 0.003 | 0.686 | −5.78 (0.061) | 0.16 (0.541) | +9.50 | 0.157 | no (M3 never promotes) | deep detectors disagree; drift flagged |
| HGT3237868.wav | 3.25 s | 0.935 | 0.998 | +6.05 (0.733) | 0.01 (0.384) | +11.38 | 0.663 | no | fake with a stable mains hum |
| HGT1794158.wav | 3.16 s | 0.005 | 0.134 | −5.79 (0.061) | 0.97 (0.769) | −4.72 | 0.202 | no (base below 0.5) | real with editing seams |
| HGT2080120.wav | 4.74 s | 0.964 | 0.999 | +6.30 (0.738) | 0.15 (0.538) | +7.64 | 0.698 | no | 21% voiced, near the gate |
| HGT2305393.wav | 4.39 s | 0.471 | 0.995 | −1.04 (0.453) | 0.92 (0.723) | +6.38 | 0.507 | no (M3 never promotes) | genuinely uncertain; click flagged |

Every file: container `wav/pcm_s16le 16000 Hz mono 16-bit, 1 tag(s) [encoder=Lavf58.29.100]: no class evidence in the container; written by FFmpeg (libavformat), so the file was re-muxed or transcoded at least once` (score 0.5, routing). The routing log's first line is therefore the same for all eight and is shown once here: `container: PCM WAV, not lossy, FFmpeg-written; compression forensics run for evidence, not score`.

---

## 1. HGT1013455.wav: confident real (p = 0.002)

Routing log:
```
compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
enf: no mains hum; no environment evidence either way
splice: no editing seams
speaker_drift: one consistent voice (min window similarity 0.09); evidence only, never fused
speech_gate: is_speech=true (voiced 74% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.146; M3 margin -7.07, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.002): XLS-R layer-7 probe: calibrated log-likelihood ratio −6.30 (real-like, P=0.00)
- **handcrafted** (fused, 0.209): spectral/prosody features real-like (P=0.21): spectral contrast, band 0 1.1 SD below real speech (toward synthetic); CQCC 1 frame-to-frame change 2.6 SD above real speech (toward synthetic); MFCC 13 variability 1.7 SD above real speech (toward synthetic)
- **spectra_aasist** (suppressor, 0.001): Spectra-AASIST: spoof-minus-bonafide margin −7.07 (real-like, P=0.00)
- **compression** (evidence, 0.968): compression-trace features synthetic-like (P=0.97): whole-clip floor depth 1.7 SD below real speech (toward synthetic); 3-7 kHz valley depth 0.7 SD below real speech (toward synthetic); deep-hole flicker frame to frame 1.7 SD above real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 60 Hz candidate: median SNR 2.8 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 28 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.09 (mean 0.41) over 9 windows
- **speech_gate** (gate, 0.5): speech present: voiced 74% of frames, pitch spread 0.32, loudness std 10.6 dB; detectors apply

Why the system said what it said. All three model-based inputs agree the voice is real, and the probe, which carries 80% of the score, is far into its real range (rank 0.044 among inner rows). The handcrafted detector's rank of 0.556 is above its median even though its probability is 0.21: that is the test set's darker spectral envelope pushing every test file up its scale, the residual shift described in `docs/reports/2026-09-26_handcrafted-v4.md`, and at 20% weight it adds only 0.11 to the base. The compression detector reads the file as synthetic-like (0.97) on floor depth and hole flicker, which on this test set is the signature of the sponsor's codec stage rather than of synthesis (72% of the test set reads as laundered); it is evidence only and moves nothing. Nothing else fires. The result is a calibrated 0.002, not a saturated 0.

## 2. HGT1310023.wav: confident fake (p = 1.000)

Routing log:
```
compression: effective bandwidth 7250 Hz, at or below the 7.25 kHz test-set band match
enf: no mains hum; no environment evidence either way
splice: no editing seams
speaker_drift: one consistent voice (min window similarity 0.27); evidence only, never fused
speech_gate: is_speech=true (voiced 81% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.944; M3 margin +10.91, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 1.0): XLS-R layer-7 probe: calibrated log-likelihood ratio +13.85 (synthetic-like, P=1.00)
- **handcrafted** (fused, 0.999): spectral/prosody features synthetic-like (P=1.00): LFCC 1 variability 0.2 SD above real speech (toward synthetic); LFCC 16 frame-to-frame change 1.6 SD below real speech (toward synthetic); spectral contrast, band 4 2.0 SD below real speech (toward synthetic)
- **spectra_aasist** (suppressor, 1.0): Spectra-AASIST: spoof-minus-bonafide margin +10.91 (synthetic-like, P=1.00)
- **compression** (evidence, 0.941): compression-trace features synthetic-like (P=0.94): whole-clip floor depth 1.0 SD below real speech (toward synthetic); 3-7 kHz valley depth 0.7 SD below real speech (toward synthetic); effective bandwidth 0.5 SD below real speech (toward real)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 50 Hz candidate: median SNR 3.9 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 27 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.27 (mean 0.50) over 6 windows
- **speech_gate** (gate, 0.5): speech present: voiced 81% of frames, pitch spread 0.27, loudness std 9.3 dB; detectors apply

Why. The probe's LLR of +13.85 is as far into the synthetic range as the inner rows go (rank 0.94), the handcrafted detector agrees on smoothed high-order LFCC dynamics (the "too regular frame to frame" cue that the added feature families made visible), and Spectra agrees with a margin of +10.9. M3 cannot raise a score and does not need to. The bandwidth sits exactly at the 7.25 kHz band match, which is the common case for this test set (981 of 1,671 files) and carries no class information after band matching.

## 3. HGT3739624.wav: M3 suppressed a likely false alarm (p = 0.031; z-mean 0.817)

Routing log:
```
compression: effective bandwidth 7250 Hz, at or below the 7.25 kHz test-set band match
enf: no mains hum; no environment evidence either way
splice: no editing seams
speaker_drift: one consistent voice (min window similarity 0.14); evidence only, never fused
speech_gate: is_speech=true (voiced 80% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.638; M3 false-alarm suppression applied (margin -6.63 < -3 and base > 0.5: x0.5 -> 0.319)
```

Evidence:
- **m1b_v3** (fused, 0.871): XLS-R layer-7 probe: calibrated log-likelihood ratio +1.91 (synthetic-like, P=0.87)
- **handcrafted** (fused, 0.957): spectral/prosody features synthetic-like (P=0.96): spectral contrast, band 0 1.0 SD below real speech (toward synthetic); LFCC 17 frame-to-frame change 1.4 SD below real speech (toward synthetic); CQCC 1 frame-to-frame change 3.2 SD above real speech (toward synthetic)
- **spectra_aasist** (suppressor, 0.001): Spectra-AASIST: spoof-minus-bonafide margin −6.63 (real-like, P=0.00)
- **compression** (evidence, 0.94): compression-trace features synthetic-like (P=0.94): local-hole runs per frame 0.1 SD below real speech (toward synthetic); 3-7 kHz valley depth 0.5 SD below real speech (toward synthetic); whole-clip floor depth 1.1 SD below real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 60 Hz candidate: median SNR 4.5 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 21 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.14 (mean 0.48) over 7 windows
- **speech_gate** (gate, 0.5): speech present: voiced 80% of frames, pitch spread 0.22, loudness std 10.4 dB; detectors apply

Why. This is the one rule that changed a decision on this page. Both fused columns lean synthetic, but the probe only weakly: an LLR of +1.9 is near its boundary (rank 0.61), and the handcrafted detector's 0.96 rests on the same envelope cues that over-call the whole test set. The blend lands at 0.638, above 0.5. Spectra is strongly convinced the voice is real (margin −6.6, well below the −3 threshold), so the suppression rule halves the base to 0.319 and the Platt map turns that into 0.031. Under the 06:02 equal-weight rule, where Spectra was one vote in three, the same file scored 0.817. We do not know this file's label. What we know from labeled data is that this rule touched 123 holdout files and 34 In-the-Wild files and every one of them was real (`outputs/fusion/orchestration_ablation.md`), and that if it is wrong here the cost is one miss (weight 1), never a false alarm (weight 9.33). That asymmetry is why M3 has this role and no other.

## 4. HGT1046947.wav: the deep detectors disagree; drift flagged (p = 0.003; z-mean 0.686)

Routing log:
```
compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
enf: no mains hum; no environment evidence either way
splice: no editing seams
speaker_drift: voice drifts (min window similarity -0.10); evidence only, never fused
speech_gate: is_speech=true (voiced 28% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.157; M3 margin +9.50, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.003): XLS-R layer-7 probe: calibrated log-likelihood ratio −5.78 (real-like, P=0.00)
- **handcrafted** (fused, 0.156): spectral/prosody features real-like (P=0.16): LFCC 16 frame-to-frame change 3.0 SD below real speech (toward synthetic); spectral contrast, band 0 0.8 SD above real speech (toward real); LFCC 1 variability 1.1 SD below real speech (toward real)
- **spectra_aasist** (suppressor, 1.0): Spectra-AASIST: spoof-minus-bonafide margin +9.50 (synthetic-like, P=1.00)
- **compression** (evidence, 0.908): compression-trace features synthetic-like (P=0.91): 3-7 kHz valley depth 0.9 SD below real speech (toward synthetic); 0.3-3 kHz floor depth 4.2 SD below real speech (toward real); whole-clip floor depth 1.2 SD below real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 50 Hz candidate: median SNR 6.0 dB, above 12 dB in 36% of frames): no environment evidence either way
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 34 dB, informational)
- **speaker_drift** (evidence, 0.6): voice drifts across the clip: minimum window-to-window speaker similarity −0.10 (mean 0.37) over 14 windows; consistent with a second voice or an unstable cloned identity
- **speech_gate** (gate, 0.5): speech present: voiced 28% of frames, pitch spread 0.34, loudness std 18.5 dB; detectors apply

Why. The two deep detectors are at opposite extremes: the probe says real at −5.8, Spectra says fake at +9.5. This is one of the 154 test files on which they disagree (`docs/reports/2026-09-26_m3-spectra.md`). The shipped rule sides with the probe, because Spectra is never allowed to raise a score, and the handcrafted detector leans real as well, so the base is 0.157 and the probability 0.003; under the equal-weight rule Spectra's vote carried the file to 0.686. The evidence detectors add context without changing the number: over 7 s with only 28% of frames voiced and loudness swinging 18.5 dB, the ECAPA windows disagree with each other (minimum cosine −0.10 over 14 windows), which is what a real recording with long pauses, a second voice or a moving microphone looks like; there is also a weak 50 Hz component that clears the 12 dB bar in 36% of frames, below the 60% needed to call it hum. Drift is evidence only because on labeled data the fakes are the most self-consistent voices (`docs/reports/2026-09-26_gate-and-drift.md`). The routing log records the disagreement instead of hiding it.

## 5. HGT3237868.wav: a fake with a stable mains hum (p = 0.935)

Routing log:
```
compression: effective bandwidth 7250 Hz, at or below the 7.25 kHz test-set band match
enf: mains hum present, stable (recording-environment evidence)
splice: no editing seams
speaker_drift: one consistent voice (min window similarity 0.53); evidence only, never fused
speech_gate: is_speech=true (voiced 94% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.663; M3 margin +11.38, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.998): XLS-R layer-7 probe: calibrated log-likelihood ratio +6.05 (synthetic-like, P=1.00)
- **handcrafted** (fused, 0.012): spectral/prosody features real-like (P=0.01): LFCC 1 variability 1.4 SD below real speech (toward real); MFCC 9 mean 1.5 SD above real speech (toward real); MFCC 10 mean 2.3 SD above real speech (toward real)
- **spectra_aasist** (suppressor, 1.0): Spectra-AASIST: spoof-minus-bonafide margin +11.38 (synthetic-like, P=1.00)
- **compression** (evidence, 0.569): compression-trace features synthetic-like (P=0.57): 6-7 kHz level vs 1-3 kHz 0.4 SD below real speech (toward real); deep-hole persistence frame to frame 0.9 SD below real speech (toward real); 3-7 kHz valley depth 0.5 SD below real speech (toward synthetic)
- **enf** (evidence, 0.4): stable 49.94 Hz mains hum (SNR 14.0 dB in 100% of frames, drift 61 mHz): consistent with a genuine electrical recording environment
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 21 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.53 (mean 0.72) over 6 windows
- **speech_gate** (gate, 0.5): speech present: voiced 94% of frames, pitch spread 0.22, loudness std 8.5 dB; detectors apply

Why. Both deep detectors call this synthetic with large margins, and the ENF detector finds a clean, stable 49.94 Hz hum in every frame, the kind of thing a genuine mains-powered recording chain leaves behind. On its own the hum would argue real, and the evidence sentence says so. It stays evidence because on the training data hum is a property of the corpus, not the class: 36% of LJ Speech carries it and grad_tts, trained on LJ, reproduces a 50/60 Hz component in 19% of its clips (`docs/reports/2026-09-26_cpu-detectors.md`); a learned ENF column would have taught the system "hum = real". The handcrafted detector also reads the file as real (0.01) on cepstral means. What decides it is the probe at +6.05 (rank 0.73); the handcrafted column's 20% pulls the base down to 0.663 and Spectra, agreeing on fake, cannot suppress. Note the drift statistic: a minimum cosine of 0.53 makes this the most self-consistent voice of the eight, which is the direction the drift report found for fakes. The routing log shows a hum on a file called fake rather than smoothing it over; that is the honest reading.

## 6. HGT1794158.wav: real with two editing seams (p = 0.005; z-mean 0.134)

Routing log:
```
compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
enf: no mains hum; no environment evidence either way
splice: 2 editing seam(s)
speaker_drift: one consistent voice (min window similarity 0.40); evidence only, never fused
speech_gate: is_speech=true (voiced 90% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.202; M3 margin -4.72, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.003): XLS-R layer-7 probe: calibrated log-likelihood ratio −5.79 (real-like, P=0.00)
- **handcrafted** (fused, 0.97): spectral/prosody features synthetic-like (P=0.97): spectral contrast, band 0 1.2 SD below real speech (toward synthetic); CQCC 1 frame-to-frame change 3.4 SD above real speech (toward synthetic); LFCC 1 variability 0.4 SD above real speech (toward synthetic)
- **spectra_aasist** (suppressor, 0.009): Spectra-AASIST: spoof-minus-bonafide margin −4.72 (real-like, P=0.01)
- **compression** (evidence, 0.98): compression-trace features synthetic-like (P=0.98): whole-clip floor depth 1.1 SD below real speech (toward synthetic); 0.3-3 kHz floor depth 1.3 SD above real speech (toward synthetic); 3-7 kHz valley depth 0.6 SD below real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 50 Hz candidate: median SNR 3.7 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.6): editing seam at 2.30 s: 2 DC-offset jump(s) (largest 7.2% of full scale); consistent with a cut or inserted segment (background-floor range 26 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.40 (mean 0.64) over 6 windows
- **speech_gate** (gate, 0.5): speech present: voiced 90% of frames, pitch spread 0.31, loudness std 9.0 dB; detectors apply

Why. The splice detector found two DC-offset jumps at 2.30 s, the largest 7.2% of full scale, and reports them as a possible cut. That is evidence of editing, not of synthesis: the sponsor said no clip is partially synthetic, and on the training data DC jumps appear in 9% of LibriSpeech real clips as well as in several generators' output. The handcrafted detector says synthetic (0.97), on the same band-0 contrast and CQCC-change cues that mark the test set's envelope shift, but at 20% weight it lifts the base only to 0.202. Both deep detectors say real. Spectra's margin is below −3, but the suppression rule needs a base above 0.5 and does not fire; it is not needed. The shipped 0.005 against the equal-weight 0.134 shows the handcrafted column's over-calling costing less under the rank blend than under a z-score mean.

## 7. HGT2080120.wav: 21% voiced, near the gate (p = 0.964)

Routing log:
```
compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
enf: no mains hum; no environment evidence either way
splice: no editing seams
speaker_drift: one consistent voice (min window similarity 0.99); evidence only, never fused
speech_gate: is_speech=true (voiced 21% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.698; M3 margin +7.64, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.998): XLS-R layer-7 probe: calibrated log-likelihood ratio +6.30 (synthetic-like, P=1.00)
- **handcrafted** (fused, 0.146): spectral/prosody features real-like (P=0.15): spectral contrast, band 0 1.7 SD above real speech (toward real); spectral flux variability 2.0 SD below real speech (toward real); LFCC 16 frame-to-frame change 1.9 SD below real speech (toward synthetic)
- **spectra_aasist** (suppressor, 1.0): Spectra-AASIST: spoof-minus-bonafide margin +7.64 (synthetic-like, P=1.00)
- **compression** (evidence, 0.967): compression-trace features synthetic-like (P=0.97): whole-clip floor depth 0.9 SD below real speech (toward synthetic); 3-7 kHz valley depth 0.7 SD below real speech (toward synthetic); deep-hole runs per frame 1.6 SD above real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 60 Hz candidate: median SNR 3.9 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.5): no clicks or DC-offset jumps: no sign of a cut or inserted segment (background-floor range 97 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.99 (mean 0.99) over 2 windows
- **speech_gate** (gate, 0.5): speech present: voiced 21% of frames, pitch spread 0.13, loudness std 13.4 dB; detectors apply

Why. Only 21% of frames are voiced (the test set's minimum is 9.9%; the gate's bar is 5%), the pitch spread of 0.13 is low, and the 97 dB background-floor range says the clip is mostly near-silence with a short burst of speech: after trimming, the drift detector gets only two 1 s windows and reports nothing useful (0.99). All five gate cues clear, so the detectors apply and the file is judged, not abstained on; the gate's thresholds were set beyond the test set's extremes for exactly this reason (`docs/reports/2026-09-26_gate-and-drift.md`). Both deep detectors call it synthetic with wide margins; the handcrafted detector says real on band-0 contrast and low flux variability, and pulls the base from 0.74 to 0.70. Had the gate tripped, the file would sit in the pinned block below every scored file and the routing log would say so.

## 8. HGT2305393.wav: genuinely uncertain, and it says so (p = 0.471; z-mean 0.995)

Routing log:
```
compression: effective bandwidth 7500 Hz, above the 7.25 kHz band match
enf: no mains hum; no environment evidence either way
splice: 1 editing seam(s)
speaker_drift: one consistent voice (min window similarity 0.28); evidence only, never fused
speech_gate: is_speech=true (voiced 74% of frames); default-answer policy not applied
fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) = 0.507; M3 margin +6.38, no suppression (M3 never promotes)
```

Evidence:
- **m1b_v3** (fused, 0.26): XLS-R layer-7 probe: calibrated log-likelihood ratio −1.04 (real-like, P=0.26)
- **handcrafted** (fused, 0.918): spectral/prosody features synthetic-like (P=0.92): spectral contrast, band 0 1.0 SD below real speech (toward synthetic); spectral flux variability 4.2 SD above real speech (toward synthetic); CQCC 2 frame-to-frame change 3.6 SD above real speech (toward synthetic)
- **spectra_aasist** (suppressor, 0.998): Spectra-AASIST: spoof-minus-bonafide margin +6.38 (synthetic-like, P=1.00)
- **compression** (evidence, 0.983): compression-trace features synthetic-like (P=0.98): deep-hole flicker frame to frame 2.2 SD above real speech (toward synthetic); 3-7 kHz valley depth 0.8 SD below real speech (toward synthetic); whole-clip floor depth 1.5 SD below real speech (toward synthetic)
- **enf** (evidence, 0.5): no mains hum at 50 or 60 Hz (best 60 Hz candidate: median SNR 3.3 dB, above 12 dB in 0% of frames): no environment evidence either way
- **splice** (evidence, 0.6): editing seam at 1.59 s: 1 click(s) (largest step 17x the surrounding waveform); consistent with a cut or inserted segment (background-floor range 26 dB, informational)
- **speaker_drift** (evidence, 0.5): one consistent voice: minimum window-to-window speaker similarity 0.28 (mean 0.50) over 8 windows
- **speech_gate** (gate, 0.5): speech present: voiced 74% of frames, pitch spread 0.30, loudness std 10.7 dB; detectors apply

Why. The probe is slightly on the real side (LLR −1.0, rank 0.45), the handcrafted detector is on the synthetic side (rank 0.72), and the blend lands at 0.507, almost exactly on the boundary; the Platt map returns 0.471. Spectra is confident the file is fake, but the rule that lets it lower a score does not let it raise one, so its vote is recorded and not counted; under the equal-weight rule the same file scored 0.995. A single-sample click at 1.59 s, 17 times the surrounding steps, is flagged as a possible seam. On the training data such clicks are a vocoder artifact (27% of wavegrad2 clips, 14% of diffgan_tts), so the observation leans fake, but it is evidence, not score, because seams also occur in real recordings and a two-valued score cannot be validated on labels. The sponsor's kickoff asked for calibrated probabilities and never for confident wrongness on a subpopulation (`docs/reports/2026-09-25_sponsor-questions.md`); a 0.47 with both sides of the disagreement written down is what that looks like.

---

## What the eight files show, in one place

- Two rules can change a score, and each is visible in the log when it does: M3 suppression fired on file 3 and on no other; the gate fired on none (0 of 1,671 test files, as designed).
- Six detectors contributed evidence without touching the score, and the log says so each time ("evidence only, never fused"): hum on file 5, seams on files 6 and 8, drift on file 4, codec traces on all eight.
- The rejected equal-weight rule would have moved four of the eight files across 0.5 (files 3, 4, 6 in one direction of risk, file 8 in the other), all by letting a detector with undisclosed training data vote in full.
- Disagreements are shown, not hidden: files 4 and 8 carry a deep-detector split in their records, with the fused probability reflecting the rule, not a tie-break.

---

## Addendum (Sat ~12:30): the same eight files under fusion_v2 (`A3_w0.2_E`)

Re-run with the same command plus `--fusion models/fusion_v2/constants.json`, which adds the M5 head (`models/m5_xlsr_ft_20260926-0741`) as a third weighted scorer: base = 0.6 · rank(M1b v3) + 0.2 · rank(handcrafted v5) + 0.2 · rank(M5), then the same M3 suppression and the same Platt and determinate maps. Live probabilities agree with the logged candidate TSV (`submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv`) to 5.4e-5 or better on all eight files. Every detector sentence above is unchanged; the new lines per file are M5's evidence and the fusion line.

| File | p under fusion_v1 | p under fusion_v2 | M5 logit (P) | Base under v2 | M3 suppression | M5's evidence sentence |
|---|---|---|---|---|---|---|
| HGT1013455.wav | 0.0024 | 0.0028 | −5.26 (0.01) | 0.151 | no | XLS-R fine-tuned head (M5, 12 layers, attentive pooling): logit −5.26 (real-like, P=0.01) |
| HGT1310023.wav | 0.9995 | 0.9994 | +5.73 (1.00) | 0.931 | no | logit +5.73 (synthetic-like, P=1.00) |
| HGT3739624.wav | 0.0312 | 0.0305 | −0.78 (0.31) | 0.622 → 0.311 | **yes** | logit −0.78 (real-like, P=0.31) |
| HGT1046947.wav | 0.0027 | 0.0348 | +5.64 (1.00) | 0.319 | no (M3 never promotes) | logit +5.64 (synthetic-like, P=1.00) |
| HGT3237868.wav | 0.9347 | 0.8973 | +0.14 (0.54) | 0.631 | no | logit +0.14 (synthetic-like, P=0.54) |
| HGT1794158.wav | 0.0049 | 0.0097 | −3.59 (0.03) | 0.241 | no | logit −3.59 (real-like, P=0.03) |
| HGT2080120.wav | 0.9640 | 0.9384 | −0.13 (0.47) | 0.663 | no | logit −0.13 (real-like, P=0.47) |
| HGT2305393.wav | 0.4714 | 0.4656 | −2.23 (0.10) | 0.501 | no (M3 never promotes) | logit −2.23 (real-like, P=0.10) |

The fusion line under v2 reads, for example (file 3): `fusion: e_on_a (A3_w0.2_E): 0.6 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) + 0.2 x rank(m5_xlsr_ft) = 0.622; M3 false-alarm suppression applied (margin -6.63 < -3 and base > 0.5: x0.5 -> 0.311)`.

What changes and what does not:
- No file crosses 0.5, consistent with the sweep's count of 3 crossings over the whole test set. M3 suppression still fires on file 3 and only there.
- The largest move is file 4 (0.003 → 0.035): M5 sides with Spectra (fake, +5.64) against the probe (real, −5.78). With 20% of the weight it lifts the base from 0.157 to 0.319, still well inside the real range; the rule's asymmetry (Spectra cannot promote) is unchanged, and this file now records a two-against-one disagreement among the deep detectors rather than one-against-one.
- Files 5 and 7 lose a little confidence (0.935 → 0.897, 0.964 → 0.938) because M5 is near its boundary on both (+0.14, −0.13), while the probe and Spectra stay far into the synthetic range.
- File 8 stays the coin flip it was (0.471 → 0.466); M5 leans real (−2.23), Spectra leans fake, the probe is near the boundary.

**Router on vs off under fusion_v2** (`scripts/orchestration_ablation.py --constants models/fusion_v2/constants.json`, output `outputs/fusion/orchestration_ablation_v2.md`; "fuse everything equally" is now the four-way equal rank mean of M1b, handcrafted, M5 and Spectra):

| Split | Router on | No M3 suppression | Router off | Fuse everything equally | Files M3 suppression touched | Files the gate touched |
|---|---|---|---|---|---|---|
| Holdout, 3,858 rows | 0.0065 | 0.0200 | 0.0200 | 0.0045 | 83 (all real) | 31 |
| In-the-Wild, brief cost | 0.2290 | 0.2737 | 0.2737 | 0.2137 | 24 (all real) | 9 |
| In-the-Wild, sponsor-code cost | 0.2415 | 0.2535 | 0.2505 | 0.1955 | | |
| NSA test, share above 0.5 | 27.5% | 30.3% | 30.3% | 28.8% | 47 | 0 |

Under fusion_v2 the router-on holdout and In-the-Wild numbers reproduce the sweep's readout for `A3 w 0.2 + E` (0.0065 and 0.228 before the gate; the gate's 9 In-the-Wild rows cost 0.001 under the brief's cost and 0.003 under the sponsor code's cost here, where under fusion_v1 they gained 0.003). M3 suppression still touches only real files and is still the rule that changes decisions (0.020 → 0.0065 on the holdout, 0.274 → 0.229 on In-the-Wild). One reading differs from fusion_v1: with M5 in the blend, the four-way equal-weight rank mean beats the shipped rule on every labeled readout (holdout 0.0045, In-the-Wild 0.214 brief and 0.196 sponsor-code). That rule gives Spectra a full vote, which the fusion consult ruled out because its training data is undisclosed and its inner rows may be in-sample; the numbers are recorded here as measured, and the decision rests on that argument, not on these readouts.
