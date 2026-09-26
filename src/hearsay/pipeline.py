"""End-to-end inference for one clip: decode once, run every registered engineered detector
and the two deep scorers, fuse with the persisted constants, apply the non-speech gate, and
assemble the `AnalyzeResponse` the frontend contract describes
(docs/handoffs/2026-09-26_frontend-contract.md). `scripts/run_pipeline.py` (the Docker
entrypoint) and `hearsay.api` both call `analyze_clip`; nothing here refits anything.

Score paths, kept identical to the exports fusion was fit on (outputs/detector_scores/):
- m1b_v3: `prepare_segment` -> `embed_segment` on the frozen XLS-R backbone -> `Probe.llr`
  (the nsa_test_v3 embeddings were extracted with `extract_embeddings --segment`; the saved
  probe's `segment` flag is False only because train_probe.py looked for extract_meta.json
  under the comma-joined train name, so that flag is not consulted here). Embeddings were
  stored as float16 before the probe saw them; `M1_EMBED_FP16` reproduces that rounding.
- handcrafted_v5: the registered `handcrafted` detector pinned to models/hc_selected; its
  `hc_logit` feature is the export's logit (logit of the clipped P(synthetic)).
- spectra_aasist: `hearsay.spectra.prepare_input` -> `score_clip` in zero-pad mode, 16 windows,
  no peak normalization (scripts/score_spectra.py defaults), synth_logit = spoof - bonafide.

Fusion (`FusionConstants`), from a constants file, never refit:
- models/fusion_v1/constants.json (scripts/fuse_sweep.py --write; the shipped rule, `e_on_a`,
  "E on A alpha 0.2"): rank_d = searchsorted(inner_oof_sorted[d], logit_d) / len for m1b_v3
  and handcrafted_v5; base = (1 - alpha) * rank_m1b + alpha * rank_hc; M3 as false-alarm
  suppression only: if spectra_aasist logit < -3 and base > 0.5 then base *= 0.5 (M3 never
  promotes; nothing is fit on M3); p = sigmoid(a * base + b + prior_shift).
- models/fusion_v0/constants.json (scripts/fuse.py; `zmean`, `stack_nonlj`): z_d = (logit_d -
  mean_d) / std_d; zmean = mean of z; stack_nonlj = weights . z + intercept; p = sigmoid(a *
  fused + b + prior_shift) with that rule's Platt map.
A fused detector that errored is imputed (z = 0 or rank = 0.5, its inner-fold centre; a missing
M3 means no suppression) and named in the routing log. A rule the file does not define is
refused.

Policy (`hearsay.detectors.speech_gate.apply_default_answer`, the fusion consult's placement):
determinate files score BLOCK_TOP + (1 - BLOCK_TOP) * p, i.e. in [0.001, 1]; a file with no
speech to judge goes to the pinned block [FAILURE_TOP, BLOCK_TOP) ordered by a weak signal plus
a hash jitter of its filename; a decode failure goes below FAILURE_TOP. `--flip` maps 1 - p the
same way (the pre-flipped TSV, used only if NSA's scoring turns out inverted); the block stays
at the minimum.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hearsay import SR
from hearsay.audio import DecodeError
from hearsay.detectors.base import NEUTRAL_SCORE, ClipContext, DetectorResult, safe_run
from hearsay.detectors.speech_gate import BLOCK_TOP, FAILURE_TOP, apply_default_answer
from hearsay.metrics import PI_SYNTH, sigmoid

REPO = Path(__file__).resolve().parents[2]
# The shipped rule's file is fusion_v1 (DEFAULT_CONSTANTS_PATH); CONSTANTS_PATH keeps naming the
# fusion_v0 file, which docker/build.sh stages beside it and tests/test_docker_image.py pins.
CONSTANTS_V0_PATH = REPO / "models" / "fusion_v0" / "constants.json"
CONSTANTS_V1_PATH = REPO / "models" / "fusion_v1" / "constants.json"
CONSTANTS_PATH = CONSTANTS_V0_PATH
DEFAULT_CONSTANTS_PATH = CONSTANTS_V1_PATH
PROBE_DIR = REPO / "models" / "m1_wav2vec2-xls-r-300m_L7_20260926-0521"
HC_DIR = REPO / "models" / "hc_selected"
SPECTRA_ID = "lab260/Spectra-AASIST"

RANK_RULE = "e_on_a"  # the fusion_v1 rule ("E on A alpha 0.2")
Z_RULES = ("zmean", "stack_nonlj")  # the fusion_v0 rules
RULES = (RANK_RULE, *Z_RULES)
DEFAULT_RULE = RANK_RULE
M1_ONLY = "m1b_only"  # no constants: the probe's own LLR plus the prior shift, as make_probe_csv.py
DETECTOR_ORDER = ("m1b_v3", "handcrafted_v5", "spectra_aasist")
RANKED = ("m1b_v3", "handcrafted_v5")  # the two detectors e_on_a ranks; spectra_aasist only suppresses
SCORER_FLAGS = {"m1b": "m1b_v3", "handcrafted": "handcrafted_v5", "spectra": "spectra_aasist"}
M1_MODES = ("segment", "windows", "auto")
DEEP = ("m1b_v3", "spectra_aasist")  # scored by `Models`, not by a registered detector
FUSED_FROM_DETECTOR = {"handcrafted_v5": ("handcrafted", "hc_logit")}
SPECTRA_PAD_MODE = "zero"
SPECTRA_MAX_WINDOWS = 16
M1_EMBED_FP16 = True  # the extracted embeddings were saved as float16; match that rounding
# 50-file parity vs the m1b_v3 export (Sat 06:50): fp16 max |dlogit| 5.3e-4, fp32 4.7e-3.
MAX_THREADS = 6  # CPU-only lane: never more than six cores

ROLE = {
    "handcrafted": "fused", "m1b_v3": "fused", "spectra_aasist": "fused",
    "container": "routing", "speech_gate": "gate",
    "compression": "evidence", "enf": "evidence", "splice": "evidence", "speaker_drift": "evidence",
}  # fmt: skip
CONTRACT_VERSION = "v0"


# --- fusion constants ------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionOutput:
    rule: str
    inputs: dict[str, float | None]  # raw logits, None when the detector produced none
    terms: dict[str, float]  # per-detector z (fusion_v0) or ECDF rank (fusion_v1)
    weights: dict[str, float]
    fused: float  # the rule's output before the Platt map (base rank, or the stacked z)
    p: float  # Platt probability, our direction, before the policy's determinate map
    imputed: tuple[str, ...]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class FusionConstants:
    """The numbers scripts/fuse.py or scripts/fuse_sweep.py persisted; loaded once, never refit."""

    detectors: tuple[str, ...]
    platt: dict[str, dict[str, float]]  # rule -> {a, b, prior_shift}
    pi_synth: float
    source: str = "<dict>"
    # fusion_v0 (z rules)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    stack_weights: dict[str, float] | None = None
    stack_intercept: float | None = None
    # fusion_v1 (rank rule)
    rank_ref: dict[str, np.ndarray] | None = None
    alpha: float | None = None
    e_rule: dict[str, Any] | None = None
    final: str | None = None
    how: str | None = None

    @classmethod
    def from_dict(cls, c: Mapping[str, Any], source: str = "<dict>") -> FusionConstants:
        if "rank_ref_inner_oof_sorted" in c:
            return cls._from_v1(c, source)
        return cls._from_v0(c, source)

    @classmethod
    def _from_v1(cls, c: Mapping[str, Any], source: str) -> FusionConstants:
        refs = c["rank_ref_inner_oof_sorted"]
        rank_ref = {}
        for d in RANKED:
            if d not in refs:
                raise ValueError(f"constants: no rank reference for {d!r}")
            ref = np.asarray(refs[d], dtype=np.float64)
            if ref.size == 0 or not np.all(np.diff(ref) >= 0):
                raise ValueError(f"constants: rank reference for {d!r} is empty or not sorted")
            rank_ref[d] = ref
        alpha = c.get("alpha_handcrafted")
        if alpha is None or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"constants: alpha_handcrafted must be in [0, 1], got {alpha!r}")
        pl = c["platt"]
        e = dict(c.get("e_rule") or {"applied": False})
        if e.get("applied"):
            for k in ("m3_logit_below", "base_rank_above", "multiply_by"):
                if k not in e:
                    raise ValueError(f"constants: e_rule.{k} missing")
        return cls(detectors=DETECTOR_ORDER,
                   platt={RANK_RULE: {k: float(pl[k]) for k in ("a", "b", "prior_shift")}},
                   pi_synth=float(c.get("pi_synth", PI_SYNTH)), source=source, rank_ref=rank_ref,
                   alpha=float(alpha), e_rule=e, final=str(c.get("final", RANK_RULE)),
                   how=c.get("how"))  # fmt: skip

    @classmethod
    def _from_v0(cls, c: Mapping[str, Any], source: str) -> FusionConstants:
        dets = tuple(c["detectors"])
        std = c["standardize"]
        for d in dets:
            if d not in std:
                raise ValueError(f"constants: no standardize entry for {d!r}")
        rules = c.get("rules", {})
        platt = {}
        for name, r in rules.items():
            if "platt" in r:
                pl = r["platt"]
                platt[name] = {k: float(pl[k]) for k in ("a", "b", "prior_shift")}
        st = rules.get("stack_nonlj")
        w = i = None
        if st is not None and "weights" in st:
            if len(st["weights"]) != len(dets):
                raise ValueError("constants: stack_nonlj weights do not match detectors")
            w = dict(zip(dets, (float(v) for v in st["weights"]), strict=True))
            i = float(st["intercept"])
        return cls(detectors=dets, platt=platt, pi_synth=float(c.get("pi_synth", PI_SYNTH)),
                   source=source, mean={d: float(std[d]["mean"]) for d in dets},
                   std={d: float(std[d]["std"]) for d in dets}, stack_weights=w, stack_intercept=i,
                   how=(c.get("how") or {}).get("stack_nonlj"))  # fmt: skip

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONSTANTS_PATH) -> FusionConstants:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), source=str(path))

    def rules(self) -> tuple[str, ...]:
        """The rules this file defines, in RULES order."""
        out = []
        for r in RULES:
            if r not in self.platt:
                continue
            if r == RANK_RULE and self.rank_ref is None:
                continue
            if r in Z_RULES and not self.mean:
                continue
            if r == "stack_nonlj" and not self.stack_weights:
                continue
            out.append(r)
        return tuple(out)

    def check_rule(self, rule: str) -> None:
        if rule not in self.rules():
            raise ValueError(f"rule {rule!r} is not defined by {self.source} (has {self.rules()})")

    def weights(self, rule: str) -> dict[str, float]:
        """Per-detector weight: equal for zmean, the stacker's coefficients for stack_nonlj,
        (1 - alpha, alpha, 0) for e_on_a (spectra only suppresses; it has no weight)."""
        self.check_rule(rule)
        if rule == RANK_RULE:
            return {"m1b_v3": 1.0 - self.alpha, "handcrafted_v5": self.alpha, "spectra_aasist": 0.0}
        if rule == "zmean":
            return {d: 1.0 / len(self.detectors) for d in self.detectors}
        return dict(self.stack_weights)

    def standardize(self, logits: Mapping[str, float | None]) -> tuple[dict[str, float], tuple[str, ...]]:
        z, imputed = {}, []
        for d in self.detectors:
            v = logits.get(d)
            if v is None or not math.isfinite(float(v)):
                z[d], imputed = 0.0, [*imputed, d]  # the inner-fold mean, as fusion v1 imputes
            else:
                z[d] = (float(v) - self.mean[d]) / self.std[d]
        return z, tuple(imputed)

    def rank(self, det: str, logit: float) -> float:
        """ECDF position against the detector's inner-OOF logits (scripts/fuse_sweep.py's rank)."""
        ref = self.rank_ref[det]
        return float(np.searchsorted(ref, float(logit)) / len(ref))

    def fuse(self, logits: Mapping[str, float | None], rule: str = DEFAULT_RULE) -> FusionOutput:
        """The persisted rule's arithmetic, read back from the constants: no fitting."""
        self.check_rule(rule)
        inputs = {d: (None if logits.get(d) is None else float(logits[d])) for d in self.detectors}
        pl = self.platt[rule]
        if rule == RANK_RULE:
            terms, imputed, detail = {}, [], {"final": self.final}
            for d in RANKED:
                v = inputs[d]
                if v is None or not math.isfinite(v):
                    terms[d], imputed = 0.5, [*imputed, d]  # the inner-fold median
                else:
                    terms[d] = self.rank(d, v)
            a = self.alpha
            base = (1 - a) * terms["m1b_v3"] + a * terms["handcrafted_v5"]  # fuse_sweep's order of ops
            detail["base"] = base
            m3 = inputs["spectra_aasist"]
            m3_ok = m3 is not None and math.isfinite(m3)
            if not m3_ok:
                imputed.append("spectra_aasist")  # no evidence: no suppression
            e = self.e_rule or {}
            applied = bool(e.get("applied") and m3_ok and m3 < e["m3_logit_below"]
                           and base > e["base_rank_above"])  # fmt: skip
            fused = base * e["multiply_by"] if applied else base
            detail["e_applied"] = applied
            detail["e_rule"] = e
        else:
            terms, imputed = self.standardize(logits)
            zs = np.array([terms[d] for d in self.detectors], dtype=np.float64)
            if rule == "zmean":
                fused = float(np.mean(zs))
            else:
                w = np.array([self.weights(rule)[d] for d in self.detectors], dtype=np.float64)
                fused = float(zs @ w + self.stack_intercept)  # LogisticRegression.decision_function
            detail = {}
            imputed = tuple(imputed)
        p = float(sigmoid(pl["a"] * fused + pl["b"] + pl["prior_shift"]))
        return FusionOutput(rule, inputs, terms, self.weights(rule), float(fused), p, tuple(imputed), detail)


def m1_only(logit: float | None, pi_synth: float = PI_SYNTH) -> FusionOutput:
    """No fusion: P = sigmoid(LLR + logit(pi_synth)), scripts/make_probe_csv.py's arithmetic;
    a missing LLR gives the prior-only posterior, its fallback."""
    shift = math.log(pi_synth / (1 - pi_synth))
    ok = logit is not None and math.isfinite(float(logit))
    fused = float(logit) + shift if ok else shift
    return FusionOutput(M1_ONLY, {"m1b_v3": float(logit) if ok else None},
                        {"m1b_v3": float(logit) if ok else 0.0}, {"m1b_v3": 1.0}, fused,
                        float(sigmoid(fused)), () if ok else ("m1b_v3",))  # fmt: skip


# --- models held in memory -------------------------------------------------------------------


def set_cpu_threads(n: int | None = None) -> int:
    """torch threads for this process: min(n or all cores, MAX_THREADS)."""
    import torch

    cores = os.cpu_count() or 1
    k = max(1, min(int(n) if n else cores, MAX_THREADS))
    torch.set_num_threads(k)
    return k


def truncate_backbone(model, layer: int) -> int:
    """Drop encoder layers above `layer`. hidden_states[k] is the input to encoder layer k (both
    Wav2Vec2 encoder variants append it before running the layer), so keeping layers[:layer+1]
    leaves hidden_states[layer] bit-identical and skips the 24-layer model's remaining work
    (about two thirds of M1's CPU time at layer 7). Verified on the real backbone
    (tests/test_pipeline.py::test_truncated_backbone_keeps_layer_7_identical)."""
    layers = model.encoder.layers
    keep = min(int(layer) + 1, len(layers))
    model.encoder.layers = layers[:keep]
    return keep - 1


class Models:
    """The heavy models, loaded once: XLS-R backbone + probe, Spectra-AASIST, the handcrafted
    bundle. CPU only (the GPU belongs to the training lane)."""

    def __init__(self, device: str = "cpu", probe_dir: str | Path = PROBE_DIR,
                 hc_dir: str | Path = HC_DIR, threads: int | None = None,
                 load_deep: bool = True, m1_mode: str = "segment",
                 load_spectra: bool = True, truncate: bool = True) -> None:  # fmt: skip
        if device != "cpu":
            raise ValueError("the inference pipeline is CPU-only; device must be 'cpu'")
        if m1_mode not in M1_MODES:
            raise ValueError(f"m1_mode must be one of {M1_MODES}, got {m1_mode!r}")
        self.m1_mode = "segment" if m1_mode == "auto" else m1_mode
        self.device = device
        self.with_spectra = load_spectra
        self.truncate = truncate
        self.truncated_to_layer: int | None = None
        self.threads = set_cpu_threads(threads)
        self.probe_dir = Path(probe_dir)
        self.hc_dir = Path(hc_dir)
        self.load_seconds: dict[str, float] = {}
        from hearsay.detectors.handcrafted import HandcraftedDetector

        self.handcrafted = HandcraftedDetector(model_dir=self.hc_dir)
        self.probe = self.backbone = self.spectra = None
        if load_deep:
            self.load_deep()

    def load_deep(self) -> None:
        from hearsay.embed import load_backbone
        from hearsay.probe import Probe
        from hearsay.spectra import load_spectra

        t = time.time()
        self.probe = Probe.load(self.probe_dir)
        self.backbone = load_backbone(self.probe.backbone, device=self.device)
        if self.truncate:
            self.truncated_to_layer = truncate_backbone(self.backbone, self.probe.layer)
        self.load_seconds["m1"] = round(time.time() - t, 2)
        if bool(getattr(self.probe, "segment", False)) != (self.m1_mode == "segment"):
            print(f"note: probe.segment={getattr(self.probe, 'segment', None)} but m1_mode="
                  f"{self.m1_mode!r}; the flag is not trusted (see the module docstring)", flush=True)  # fmt: skip
        if self.with_spectra:
            t = time.time()
            self.spectra = load_spectra(self.device)
            self.load_seconds["spectra"] = round(time.time() - t, 2)

    # score paths (see the module docstring for why each looks the way it does)

    def m1_logit(self, x: np.ndarray) -> float:
        from hearsay.embed import embed_clip, embed_segment, prepare_segment

        if self.m1_mode == "windows":  # the v1 path: 4 s windows, tiled when short
            e = embed_clip(self.backbone, x, self.probe.win_s, self.probe.max_windows)
        else:
            e = embed_segment(self.backbone, prepare_segment(x))
        if M1_EMBED_FP16:
            e = e.astype(np.float16).astype(np.float32)
        return float(self.probe.llr(e)[0])

    def spectra_logit(self, x: np.ndarray) -> float:
        from hearsay.spectra import prepare_input, score_clip, synth_logit

        if self.spectra is None:
            raise RuntimeError("Spectra-AASIST was not loaded (load_spectra=False)")
        out = score_clip(self.spectra, prepare_input(x), (SPECTRA_PAD_MODE,), SPECTRA_MAX_WINDOWS)
        return synth_logit(out[SPECTRA_PAD_MODE])

    def engineered(self) -> list:
        """Every registered engineered detector, with handcrafted pinned to `hc_dir`."""
        import hearsay.detectors.engineered  # noqa: F401 - registers the D-track detectors
        from hearsay.detectors import all_detectors

        return [self.handcrafted if d.name == self.handcrafted.name else d for d in all_detectors()]

    def version(self) -> dict[str, Any]:
        hc = self.hc_dir.resolve().name if self.hc_dir.exists() else self.hc_dir.name
        return {"m1": self.probe_dir.resolve().name, "m1_mode": self.m1_mode,
                "m1_truncated_to_layer": self.truncated_to_layer, "handcrafted": hc,
                "spectra": SPECTRA_ID}  # fmt: skip


# --- one clip --------------------------------------------------------------------------------


def git_sha(repo: str | Path = REPO) -> str:
    """Short git sha: HEARSAY_GIT_SHA (baked into the image) or `git rev-parse`."""
    env = os.environ.get("HEARSAY_GIT_SHA")
    if env:
        return env[:12]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                                       text=True, stderr=subprocess.DEVNULL).strip()  # fmt: skip
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def detector_entry(r: DetectorResult, seconds: float, role: str | None = None) -> dict[str, Any]:
    return {
        "name": r.name, "role": role or ROLE.get(r.name, "evidence"), "status": r.status,
        "score": float(r.score), "evidence": r.evidence,
        "features": {k: float(v) for k, v in r.features.items()},
        "seconds": round(float(seconds), 3), "error": r.error,
    }  # fmt: skip


def _error_entry(name: str, msg: str, seconds: float = 0.0) -> dict[str, Any]:
    r = DetectorResult(name, NEUTRAL_SCORE, f"detector failed: {msg}", status="error", error=msg)
    return detector_entry(r, seconds)


def _deep_entry(name: str, logit: float, seconds: float) -> dict[str, Any]:
    p = float(sigmoid(logit))
    if name == "m1b_v3":
        why = (f"XLS-R layer-7 probe: calibrated log-likelihood ratio {logit:+.2f} "
               f"({'synthetic' if logit > 0 else 'real'}-like, P={p:.2f})")  # fmt: skip
    else:
        why = (f"Spectra-AASIST: spoof-minus-bonafide margin {logit:+.2f} "
               f"({'synthetic' if logit > 0 else 'real'}-like, P={p:.2f})")  # fmt: skip
    r = DetectorResult(name, min(max(p, 0.0), 1.0), why, {"logit": float(logit)})
    return detector_entry(r, seconds, "fused")


def _feat(items: list[dict], name: str, key: str) -> float | None:
    for it in items:
        if it["name"] == name and it["status"] == "ok":
            v = it["features"].get(key)
            return None if v is None else float(v)
    return None


def fusion_line(fo: FusionOutput) -> str:
    """The routing log's fusion entry: what was combined and, for e_on_a, whether M3 suppressed."""
    if fo.rule == M1_ONLY:
        line = "fusion: none (m1b_only: the XLS-R probe's LLR plus the 0.3 prior shift)"
    elif fo.rule == RANK_RULE:
        w = fo.weights
        d = fo.detail
        line = (f"fusion: {fo.rule} ({d.get('final')}): {w['m1b_v3']:.1f} x rank(m1b_v3) + "
                f"{w['handcrafted_v5']:.1f} x rank(handcrafted_v5) = {d.get('base', fo.fused):.3f}")  # fmt: skip
        m3 = fo.inputs.get("spectra_aasist")
        e = d.get("e_rule") or {}
        if d.get("e_applied"):
            line += (f"; M3 false-alarm suppression applied (margin {m3:+.2f} < {e['m3_logit_below']:g} and "
                     f"base > {e['base_rank_above']:g}: x{e['multiply_by']:g} -> {fo.fused:.3f})")  # fmt: skip
        elif m3 is None:
            line += "; M3 unavailable, no suppression"
        else:
            line += f"; M3 margin {m3:+.2f}, no suppression (M3 never promotes)"
    else:
        line = f"fusion: {fo.rule} over {', '.join(fo.inputs)}"
    if fo.imputed:
        line += f"; imputed at the inner-fold centre: {', '.join(fo.imputed)}"
    return line


def routing_log(items: list[dict], is_speech: bool, fo: FusionOutput, decoded: bool,
                apply_gate: bool = True) -> list[str]:  # fmt: skip
    """Plain-English record of what ran and why, from the detector features."""
    log: list[str] = []
    if not decoded:
        log.append("decode: ffmpeg produced no samples; no detector ran; the default answer applies")
    pcm = _feat(items, "container", "is_pcm_wav")
    lossy = _feat(items, "container", "lossy")
    if pcm is not None and lossy is not None:
        ff = _feat(items, "container", "ffmpeg_written")
        kind = "PCM WAV" if pcm else "not PCM WAV"
        parts = [kind, "lossy codec" if lossy else "not lossy"]
        if ff:
            parts.append("FFmpeg-written")
        use = ("compression forensics apply (evidence)" if lossy
               else "compression forensics run for evidence, not score")  # fmt: skip
        log.append(f"container: {', '.join(parts)}; {use}")
    bw = _feat(items, "compression", "bw_hz")
    if bw is not None:
        where = "at or below the 7.25 kHz test-set band match" if bw <= 7250 else "above the 7.25 kHz band match"
        log.append(f"compression: effective bandwidth {bw:.0f} Hz, {where}")
    enf = _feat(items, "enf", "enf_present")
    if enf is not None:
        if enf:
            stable = _feat(items, "enf", "enf_stable")
            log.append("enf: mains hum present, " + ("stable (recording-environment evidence)"
                                                    if stable else "discontinuous (spliced-background evidence)"))  # fmt: skip
        else:
            log.append("enf: no mains hum; no environment evidence either way")
    seams = _feat(items, "splice", "n_seams")
    if seams is not None:
        log.append(f"splice: {seams:.0f} editing seam(s)" if seams else "splice: no editing seams")
    drift = _feat(items, "speaker_drift", "drift")
    if drift is not None:
        cmin = _feat(items, "speaker_drift", "cos_min")
        log.append(f"speaker_drift: {'voice drifts' if drift else 'one consistent voice'} "
                   f"(min window similarity {cmin:.2f}); evidence only, never fused")  # fmt: skip
    voiced = _feat(items, "speech_gate", "voiced_frac")
    gate_status = next((it["status"] for it in items if it["name"] == "speech_gate"), "missing")
    gated = apply_gate and not is_speech
    policy = ("default-answer policy applied (pinned below every scored file)" if gated
              else "default-answer policy not applied" if apply_gate
              else "policy off (--policy none)")  # fmt: skip
    if gate_status == "ok":
        log.append(f"speech_gate: is_speech={str(is_speech).lower()} (voiced {voiced:.0%} of frames); {policy}")
    else:
        log.append(f"speech_gate: {gate_status}; {policy}")
    log.append(fusion_line(fo))
    return log


def verdict_for(p: float, is_speech: bool, decoded: bool) -> str:
    if not decoded or not is_speech:
        return "undetermined"
    return "synthetic" if p >= 0.5 else "real"


def final_score(p_fused: float, is_speech: bool, apply_gate: bool = True, *, key: str | None = None,
                order_by: float | None = None, failed: bool = False, flip: bool = False) -> float:  # fmt: skip
    """The policy on one file (array in, array out upstream): the determinate map
    BLOCK_TOP + (1 - BLOCK_TOP) * p for speech files, the pinned block for gated ones (ordered by
    `order_by`, jittered by a hash of `key`), below FAILURE_TOP when `failed`. `flip` maps 1 - p
    the same way (the pre-flipped TSV); `apply_gate=False` maps every file as determinate."""
    p = 1.0 - p_fused if flip else p_fused
    speech = bool(is_speech or not apply_gate)
    sig = None if order_by is None else np.array([1.0 - order_by if flip else order_by])
    out = apply_default_answer(np.array([p]), np.array([speech]), order_by=sig,
                               keys=None if key is None else [key],
                               failed=np.array([bool(failed) and not speech]))  # fmt: skip
    return float(out[0])


def fusion_block(fo: FusionOutput) -> dict[str, Any]:
    return {"rule": fo.rule, "inputs": fo.inputs, "weights": fo.weights, "terms": fo.terms,
            "fused": fo.fused, "p_fused": fo.p, "imputed": list(fo.imputed), "detail": fo.detail,
            "block_top": BLOCK_TOP, "failure_top": FAILURE_TOP}  # fmt: skip


def analyze_clip(ctx: ClipContext, models: Models, consts: FusionConstants | None,
                 rule: str = DEFAULT_RULE, detectors: Sequence | None = None,
                 version: Mapping[str, Any] | None = None, apply_gate: bool = True,
                 scorers: Sequence[str] = DETECTOR_ORDER,
                 pi_synth: float = PI_SYNTH, flip: bool = False) -> dict[str, Any]:  # fmt: skip
    """One `AnalyzeResponse` for one clip. Never raises on a bad file: a decode failure gives an
    `undetermined` response with every detector in error and the default answer applied.
    `apply_gate=False` (the runner's `--policy none`) reports the gate but maps every file as
    determinate, for parity against TSVs that predate the gate. `scorers` names the fused
    columns to compute (the runner's --detectors); `consts=None` means no fusion: the M1 probe's
    posterior alone (`m1_only`). `flip` emits the pre-flipped score. The evidence detectors
    always run."""
    t_start = time.time()
    items: list[dict[str, Any]] = []
    scorers = tuple(scorers)
    hc_name = FUSED_FROM_DETECTOR["handcrafted_v5"][0]
    dets = [d for d in (list(detectors) if detectors is not None else models.engineered())
            if d.name != hc_name or "handcrafted_v5" in scorers]  # fmt: skip
    deep = [(n, fn) for n, fn in (("m1b_v3", models.m1_logit), ("spectra_aasist", models.spectra_logit))
            if n in scorers]  # fmt: skip
    try:
        x = ctx.audio
        decoded = True
    except DecodeError as e:
        x, decoded = None, False
        msg = f"DecodeError: {e}"
        items = [_error_entry(d.name, msg) for d in dets] + [_error_entry(n, msg) for n, _ in deep]
    if decoded:
        for d in dets:
            t0 = time.time()
            items.append(detector_entry(safe_run(d, ctx), time.time() - t0))
        for name, fn in deep:
            t0 = time.time()
            try:
                lg = float(fn(x))
                if not math.isfinite(lg):
                    raise ValueError(f"non-finite logit {lg}")
                items.append(_deep_entry(name, lg, time.time() - t0))
            except Exception as e:  # noqa: BLE001 - one failing scorer must not lose the row
                items.append(_error_entry(name, f"{type(e).__name__}: {e}", time.time() - t0))

    fo = fuse_items(items, consts, rule, pi_synth)
    gate = _feat(items, "speech_gate", "is_speech")
    is_speech = bool(decoded and (gate is None or gate))  # a gate error does not gate
    m1 = fo.inputs.get("m1b_v3")
    order_by = float(sigmoid(m1)) if m1 is not None else fo.p  # the block's weak ordering signal
    p = final_score(fo.p, is_speech, apply_gate, key=ctx.path.name, order_by=order_by,
                    failed=not decoded, flip=flip)  # fmt: skip

    ver = dict(version) if version is not None else {"git_sha": git_sha(), "models": models.version()}
    ver.setdefault("fusion", "none" if consts is None
                   else (Path(consts.source).name if consts.source != "<dict>" else consts.source))  # fmt: skip
    ver.setdefault("contract", CONTRACT_VERSION)
    ver.setdefault("rule", fo.rule)
    ver.setdefault("policy", "speech_gate" if apply_gate else "none")
    ver.setdefault("polarity", "flipped" if flip else "our_direction")
    return {
        "filename": ctx.path.name,
        "duration_s": round(x.size / SR, 2) if decoded else None,
        "probability_synthetic": p,
        "verdict": verdict_for(fo.p, is_speech, decoded),
        "is_speech": is_speech,
        "default_answer_applied": apply_gate and not is_speech,
        "fusion": fusion_block(fo),
        "detectors": items,
        "routing_log": routing_log(items, is_speech, fo, decoded, apply_gate),
        "flag": "" if decoded else "decode_error",
        "seconds": round(time.time() - t_start, 3),
        "version": ver,
    }


def collect_logits(items: list[dict], names: Sequence[str]) -> dict[str, float | None]:
    """The fused columns' raw logits out of the detector entries (None when absent or errored)."""
    logits: dict[str, float | None] = {}
    for d in names:
        if d in DEEP:
            logits[d] = _feat(items, d, "logit")
        elif d in FUSED_FROM_DETECTOR:
            det_name, key = FUSED_FROM_DETECTOR[d]
            logits[d] = _feat(items, det_name, key)
        else:
            logits[d] = None
    return logits


def fuse_items(items: list[dict], consts: FusionConstants | None, rule: str,
               pi_synth: float = PI_SYNTH) -> FusionOutput:  # fmt: skip
    if consts is None:
        return m1_only(collect_logits(items, ("m1b_v3",))["m1b_v3"], pi_synth)
    return consts.fuse(collect_logits(items, consts.detectors), rule)


def refuse(doc: Mapping[str, Any], consts: FusionConstants | None, rule: str,
           apply_gate: bool = True, pi_synth: float = PI_SYNTH, flip: bool = False) -> dict[str, Any]:  # fmt: skip
    """Re-fuse a cached response under another rule, gate policy or polarity from its stored
    logits; no model runs. Raises KeyError when the cache lacks a logit the rule needs (rescore
    instead)."""
    inputs = doc["fusion"]["inputs"]
    if consts is None:
        fo = m1_only(inputs["m1b_v3"], pi_synth)
    else:
        fo = consts.fuse({d: inputs[d] for d in consts.detectors}, rule)
    is_speech = bool(doc["is_speech"])
    decoded = doc.get("duration_s") is not None
    m1 = fo.inputs.get("m1b_v3")
    order_by = float(sigmoid(m1)) if m1 is not None else fo.p
    p = final_score(fo.p, is_speech, apply_gate, key=doc["filename"], order_by=order_by,
                    failed=not decoded, flip=flip)  # fmt: skip
    out = dict(doc)
    out["probability_synthetic"] = p
    out["default_answer_applied"] = apply_gate and not is_speech
    out["verdict"] = verdict_for(fo.p, is_speech, decoded)
    out["fusion"] = fusion_block(fo)
    log = [line for line in doc.get("routing_log", []) if not line.startswith(("fusion:", "speech_gate:"))]
    out["routing_log"] = [*log, *routing_log(doc.get("detectors", []), is_speech, fo, True, apply_gate)[-2:]]
    out["version"] = {**doc.get("version", {}), "rule": fo.rule,
                      "policy": "speech_gate" if apply_gate else "none",
                      "polarity": "flipped" if flip else "our_direction"}  # fmt: skip
    return out
