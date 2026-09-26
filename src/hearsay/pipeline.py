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

Fusion (`FusionConstants`, from models/fusion_v0/constants.json written by scripts/fuse.py):
z_d = (logit_d - mean_d) / std_d; zmean = mean of z; stack_nonlj = weights . z + intercept;
p = sigmoid(a * fused + b + prior_shift) with that rule's Platt map. A fused detector that
errored is imputed at z = 0 (its inner-fold mean) and named in the routing log. Then
`speech_gate.apply_default_answer`: a file with no speech to judge (or that failed to decode)
gets the default answer at the real end of the ranking.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hearsay import SR
from hearsay.audio import DecodeError
from hearsay.detectors.base import NEUTRAL_SCORE, ClipContext, DetectorResult, safe_run
from hearsay.detectors.speech_gate import DEFAULT_ANSWER, apply_default_answer
from hearsay.metrics import PI_SYNTH, sigmoid

REPO = Path(__file__).resolve().parents[2]
CONSTANTS_PATH = REPO / "models" / "fusion_v0" / "constants.json"
PROBE_DIR = REPO / "models" / "m1_wav2vec2-xls-r-300m_L7_20260926-0521"
HC_DIR = REPO / "models" / "hc_selected"
SPECTRA_ID = "lab260/Spectra-AASIST"

RULES = ("zmean", "stack_nonlj")
DEFAULT_RULE = "zmean"
M1_ONLY = "m1b_only"  # no constants: the probe's own LLR plus the prior shift, as make_probe_csv.py
DETECTOR_ORDER = ("m1b_v3", "handcrafted_v5", "spectra_aasist")
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
    z: dict[str, float]
    weights: dict[str, float]
    fused: float
    p: float
    imputed: tuple[str, ...]


@dataclass(frozen=True)
class FusionConstants:
    """The numbers scripts/fuse.py persisted; loaded once, never refit."""

    detectors: tuple[str, ...]
    mean: dict[str, float]
    std: dict[str, float]
    platt: dict[str, dict[str, float]]  # rule -> {a, b, prior_shift}
    stack_weights: dict[str, float] | None
    stack_intercept: float | None
    pi_synth: float
    source: str = "<dict>"

    @classmethod
    def from_dict(cls, c: Mapping[str, Any], source: str = "<dict>") -> FusionConstants:
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
        return cls(
            detectors=dets,
            mean={d: float(std[d]["mean"]) for d in dets},
            std={d: float(std[d]["std"]) for d in dets},
            platt=platt, stack_weights=w, stack_intercept=i,
            pi_synth=float(c.get("pi_synth", 0.3)), source=source,
        )  # fmt: skip

    @classmethod
    def load(cls, path: str | Path = CONSTANTS_PATH) -> FusionConstants:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), source=str(path))

    def rules(self) -> tuple[str, ...]:
        return tuple(r for r in RULES if r in self.platt and (r != "stack_nonlj" or self.stack_weights))

    def weights(self, rule: str) -> dict[str, float]:
        """Per-detector weight on z: equal for zmean, the stacker's coefficients otherwise."""
        if rule == "zmean":
            return {d: 1.0 / len(self.detectors) for d in self.detectors}
        if rule == "stack_nonlj":
            if self.stack_weights is None:
                raise ValueError("constants have no stack_nonlj weights")
            return dict(self.stack_weights)
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")

    def standardize(self, logits: Mapping[str, float | None]) -> tuple[dict[str, float], tuple[str, ...]]:
        z, imputed = {}, []
        for d in self.detectors:
            v = logits.get(d)
            if v is None or not math.isfinite(float(v)):
                z[d], imputed = 0.0, [*imputed, d]  # the inner-fold mean, as fusion v1 imputes
            else:
                z[d] = (float(v) - self.mean[d]) / self.std[d]
        return z, tuple(imputed)

    def fuse(self, logits: Mapping[str, float | None], rule: str = DEFAULT_RULE) -> FusionOutput:
        """scripts/fuse.py's arithmetic, read back from the constants: no fitting."""
        if rule not in self.platt:
            raise ValueError(f"constants have no Platt map for rule {rule!r}")
        z, imputed = self.standardize(logits)
        zs = np.array([z[d] for d in self.detectors], dtype=np.float64)
        if rule == "zmean":
            fused = float(np.mean(zs))
        elif rule == "stack_nonlj":
            w = np.array([self.weights(rule)[d] for d in self.detectors], dtype=np.float64)
            fused = float(zs @ w + self.stack_intercept)  # LogisticRegression.decision_function
        else:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        pl = self.platt[rule]
        p = float(sigmoid(pl["a"] * fused + pl["b"] + pl["prior_shift"]))
        inputs = {d: (None if logits.get(d) is None else float(logits[d])) for d in self.detectors}
        return FusionOutput(rule, inputs, z, self.weights(rule), fused, p, imputed)


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


def routing_log(items: list[dict], is_speech: bool, fo: FusionOutput, decoded: bool) -> list[str]:
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
    if gate_status == "ok":
        log.append(f"speech_gate: is_speech={str(is_speech).lower()} (voiced {voiced:.0%} of frames); "
                   f"default-answer policy {'applied' if not is_speech else 'not applied'}")  # fmt: skip
    else:
        log.append(f"speech_gate: {gate_status}; default-answer policy "
                   f"{'applied' if not is_speech else 'not applied'}")  # fmt: skip
    if fo.rule == M1_ONLY:
        fusion = "fusion: none (m1b_only: the XLS-R probe's LLR plus the 0.3 prior shift)"
    else:
        fusion = f"fusion: {fo.rule} over {', '.join(fo.inputs)}"
    if fo.imputed:
        fusion += f"; imputed at the inner-fold mean: {', '.join(fo.imputed)}"
    log.append(fusion)
    return log


def gated_probability(p_fused: float, is_speech: bool, apply_gate: bool = True) -> float:
    """The default-answer policy on one fused probability (array in, array out upstream)."""
    if not apply_gate:
        return float(p_fused)
    return float(apply_default_answer(np.array([p_fused]), np.array([is_speech]))[0])


def fusion_block(fo: FusionOutput) -> dict[str, Any]:
    return {"rule": fo.rule, "inputs": fo.inputs, "weights": fo.weights, "z": fo.z,
            "fused": fo.fused, "p_fused": fo.p, "imputed": list(fo.imputed),
            "default_answer": DEFAULT_ANSWER}  # fmt: skip


def verdict_for(p: float, is_speech: bool, decoded: bool) -> str:
    if not decoded or not is_speech:
        return "undetermined"
    return "synthetic" if p >= 0.5 else "real"


def analyze_clip(ctx: ClipContext, models: Models, consts: FusionConstants | None,
                 rule: str = DEFAULT_RULE, detectors: Sequence | None = None,
                 version: Mapping[str, Any] | None = None, apply_gate: bool = True,
                 scorers: Sequence[str] = DETECTOR_ORDER,
                 pi_synth: float = PI_SYNTH) -> dict[str, Any]:  # fmt: skip
    """One `AnalyzeResponse` for one clip. Never raises on a bad file: a decode failure gives an
    `undetermined` response with every detector in error and the default answer applied.
    `apply_gate=False` (the runner's `--policy none`) reports the gate but leaves the fused
    probability alone, for parity against TSVs that predate the gate. `scorers` names the fused
    columns to compute (the runner's --detectors); `consts=None` means no fusion: the M1 probe's
    posterior alone (`m1_only`). The evidence detectors always run."""
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
    p = gated_probability(fo.p, is_speech, apply_gate)

    ver = dict(version) if version is not None else {"git_sha": git_sha(), "models": models.version()}
    ver.setdefault("fusion", "none" if consts is None
                   else (Path(consts.source).name if consts.source != "<dict>" else consts.source))  # fmt: skip
    ver.setdefault("contract", CONTRACT_VERSION)
    ver.setdefault("rule", rule)
    ver.setdefault("policy", "speech_gate" if apply_gate else "none")
    return {
        "filename": ctx.path.name,
        "duration_s": round(x.size / SR, 2) if decoded else None,
        "probability_synthetic": p,
        "verdict": verdict_for(p, is_speech, decoded),
        "is_speech": is_speech,
        "default_answer_applied": apply_gate and not is_speech,
        "fusion": fusion_block(fo),
        "detectors": items,
        "routing_log": routing_log(items, is_speech, fo, decoded),
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
           apply_gate: bool = True, pi_synth: float = PI_SYNTH) -> dict[str, Any]:  # fmt: skip
    """Re-fuse a cached response under another rule or gate policy from its stored logits; no
    model runs. Raises KeyError when the cache lacks a logit the rule needs (rescore instead)."""
    inputs = doc["fusion"]["inputs"]
    if consts is None:
        fo = m1_only(inputs["m1b_v3"], pi_synth)
    else:
        fo = consts.fuse({d: inputs[d] for d in consts.detectors}, rule)
    is_speech = bool(doc["is_speech"])
    decoded = doc.get("duration_s") is not None
    p = gated_probability(fo.p, is_speech, apply_gate)
    out = dict(doc)
    out["probability_synthetic"] = p
    out["default_answer_applied"] = apply_gate and not is_speech
    out["verdict"] = verdict_for(p, is_speech, decoded)
    out["fusion"] = {**doc["fusion"], **fusion_block(fo)}
    log = [line for line in doc.get("routing_log", []) if not line.startswith("fusion:")]
    out["routing_log"] = [*log, routing_log([], is_speech, fo, True)[-1]]
    out["version"] = {**doc.get("version", {}), "rule": rule,
                      "policy": "speech_gate" if apply_gate else "none"}  # fmt: skip
    return out
