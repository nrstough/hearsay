"""Spectra-AASIST (lab260) score stream, M3.

Model card: logits (batch, 2), index 0 = spoof, index 1 = bonafide; trained on 64,600-sample
(~4 s) crops after pre-emphasis, short clips repeat-padded (`pad_random`).

M3 input path, shared with M1 and the handcrafted detector (scripts/score_spectra.py):
load_audio -> prepare_input (= hearsay.embed.prepare_segment: band match, trim, optional
test-length crop, cap 8 s) -> short-clip handling in one of three modes (repeat / zero / whole)
-> 64,600-sample windows at 50% hop -> pre-emphasis -> model -> mean of the window logits, and

    synth_logit = logit_spoof - logit_bonafide      (increases with synthetic likelihood)

The export carries the raw margin as `logit` and its sigmoid as `score`; nothing is fitted.
Direction is asserted on inner-fold rows before any fusion file is written, and by
scripts/spectra_direction.py on labeled samples (that script uses the default `repeat` path,
which is byte-identical to what it ran with before M3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import roc_auc_score, roc_curve

from hearsay import SR
from hearsay.audio import windows
from hearsay.embed import prepare_segment, test_duration_sampler
from hearsay.metrics import C_FA, C_MISS, PI_SYNTH, by_group, report, sigmoid

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights" / "Spectra-AASIST"
WIN = 64_600
HOP = WIN // 2
PAD_MODES = ("repeat", "zero", "whole")
MAX_INPUT_SAMPLES = 128_000  # AASIST pos_T covers 400 SSL frames; frame 401 appears at 128,400
MIN_WHOLE_SAMPLES = SR  # 1 s floor for whole mode (Encoder max_pool2d(3,3) needs >= 3 frames)


def load_spectra(device: str | None = None) -> torch.nn.Module:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    spec = importlib.util.spec_from_file_location("spectra_model", WEIGHTS / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # model.py hardcodes the hub id for its XLS-R skeleton; point it at weights/ instead.
    # Every encoder weight is then overwritten by Spectra-AASIST's own safetensors.
    hub_cls = mod.Wav2Vec2Model

    class _LocalWav2Vec2:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            if name == "facebook/wav2vec2-xls-r-300m":
                name = REPO / "weights" / "wav2vec2-xls-r-300m"
            return hub_cls.from_pretrained(name, *a, **kw)

    mod.Wav2Vec2Model = _LocalWav2Vec2
    model = mod.SpectraAASIST.from_pretrained(str(WEIGHTS))
    return model.eval().to(device)


# --- input path -----------------------------------------------------------------------------


def prepare_input(x: np.ndarray, crop_s: float | None = None, seed: int | None = None,
                  band_match: bool = True) -> np.ndarray:  # fmt: skip
    """The shared M1 path: band match -> trim -> optional crop -> cap 8 s (embed.prepare_segment)."""
    return prepare_segment(x, crop_s, seed, band_match=band_match)


def peak_normalize(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Scale the clip so its peak is `target` (silence stays silent). Optional, applied to every
    row alike after prepare_input: Spectra does not level-normalize (normalize_waveform=False),
    training real audio peaks ~0.5 while NSA test clips and some spoof generators sit ~1.0, so
    raw level is a synthetic cue that would push loud real test clips toward "fake"."""
    x = np.asarray(x, dtype=np.float32)
    m = float(np.abs(x).max()) if x.size else 0.0
    if m == 0.0:
        return x
    return (x * np.float32(target / m)).astype(np.float32)


def pad_windows(x: np.ndarray, pad_mode: str = "repeat", max_windows: int = 16) -> np.ndarray:
    """(n, L) float32 windows. Clips >= WIN: 50%-hop windows, identical in every mode.
    Shorter clips: repeat = tile to WIN (the model card's pad_random; a tiling seam), zero =
    zero-pad to WIN, whole = the clip at its own length with a 1 s zero-pad floor."""
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}")
    x = np.asarray(x, dtype=np.float32)
    if x.size >= WIN or pad_mode == "repeat":
        return windows(x, WIN, HOP)[:max_windows]  # the pre-M3 path, byte-identical
    if pad_mode == "zero":
        return np.pad(x, (0, WIN - x.size))[None]
    if x.size < MIN_WHOLE_SAMPLES:
        x = np.pad(x, (0, MIN_WHOLE_SAMPLES - x.size))
    return np.ascontiguousarray(x[None])


def pad_samples(n_samples: int, pad_mode: str) -> int:
    """Samples pad_windows adds for a clip of n_samples in this mode (0 for clips >= WIN)."""
    if n_samples >= WIN:
        return 0
    if pad_mode in ("repeat", "zero"):
        return WIN - n_samples
    return max(0, MIN_WHOLE_SAMPLES - n_samples)


@torch.inference_mode()
def forward_windows(model: torch.nn.Module, w: np.ndarray) -> np.ndarray:
    """(n, 2) logits for equal-length windows. Pre-emphasis on CPU before .to(device): the order
    scripts/spectra_direction.py's ASVspoof numbers were produced with."""
    if w.shape[1] > MAX_INPUT_SAMPLES:
        raise ValueError(f"{w.shape[1]} samples exceed the 400-frame AASIST positional table (8 s)")
    device = next(model.parameters()).device
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    t = torchaudio.functional.preemphasis(t)
    return model(t.to(device)).float().cpu().numpy()


@torch.inference_mode()
def spectra_logits(model: torch.nn.Module, x: np.ndarray, max_windows: int = 16,
                   pad_mode: str = "repeat") -> np.ndarray:  # fmt: skip
    """Window-averaged (logit_spoof, logit_bonafide) for one 16 kHz mono clip."""
    return forward_windows(model, pad_windows(x, pad_mode, max_windows)).mean(axis=0)


def synth_logit(logits: np.ndarray) -> float:
    return float(logits[0] - logits[1])


def score_clip(model: torch.nn.Module, x: np.ndarray, pad_modes=("repeat",),
               max_windows: int = 16) -> dict:  # fmt: skip
    """Per-mode logits for one prepared clip: {"n_samples", "n_windows", <mode>: (2,) logits,
    "n_pad_samples_<mode>": int}. Clips >= WIN forward once and every mode gets the same logits.
    A forward whose output is not (n, 2) or not finite raises RuntimeError (a failed row)."""
    x = np.asarray(x, dtype=np.float32)
    out = {"n_samples": int(x.size)}

    def fwd(w: np.ndarray) -> np.ndarray:
        lg = forward_windows(model, w)
        if lg.ndim != 2 or lg.shape[1] != 2 or not np.all(np.isfinite(lg)):
            raise RuntimeError(f"model output invalid: shape {lg.shape}, finite {np.isfinite(lg).all()}")
        lg = lg.mean(axis=0)
        if not np.isfinite(synth_logit(lg)):
            raise RuntimeError("synth_logit not finite")
        return lg

    if x.size >= WIN:
        w = pad_windows(x, "repeat", max_windows)
        lg = fwd(w)
        out["n_windows"] = int(w.shape[0])
        for m in pad_modes:
            out[m], out[f"n_pad_samples_{m}"] = lg, 0
        return out
    out["n_windows"] = 1
    for m in pad_modes:
        out[m] = fwd(pad_windows(x, m, max_windows))
        out[f"n_pad_samples_{m}"] = pad_samples(x.size, m)
    return out


def crop_plan(manifest_csv: str | Path, seed: int) -> dict[str, tuple[float, int]]:
    """path -> (crop_s, offset seed): one test-duration draw per manifest row in order from
    test_duration_sampler(seed), offset seed = seed + row index. Reproduces
    extract_embeddings --segment --crop test --seed <seed> and extract_handcrafted exactly.
    Draws for every row before any --limit, so a pilot crops each file as the full run does."""
    paths = pd.read_csv(manifest_csv).path.tolist()
    if len(set(paths)) != len(paths):
        raise ValueError(f"{manifest_csv}: duplicate paths; the crop plan is keyed by path")
    draw = test_duration_sampler(seed)
    return {p: (draw(), seed + i) for i, p in enumerate(paths)}


def resolve_path(p: str, repo: str | Path) -> str:
    """score_detector.py's idiom: relative manifest paths resolve against the repo root."""
    return p if Path(p).is_absolute() else str(Path(repo) / p)


# --- readouts (every one applies the finite-row mask first) ---------------------------------


def finite_rows(y, s) -> tuple[np.ndarray, dict]:
    """Mask of finite scores plus {n_used, n_dropped, status}: status is "ok", "empty" or
    "skipped_one_class"."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    info = {"n_used": int(m.sum()), "n_dropped": int((~m).sum())}
    if m.sum() == 0:
        info["status"] = "empty"
    elif np.unique(y[m]).size < 2:
        info["status"] = "skipped_one_class"
    else:
        info["status"] = "ok"
    return m, info


def direction_check(y, s) -> dict:
    """ok iff AUC(synth_logit, is_spoof) > 0.5 and the spoof median is above the bona fide
    median, on finite rows. One class or no rows -> ok None (the caller decides)."""
    m, info = finite_rows(y, s)
    if info["status"] != "ok":
        return {**info, "ok": None}
    y, s = np.asarray(y)[m].astype(int), np.asarray(s, dtype=np.float64)[m]
    auc = float(roc_auc_score(y, s))
    med_spoof, med_bona = float(np.median(s[y == 1])), float(np.median(s[y == 0]))
    ok = bool(auc > 0.5 and med_spoof > med_bona)
    return {**info, "status": "ok" if ok else "failed", "ok": ok, "auc": round(auc, 4),
            "median_synth_spoof": round(med_spoof, 3),
            "median_synth_bonafide": round(med_bona, 3)}  # fmt: skip


def masked_report(y, s, frame: pd.DataFrame | None = None, pi: float = PI_SYNTH) -> dict:
    """hearsay.metrics.report plus AUC on finite rows; by_group per generator / bona fide source
    when `frame` (generator, source columns) is given and has more than one value."""
    m, info = finite_rows(y, s)
    if info["status"] != "ok":
        return info
    y, s = np.asarray(y)[m].astype(int), np.asarray(s, dtype=np.float64)[m]
    out = {**info, **report(y, s, pi), "auc": round(float(roc_auc_score(y, s)), 4)}
    if frame is not None:
        per_gen, per_src = by_group(frame[m].reset_index(drop=True), y, s, pi)
        if len(per_gen) > 1:
            out["by_generator"] = per_gen
        if len(per_src) > 1:
            out["by_source"] = per_src
    return out


def sweep_thresholds(y, s, pi: float = PI_SYNTH) -> dict:
    """Thresholds from a labeled (inner) set: the minDCF-optimal one and the EER one, on the
    synth_logit scale, sklearn's ">= threshold is synthetic" convention. None plus a flag when
    the sweep never beats the constant decision (that covers roc_curve's +inf entry, whose cost
    is exactly the constant decision)."""
    m, info = finite_rows(y, s)
    if info["status"] != "ok":
        return {**info, "thr_dcf": None, "thr_eer": None}
    y, s = np.asarray(y)[m].astype(int), np.asarray(s, dtype=np.float64)[m]
    fpr, tpr, thr = roc_curve(y, s)
    c = C_FA * fpr * (1 - pi) + C_MISS * (1 - tpr) * pi
    const = min(C_FA * (1 - pi), C_MISS * pi)
    i = int(np.argmin(c))
    out = {**info, "thr_dcf": None, "thr_dcf_flag": "", "thr_eer": None, "thr_eer_flag": ""}
    if c[i] >= const:  # includes roc_curve's +inf entry at index 0, whose cost is the constant decision
        out["thr_dcf_flag"] = "no_threshold_beats_constant_decision"
    else:
        out["thr_dcf"] = round(float(thr[i]), 4)
    j = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    if np.isfinite(thr[j]):
        out["thr_eer"] = round(float(thr[j]), 4)
    else:
        out["thr_eer_flag"] = "eer_at_inf_entry"
    return out


def stress_readout(inner_y, inner_s, y, s, frame: pd.DataFrame | None = None,
                   provisional: bool = False) -> dict:  # fmt: skip
    """A labeled stress set read at the inner-row thresholds: minDCF, EER, AUC, by_group where
    applicable, and P_FA = mean(s_bona >= thr), P_miss = mean(s_spoof < thr) at thr_dcf and
    thr_eer from sweep_thresholds(inner_y, inner_s)."""
    thr = sweep_thresholds(inner_y, inner_s)
    out = {**masked_report(y, s, frame), "inner_thresholds": thr, "provisional": provisional,
           "at_inner_thresholds": {}}  # fmt: skip
    m, _ = finite_rows(y, s)
    y, s = np.asarray(y)[m].astype(int), np.asarray(s, dtype=np.float64)[m]
    for name in ("thr_dcf", "thr_eer"):
        t = thr.get(name)
        if t is None:
            out["at_inner_thresholds"][name] = None
            continue
        sb, ss = s[y == 0], s[y == 1]
        out["at_inner_thresholds"][name] = {
            "threshold": t,
            "p_fa": round(float(np.mean(sb >= t)), 4) if sb.size else None,
            "p_miss": round(float(np.mean(ss < t)), 4) if ss.size else None,
        }
    return out


def platt_diagnostic(y, s) -> dict:
    """Class-balanced Platt map fit on the rows given (inner rows only, by the caller).
    Diagnostic only: never applied to the export (run spec D6)."""
    m, info = finite_rows(y, s)
    if info["status"] != "ok":
        return {**info, "in_sample_diagnostic": True}
    from sklearn.linear_model import LogisticRegression

    y, s = np.asarray(y)[m].astype(int), np.asarray(s, dtype=np.float64)[m]
    lr = LogisticRegression(class_weight="balanced", C=1e6).fit(s[:, None], y)
    return {**info, "in_sample_diagnostic": True, "fit_on": "inner rows",
            "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}  # fmt: skip


def level_shortcut(peak, s, y) -> dict:
    """Within-class Spearman correlation between clip peak level and synth_logit. Spectra does not
    level-normalize its input, so a strong value would mean level is doing some of the work."""
    from scipy.stats import spearmanr

    peak, s, y = np.asarray(peak, float), np.asarray(s, float), np.asarray(y).astype(int)
    out, flag = {}, False
    for cls, name in ((0, "bonafide"), (1, "spoof")):
        m = np.isfinite(peak) & np.isfinite(s) & (y == cls)
        if m.sum() < 3 or np.ptp(peak[m]) == 0 or np.ptp(s[m]) == 0:
            out[name] = None
            continue
        r = float(spearmanr(peak[m], s[m]).statistic)
        out[name] = {"spearman": round(r, 4), "n": int(m.sum())}
        flag = flag or abs(r) > 0.3
    return {"within_class_spearman_peak_vs_synth_logit": out, "flag_abs_gt_0_3": flag}


# --- export and bookkeeping -----------------------------------------------------------------


def fusion_frame(path, fold, split, synth) -> pd.DataFrame:
    """path, fold, split, score = sigmoid(synth), logit = synth. No calibration (run spec D6);
    NaN stays NaN in both columns; duplicate paths raise."""
    synth = np.asarray(synth, dtype=np.float64)
    with np.errstate(over="ignore"):
        score = sigmoid(synth)
    df = pd.DataFrame({"path": list(path), "fold": list(fold), "split": list(split),
                       "score": score, "logit": synth})  # fmt: skip
    if not df.path.is_unique:
        raise ValueError("duplicate paths in the export")
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha(repo: str | Path = REPO) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
