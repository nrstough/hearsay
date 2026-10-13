"""Step 6 of docs/specs/2026-10-11_v2-benchmark-fusion.md: score the perturbation cohorts for the
columns that have no cohort scores yet. ASK-FIRST compute (Nathan): CPU ~35 min for the ITW cohort's
shipped columns, MPS ~30-40 min for WavLM and layer 20. One MPS job on the Mac at a time.

  uv run python scripts/v2_cohort_scores.py run --cohort itw --column handcrafted_v5 --pilot 20 --confirm-ask-first
  uv run python scripts/v2_cohort_scores.py run --cohort itw --column handcrafted_v5 --confirm-ask-first

Cohorts:
  itw      500 ITW rows (scripts/bakeoff_layers.py cohort_rows) x 5 kinds. Perturb the full decoded
           clip, then the seed-200 test-length crop, exactly as the stored clean ITW scores were made.
  holdout  500 holdout rows (scripts/m3_probes.py holdout_rows) x 5 kinds. Crop first (segment), then
           perturb, as outputs/channel/m3_perturb.csv was made.

Every column runs a pilot first: 20 'none' rows against the stored clean scores (max |dlogit| within
the device tolerance and Spearman >= 0.999). A failed pilot aborts that column. Output: one file per
(cohort, column), outputs/v2/benchmark/cohorts/<cohort>/<column>.csv with path, kind, logit, error
(Interface v1 cohort contract), plus a meta JSON.
"""

from __future__ import annotations

import os

MAX_THREADS = 4
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(min(int(os.environ.get(_v, str(MAX_THREADS)) or MAX_THREADS), MAX_THREADS))

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hearsay import benchmark as B

REPO = B.REPO
OUT = REPO / "outputs/v2/benchmark/cohorts"
ITW_MANIFEST = REPO / "outputs/embeddings/wav2vec2-xls-r-300m/itw_stress_v3/manifest.csv"
ITW_CROP_SEED = 200
PER_CLASS = 250
COLUMNS = {"itw": ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist", "wavlm_l", "xlsr_L20"),
           "holdout": ("wavlm_l", "xlsr_L20")}
DEVICE = {"m1b_v3": "cpu", "xlsr_L20": "cpu", "handcrafted_v5": "cpu", "m5_xlsr_ft": "cpu",
          "spectra_aasist": "cpu", "wavlm_l": "mps"}
TOLERANCE = {"cpu": 0.02, "mps": 0.05}
SPEARMAN_MIN = 0.999
BUSY = ("extract_embeddings|repair_decode_errors|score_spectra|m5_score|m3_probes|run_pipeline|bakeoff_layers|"
        "v2_backbone|v2_hardneg")
# Columns whose cohort scoring loads no torch model (librosa features; or the stored XLS-R cohort
# embeddings): they may run while another lane's GPU job is live. Everything else waits.
TORCH_FREE = {("itw", "handcrafted_v5"), ("itw", "m1b_v3"), ("itw", "xlsr_L20")}
EMBEDDING_FAST_PATH = {("itw", "m1b_v3"), ("itw", "xlsr_L20")}

_spec = importlib.util.spec_from_file_location("m3_probes", REPO / "scripts/m3_probes.py")
MP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MP)


# ---------------------------------------------------------------------------
# Pure pieces (tested hermetically)
# ---------------------------------------------------------------------------


def cap_threads(n: int) -> int:
    if n < 1:
        raise ValueError("threads must be >= 1")
    return min(int(n), MAX_THREADS)


def itw_cohort_rows(manifest: Path = ITW_MANIFEST) -> np.ndarray:
    """The 500 ITW rows of the bake-off cohort (bakeoff_layers.cohort_rows): 250 + 250, seed 0."""
    m = pd.read_csv(manifest)
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.flatnonzero(m.label == "bonafide"), PER_CLASS, replace=False),
                rng.choice(np.flatnonzero(m.label == "spoof"), PER_CLASS, replace=False)]
    return np.sort(idx)


def itw_crops(n_rows: int, seed: int = ITW_CROP_SEED) -> list:
    """The crop-length draw sequence over all ITW manifest rows (one sampler, drawn in order)."""
    from hearsay.embed import test_duration_sampler

    draw = test_duration_sampler(seed)
    return [draw() for _ in range(n_rows)]


def seeds_for(cohort: str, rows) -> np.ndarray:
    """Perturbation seeds: ITW = the manifest row index; holdout = position in holdout_rows(500)."""
    if cohort == "itw":
        return np.asarray(rows, dtype=int)
    if cohort == "holdout":
        return np.arange(len(rows))
    raise ValueError(f"unknown cohort {cohort!r}")


def perturb(x: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Exactly scripts/m3_probes.py perturb (the same function, not a copy)."""
    return MP.perturb(x, kind, int(seed))


def check_audio(x: np.ndarray, flag=None, path: str = "") -> None:
    """Refuse decode-error and silent rows: never score zeros (CLAUDE.md decode_error gotcha)."""
    if isinstance(flag, str) and flag == "decode_error":
        raise B.BenchmarkError(f"{path}: decode_error row refused")
    x = np.asarray(x)
    if x.size == 0 or not np.isfinite(x).all() or float(np.max(np.abs(x))) == 0.0:
        raise B.BenchmarkError(f"{path}: empty, non-finite or all-zero audio refused")


def assemble(records: list[dict], expected_keys, labels: dict) -> pd.DataFrame:
    """Records (path, kind, logit, error) into a complete cohort, or raise: every expected path
    under every kind, no errors, no duplicates, finite logits, 250 real + 250 spoof per kind."""
    df = pd.DataFrame(records, columns=["path", "kind", "logit", "error"])
    err = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
    if err.any():
        raise B.BenchmarkError(f"partial cohort: {int(err.sum())} row(s) failed, e.g. {df[err].iloc[0].to_dict()}")
    if df.duplicated(["path", "kind"]).any():
        raise B.BenchmarkError("duplicate (path, kind) rows")
    want = {(p, k) for p in expected_keys for k in B.COHORT_KINDS}
    have = set(zip(df["path"], df["kind"]))
    if have != want:
        raise B.BenchmarkError(f"partial cohort: {len(want - have)} missing, {len(have - want)} unexpected (path, kind)")
    if not np.isfinite(df["logit"].astype(float)).all():
        raise B.BenchmarkError("non-finite cohort logit")
    for kind in B.COHORT_KINDS:
        y = np.array([labels[p] for p in df.loc[df["kind"] == kind, "path"]])
        if int((y == 0).sum()) != PER_CLASS or int((y == 1).sum()) != PER_CLASS:
            raise B.BenchmarkError(f"kind {kind}: expected {PER_CLASS} real + {PER_CLASS} spoof")
    return df.sort_values(["kind", "path"]).reset_index(drop=True)


def write_cohort(df: pd.DataFrame, out: Path, meta: dict, *, force: bool = False) -> Path:
    out = Path(out)
    if out.exists() and not force:
        raise SystemExit(f"refusing to overwrite {out} (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    out.with_suffix(".meta.json").write_text(json.dumps(B.clean(meta), indent=1, sort_keys=True))
    return out


def pilot_parity(new, stored, device: str) -> dict:
    new, stored = np.asarray(new, dtype=float), np.asarray(stored, dtype=float)
    if new.shape != stored.shape or new.size < 2:
        raise B.BenchmarkError("pilot: arrays differ in shape or are too short")
    max_abs = float(np.max(np.abs(new - stored)))
    rho = float(spearmanr(new, stored).statistic)
    tol = TOLERANCE[device]
    return {"ok": bool(max_abs <= tol and rho >= SPEARMAN_MIN), "max_abs_dlogit": max_abs, "spearman": rho,
            "tolerance": tol, "spearman_min": SPEARMAN_MIN, "n": int(new.size)}


def busy_processes() -> list[str]:
    out = subprocess.run(["pgrep", "-fl", BUSY], capture_output=True, text=True, check=False).stdout
    return [line for line in out.splitlines() if line.strip() and str(os.getpid()) not in line.split()[0]]


# ---------------------------------------------------------------------------
# Scorers (heavy; exercised only by the pilot + run, never by hermetic tests)
# ---------------------------------------------------------------------------


def make_scorer(column: str, cohort: str, threads: int):
    """fn(x_perturbed_raw_or_segment, r) -> logit, using each column's original recipe."""
    from hearsay.embed import embed_segment, load_backbone, prepare_segment

    if column in ("handcrafted_v5", "m5_xlsr_ft", "spectra_aasist"):
        from hearsay.detectors._learned import predictor
        from hearsay.handcrafted import features
        from hearsay.pipeline import Models
        from hearsay.spectra import score_clip, synth_logit

        # load_deep=False skips the XLS-R probe AND Spectra/M5; load only the model this column needs
        models = Models(device="cpu", load_m5=(column == "m5_xlsr_ft"), threads=threads,
                        load_deep=False, load_spectra=(column == "spectra_aasist"))
        if column == "spectra_aasist":
            models.load_spectra_model()
        elif column == "m5_xlsr_ft":
            models.load_m5_model()
        crop = itw_crops(len(pd.read_csv(ITW_MANIFEST)))

        def seg(x, r):
            return prepare_segment(x, crop[r], ITW_CROP_SEED + int(r))

        if column == "spectra_aasist":
            return lambda x, r: float(synth_logit(score_clip(models.spectra, seg(x, r), ("zero",), 16)["zero"]))
        if column == "m5_xlsr_ft":
            return lambda x, r: float(models.m5_logit(seg(x, r)))
        b = models.handcrafted.bundle

        def hc(x, r):
            kw = {"crop_mode": b["crop_mode"], "band_match": b.get("band_match", False)}
            if b.get("families"):
                kw["families"] = tuple(b["families"])
            f = features(x, crop_s=crop[r], seed=ITW_CROP_SEED + int(r), **kw)
            v = np.array([f[c] for c in b["features"]], dtype=np.float32)
            p = float(np.clip(predictor(b).predict_proba(v[None, :])[0, 1], 1e-6, 1 - 1e-6))
            return float(np.log(p / (1 - p)))

        return hc
    import joblib

    from hearsay.probe import Probe

    if column == "wavlm_l":
        bb = load_backbone("wavlm-large", DEVICE[column])
        probe = Probe.load(REPO / "models/m1_wavlm-large_L9_20260926-1906")

        def emb_logit(e):
            return float(probe.llr(e.astype(np.float16).astype(np.float32)[None])[0])
    elif column in ("xlsr_L20", "m1b_v3"):
        bb = load_backbone("wav2vec2-xls-r-300m", "mps" if (cohort == "holdout" and column == "xlsr_L20") else "cpu")
        if column == "xlsr_L20":
            j = joblib.load(REPO / "outputs/v2/bakeoff/probes/L20.joblib")

            def emb_logit(e):
                v = e.astype(np.float16).astype(np.float32)[20][None]
                return float(j["a"] * j["clf"].decision_function(v)[0] + j["b"])
        else:
            probe = Probe.load(REPO / "models/m1_wav2vec2-xls-r-300m_L7_20260926-0521")

            def emb_logit(e):
                return float(probe.llr(e.astype(np.float16).astype(np.float32)[None])[0])
    else:
        raise B.BenchmarkError(f"unknown column {column!r}")
    if cohort == "itw":
        crop = itw_crops(len(pd.read_csv(ITW_MANIFEST)))
        return lambda x, r: emb_logit(embed_segment(bb, prepare_segment(x, crop[r], ITW_CROP_SEED + int(r))))
    return lambda x, r: emb_logit(embed_segment(bb, prepare_segment(x)))  # holdout: x is already segmented


def stored_clean(column: str) -> pd.Series:
    """The stored clean ITW logits the ITW pilot compares against, keyed by canonical path."""
    cols = {"handcrafted_v5": ("outputs/detector_scores/_itw_handcrafted_v5.csv", None),
            "m5_xlsr_ft": ("outputs/detector_scores/_itw_m5.csv", None),
            "spectra_aasist": ("outputs/spectra/itw_stress/scores.csv", None),
            "wavlm_l": ("outputs/detector_scores/wavlm_l.csv", "itw"),
            "m1b_v3": ("outputs/detector_scores/m1b_v3.csv", "itw"),
            "xlsr_L20": ("outputs/v2/benchmark/scores/xlsr_L20.csv", "itw")}
    path, split = cols[column]
    d = pd.read_csv(REPO / path)
    if split:
        d = d[d["split"] == split]
    return pd.Series(d["logit"].to_numpy(float), index=[B.canonical_key(x) for x in d["path"]])


def holdout_pilot_reference() -> pd.Series:
    """The holdout cohort has no stored segment score for WavLM or layer 20, so its pilot checks the
    shared segment recipe through layer 7: outputs/channel/m3_perturb.csv m1b_v3, kind 'none' (CPU)."""
    d = pd.read_csv(REPO / "outputs/channel/m3_perturb.csv")
    d = d[d["kind"] == "none"]
    return pd.Series(d["m1b_v3"].to_numpy(float), index=[B.canonical_key(x) for x in d["path"]])


def cohort_inputs(cohort: str):
    """(keys, rows-for-seeds, labels by key, loader(key, seed) -> raw x before perturbation)."""
    from hearsay.audio import load_audio

    pops = B.load_populations()
    if cohort == "itw":
        m = pd.read_csv(ITW_MANIFEST)
        rows = itw_cohort_rows()
        keys = [B.canonical_key(p) for p in m["path"].iloc[rows]]
        y = dict(zip(pops["itw"].keys, pops["itw"].y))

        def load(i):
            return load_audio(REPO / keys[i])
        return keys, rows, {k: int(y[k]) for k in keys}, load
    MP.TEST_DURS = MP._test_durs()  # the module default is [3.4]; run_scoring resets it the same way
    h = MP.holdout_rows(500)
    keys = [B.canonical_key(p) for p in h["path"]]
    y = dict(zip(pops["holdout"].keys, pops["holdout"].y))

    def load(i):
        return MP.segment(h["path"].iloc[i], i)
    return keys, np.arange(len(keys)), {k: int(y[k]) for k in keys}, load


def embedding_logits(column: str) -> dict:
    """ITW cohort logits for m1b_v3 / xlsr_L20 from the stored bake-off cohort embeddings
    (outputs/v2/bakeoff/emb_itw_<kind>.npz: the same perturb -> seed-200 crop -> embed_segment recipe),
    fp16-rounded as the deployed probes expect. {kind: (rows, logits)}."""
    import joblib

    from hearsay.probe import Probe

    out = {}
    for kind in B.COHORT_KINDS:
        z = np.load(REPO / f"outputs/v2/bakeoff/emb_itw_{kind}.npz")
        e = z["emb"].astype(np.float16).astype(np.float32)
        if column == "xlsr_L20":
            j = joblib.load(REPO / "outputs/v2/bakeoff/probes/L20.joblib")
            lg = j["a"] * j["clf"].decision_function(e[:, 20]) + j["b"]
        else:
            lg = Probe.load(REPO / "models/m1_wav2vec2-xls-r-300m_L7_20260926-0521").llr(e)
        out[kind] = (z["row"], np.asarray(lg, dtype=float))
    return out


def run_from_embeddings(column: str, *, pilot: int, force: bool) -> dict:
    keys, rows, labels, _ = cohort_inputs("itw")
    got = embedding_logits(column)
    for kind, (r, _) in got.items():
        if not np.array_equal(r, rows):
            raise B.BenchmarkError(f"emb_itw_{kind}.npz rows differ from the cohort rows")
    stored = stored_clean(column)
    n = min(pilot or 20, len(keys))
    par = pilot_parity(got["none"][1][:n], [stored[k] for k in keys[:n]], "cpu")
    meta = {"cohort": "itw", "column": column, "device": "cpu (stored cohort embeddings)",
            "source": "outputs/v2/bakeoff/emb_itw_<kind>.npz", "pilot": {**par, "checked": column},
            "code_sha": B.code_sha(), "crop_seed": ITW_CROP_SEED, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(B.clean(par), sort_keys=True))
    if not par["ok"]:
        raise SystemExit(f"pilot parity failed for {column} on itw: not writing the cohort")
    if pilot:
        return meta
    records = [{"path": k, "kind": kind, "logit": float(lg[i]), "error": None}
               for kind, (_, lg) in got.items() for i, k in enumerate(keys)]
    write_cohort(assemble(records, keys, labels), OUT / "itw" / f"{column}.csv", meta, force=force)
    return meta


def run(cohort: str, column: str, *, pilot: int, threads: int, force: bool) -> dict:
    busy = busy_processes()
    if busy and (cohort, column) not in TORCH_FREE:
        raise SystemExit("other heavy jobs are running; wait or ask Nathan first:\n" + "\n".join(busy))
    if (cohort, column) in EMBEDDING_FAST_PATH:
        return run_from_embeddings(column, pilot=pilot, force=force)
    keys, rows, labels, load = cohort_inputs(cohort)
    seeds = seeds_for(cohort, rows)
    score = make_scorer(column, cohort, cap_threads(threads))
    device = DEVICE[column]
    if cohort == "holdout":
        pscore, stored, pdevice = make_scorer("m1b_v3", "holdout", cap_threads(threads)), holdout_pilot_reference(), "cpu"
    else:
        pscore, stored, pdevice = score, stored_clean(column), device
    t0 = time.time()
    n_pilot = min(pilot or 20, len(keys))
    pilot_new, pilot_old = [], []
    for i in range(n_pilot):
        x = load(i)
        check_audio(x, path=keys[i])
        pilot_new.append(pscore(perturb(x, "none", seeds[i]), int(rows[i]) if cohort == "itw" else i))
        pilot_old.append(stored[keys[i]])
    par = pilot_parity(pilot_new, pilot_old, pdevice)
    meta = {"cohort": cohort, "column": column, "device": device, "threads": cap_threads(threads),
            "pilot": {**par, "checked": "m1b_v3 segment recipe" if cohort == "holdout" else column},
            "code_sha": B.code_sha(), "crop_seed": ITW_CROP_SEED if cohort == "itw" else "segment",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(B.clean(par), sort_keys=True))
    if not par["ok"]:
        raise SystemExit(f"pilot parity failed for {column} on {cohort}: not running the full cohort")
    if pilot:
        return meta
    records = []
    for i, key in enumerate(keys):
        x = load(i)
        check_audio(x, path=key)
        for kind in B.COHORT_KINDS:
            try:
                logit = score(perturb(x, kind, seeds[i]), int(rows[i]) if cohort == "itw" else i)
                records.append({"path": key, "kind": kind, "logit": logit, "error": None})
            except Exception as exc:  # noqa: BLE001 - recorded, then refused by assemble: never a silent zero
                records.append({"path": key, "kind": kind, "logit": np.nan, "error": repr(exc)})
        if i % 50 == 0:
            print(f"  {column} {cohort} {i}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    df = assemble(records, keys, labels)
    meta["seconds"] = round(time.time() - t0, 1)
    write_cohort(df, OUT / cohort / f"{column}.csv", meta, force=force)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cohort", choices=sorted(COLUMNS), required=True)
    r.add_argument("--column", required=True)
    r.add_argument("--pilot", type=int, default=0, help="score only N 'none' rows and report parity")
    r.add_argument("--threads", type=int, default=MAX_THREADS)
    r.add_argument("--force", action="store_true")
    r.add_argument("--confirm-ask-first", action="store_true", help="Nathan approved this compute")
    a = ap.parse_args(argv)
    if a.column not in COLUMNS[a.cohort]:
        raise SystemExit(f"{a.column} is not scored on the {a.cohort} cohort by this step")
    if not a.confirm_ask_first:
        raise SystemExit("ask-first compute: rerun with --confirm-ask-first once Nathan has approved it")
    print(json.dumps(B.clean(run(a.cohort, a.column, pilot=a.pilot, threads=a.threads, force=a.force)),
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
