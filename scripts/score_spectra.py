"""Score the NSA fold file and test set with Spectra-AASIST and export the M3 fusion stream.

Every row goes through hearsay.spectra.prepare_input (= hearsay.embed.prepare_segment: band
match, silence trim, optional test-length crop, cap 8 s), then short-clip handling in one pad
mode, 64,600-sample windows, pre-emphasis and the model. Fold rows (inner and holdout) get the
crop draws of `extract_embeddings --segment --crop test --seed 0` (one draw per
outputs/manifests/nsa_train_sample.csv row in order, offset seed = row index), so fusion sees the
same audio per file as M1 and the handcrafted detector. Test rows are scored whole.

Modes
  full (default)   splits/nsa_folds.csv + outputs/manifests/nsa_test.csv ->
                   models/m3_spectra_<stamp>/{raw.csv, fusion.csv, meta.json}, then published to
                   outputs/detector_scores/<out-name>.csv (+ <out-name>_summary.json) and
                   outputs/spectra/<out-name>_raw.csv. Gates: per-split failure rate <= 5%,
                   direction on inner rows. logit = raw synth_logit, score = sigmoid(logit).
  --limit N        pilot on the first N fold rows (inner rows only) and --test-limit M test rows,
                   every --pad-modes mode in one decode pass; outputs/spectra/pilot/<stamp>_<name>/.
  --manifest PATH  a labeled stress set (path, label[, generator, source]) with --stress-seed
                   crops; outputs/spectra/<name>/{scores.csv, meta.json}.
  --readout        inner=<scores.csv> stress=<name>=<scores.csv>: stress readout at the inner
                   thresholds from saved scores, no inference; --out <json>.

Exit codes: 0 ok; 2 usage/config (missing labels, path not in the crop plan, stale or missing
partial); 3 failure gate or consecutive decode errors; 4 direction gate; 5 stopped by --max-chunks.

Usage:
  nice -n 10 uv run python scripts/score_spectra.py --limit 400 --test-limit 200 \
      --pad-modes repeat,zero,whole --device cpu --threads 4
  nice -n 10 uv run python scripts/score_spectra.py --manifest outputs/manifests/itw_stress.csv \
      --name itw_stress --pad-mode zero --device mps
  nohup nice -n 10 uv run python scripts/score_spectra.py --pad-mode zero --device mps \
      --stress itw=outputs/spectra/itw_stress/scores.csv > outputs/spectra/score_spectra.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from hearsay import SR
from hearsay.audio import DecodeError, load_audio
from hearsay.embed import default_device
from hearsay.spectra import (
    PAD_MODES,
    crop_plan,
    direction_check,
    file_sha256,
    fusion_frame,
    git_sha,
    level_shortcut,
    load_spectra,
    masked_report,
    peak_normalize,
    platt_diagnostic,
    prepare_input,
    resolve_path,
    score_clip,
    stress_readout,
    sweep_thresholds,
)

REPO = Path(__file__).resolve().parents[1]
BASE_COLS = ["path", "fold", "split", "score", "logit"]
LABEL_COLS = ["label", "generator", "source"]
RAW_COLS = ["logit_spoof", "logit_bonafide", "synth_logit", "n_windows", "n_samples",
            "n_pad_samples", "crop_s", "peak", "flag", "seconds"]  # fmt: skip
FAIL_RATE = 0.05
CAVEATS = {
    "in_the_wild_possibly_optimistic": True,
    "reason_in_the_wild": "the model card lists In-the-Wild only as an evaluation set (authors' EER 1.46%) "
                          "and does not disclose the training data, so the stress readout cannot be shown "
                          "to be uncontaminated.",
    "inner_rows_possibly_in_sample": True,
    "reason": "Spectra-AASIST's training data is undisclosed; LJ Speech, LibriSpeech and DiffSSD are "
              "public and plausible training corpora, and the inner rows score AUC 1.0. Treat the "
              "inner_oof column as in-sample for this detector, not as out-of-fold evidence; the "
              "holdout, the test-set share and In-the-Wild are the readings that do not depend on it.",
}
INPUT_PATH_SOURCES = ("spectra.py", "embed.py", "audio.py", "handcrafted.py")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--folds", type=Path, default=Path("splits/nsa_folds.csv"))
    ap.add_argument("--test-manifest", type=Path, default=Path("outputs/manifests/nsa_test.csv"))
    ap.add_argument("--train-manifest", type=Path,
                    default=Path("outputs/manifests/nsa_train_sample.csv"),
                    help="crop-draw order (same paths as the fold file)")  # fmt: skip
    ap.add_argument("--device", default=None, help="default: mps if available else cpu")
    ap.add_argument("--pad-mode", choices=PAD_MODES, default="zero")
    ap.add_argument("--pad-modes", help="pilot only: comma list scored in one pass")
    ap.add_argument("--band-match", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--peak-norm", action=argparse.BooleanOptionalAction, default=False,
                    help="scale every prepared clip to peak 0.95 (level is not a class cue)")  # fmt: skip
    ap.add_argument("--max-windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="crop-draw seed for fold rows")
    ap.add_argument("--limit", type=int, help="pilot: first N fold rows (inner rows only)")
    ap.add_argument("--test-limit", type=int, help="pilot: first N test rows (default --limit)")
    ap.add_argument("--manifest", type=Path, help="labeled stress set")
    ap.add_argument("--name", help="pilot / manifest run name")
    ap.add_argument("--stress-seed", type=int, default=200, help="crop-draw seed for --manifest")
    ap.add_argument("--stress", action="append", default=[], metavar="NAME=SCORES.csv",
                    help="stress scores read out at the inner thresholds (repeatable)")  # fmt: skip
    ap.add_argument("--readout", nargs="+", metavar="KEY=PATH",
                    help="inner=<scores.csv> stress=<name>=<scores.csv>; no inference")  # fmt: skip
    ap.add_argument("--out", type=Path, help="--readout output json")
    ap.add_argument("--merge-into", type=Path, help="--readout: also write the readout into this meta.json")
    ap.add_argument("--final", action="store_true",
                    help="--readout: the inner scores are the full run's (not a pilot); unsets provisional")  # fmt: skip
    ap.add_argument("--out-name", default="spectra_aasist")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-chunks", type=int, help="stop after N chunks, keep the partial (exit 5)")
    ap.add_argument("--prefetch", type=int, default=4, help="decode threads")
    ap.add_argument("--chunk", type=int, default=500, help="rows per decode chunk and checkpoint")
    ap.add_argument("--threads", type=int, default=4, help="torch threads on cpu")
    ap.add_argument("--consecutive-decode-abort", type=int, default=25)
    return ap.parse_args(argv)


def _die(msg: str) -> None:
    """Config / usage problem: message on stderr, exit code 2."""
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(2)


def _under(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M")


# --- rows -----------------------------------------------------------------------------------


def build_rows(args, root: Path) -> tuple[pd.DataFrame, str]:
    """rows: path, fold, split, label, generator, source, resolved, crop_s, seed. Second value is
    the mode: full, pilot or manifest. Raises SystemExit(2) on a config problem."""
    if args.manifest is not None:
        m = pd.read_csv(_under(root, args.manifest))
        if "label" not in m.columns:
            _die("--manifest needs a `label` column (spoof | bonafide)")
        m["label"] = m["label"].astype(str).str.lower().str.replace("-", "", regex=False)
        bad = set(m["label"]) - {"spoof", "bonafide"}
        if bad:
            _die(f"--manifest labels must be spoof | bonafide, got {sorted(bad)}")
        for c in ("generator", "source"):
            if c not in m.columns:
                m[c] = np.where(m.label == "spoof", "spoof", "bonafide") if c == "generator" else "stress"
        plan = crop_plan(_under(root, args.manifest), args.stress_seed)
        rows = m[["path", "label", "generator", "source"]].assign(fold="stress", split="stress")
        if args.limit is not None:
            rows = rows.iloc[: args.limit]
        mode = "manifest"
    else:
        f = pd.read_csv(_under(root, args.folds))
        t = pd.read_csv(_under(root, args.test_manifest))
        plan = crop_plan(_under(root, args.train_manifest), args.seed)
        missing = [p for p in f.path if p not in plan]
        if missing:
            _die(f"{len(missing)} fold-file paths are not in {args.train_manifest} "
                     f"(first: {missing[0]})")  # fmt: skip
        f = f.assign(split=np.where(f.fold == "holdout", "holdout", "inner_oof"))
        if args.limit is not None:
            f = f.iloc[: args.limit]
            f = f[f.split != "holdout"]  # the holdout is read once, after the full run
            t = t.iloc[: (args.limit if args.test_limit is None else args.test_limit)]
            mode = "pilot"
        else:
            mode = "full"
        t = t.assign(fold="test", split="test", label=np.nan, generator=np.nan, source=np.nan)
        rows = pd.concat([f[["path", "fold", "split", "label", "generator", "source"]],
                          t[["path", "fold", "split", "label", "generator", "source"]]],
                         ignore_index=True)  # fmt: skip
    rows = rows.reset_index(drop=True)
    if len(rows) == 0:
        _die("no rows to score (check --limit / --test-limit / the manifest)")
    rows["resolved"] = [resolve_path(p, root) for p in rows.path]
    crops = [plan.get(p) if s != "test" else None for p, s in zip(rows.path, rows.split, strict=True)]
    rows["crop_s"] = [np.nan if c is None else c[0] for c in crops]
    rows["seed"] = [np.nan if c is None else c[1] for c in crops]
    if not rows.path.is_unique:
        _die("duplicate paths across the fold file and the test manifest")
    return rows, mode


def config_hash(args, root: Path, mode: str, pad_modes: tuple[str, ...]) -> str:
    """Everything that determines a row's score or its crop; a stale partial never merges."""
    files = [args.folds, args.test_manifest, args.train_manifest, Path("splits/nsa_test_durations.csv")]
    if args.manifest is not None:
        files = [args.manifest, Path("splits/nsa_test_durations.csv")]
    src = Path(__file__).resolve().parents[1] / "src" / "hearsay"
    parts = {"mode": mode, "pad_modes": list(pad_modes), "band_match": args.band_match,
             "peak_norm": args.peak_norm,
             "max_windows": args.max_windows, "seed": args.seed, "stress_seed": args.stress_seed,
             "device": args.device, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
             "files": {str(p): file_sha256(_under(root, p)) for p in files if _under(root, p).exists()},
             "sources": {n: file_sha256(src / n) for n in INPUT_PATH_SOURCES if (src / n).exists()},
             "script": file_sha256(__file__)}  # fmt: skip
    w = Path(__file__).resolve().parents[1] / "weights" / "Spectra-AASIST"
    for n in ("model.py", "config.json"):
        if (w / n).exists():
            parts[n] = file_sha256(w / n)
    st = w / "model.safetensors"
    if st.exists():
        parts["safetensors"] = [st.stat().st_size, int(st.stat().st_mtime)]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# --- scoring loop ---------------------------------------------------------------------------


def _record(r, pad_modes) -> dict:
    rec = {"path": r.path, "fold": r.fold, "split": r.split, "label": r.label,
           "generator": r.generator, "source": r.source, "crop_s": r.crop_s,
           "peak": np.nan, "flag": "", "n_windows": 0, "n_samples": 0, "seconds": 0.0}  # fmt: skip
    for m in pad_modes:
        rec[f"logit_spoof_{m}"] = rec[f"logit_bonafide_{m}"] = rec[f"synth_logit_{m}"] = np.nan
        rec[f"n_pad_samples_{m}"] = 0
    return rec


def score_rows(model, rows: pd.DataFrame, pad_modes, args, partial: Path | None,
               done: dict[str, dict]) -> tuple[dict[str, dict], int]:  # fmt: skip
    """Score the rows not in `done`, chunk by chunk (decode in threads, model on this thread),
    checkpointing after each chunk. Returns (results by path, exit code: 0, 3 or 5)."""
    from hearsay.handcrafted import band_limit

    band_limit(np.zeros(SR, dtype=np.float32))  # warm the lazy FIR once, on this thread
    todo = [i for i, p in enumerate(rows.path) if p not in done]
    results = dict(done)
    t0, consecutive, n_done = time.time(), 0, 0

    def prep(i: int):
        r = rows.iloc[i]
        try:
            x = load_audio(r.resolved)
        except DecodeError:
            return i, None, np.nan, "decode_error"
        peak = float(np.abs(x).max()) if x.size else 0.0
        crop = None if pd.isna(r.crop_s) else float(r.crop_s)
        seed = None if pd.isna(r.seed) else int(r.seed)
        x = prepare_input(x, crop, seed, args.band_match)
        if args.peak_norm:
            x = peak_normalize(x)
        return i, x, peak, ""

    def checkpoint() -> None:
        if partial is not None:
            frame = pd.DataFrame([results[p] for p in rows.path if p in results])
            tmp = partial.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            os.replace(tmp, partial)

    for n_chunks, c0 in enumerate(range(0, len(todo), args.chunk), start=1):
        idx = todo[c0 : c0 + args.chunk]
        with ThreadPoolExecutor(args.prefetch) as pool:
            for i, x, peak, flag in pool.map(prep, idx):
                r = rows.iloc[i]
                rec = _record(r, pad_modes)
                rec["peak"], rec["flag"] = peak, flag
                if flag == "decode_error":
                    consecutive += 1
                    if consecutive >= args.consecutive_decode_abort:
                        results[r.path] = rec
                        checkpoint()
                        print(f"ABORT: {consecutive} consecutive decode errors at {r.path}; "
                              "is the data drive mounted?", flush=True)  # fmt: skip
                        return results, 3
                else:
                    consecutive = 0
                    t1 = time.time()
                    try:
                        out = score_clip(model, x, pad_modes, args.max_windows)
                        rec["n_windows"], rec["n_samples"] = out["n_windows"], out["n_samples"]
                        for m in pad_modes:
                            rec[f"logit_spoof_{m}"], rec[f"logit_bonafide_{m}"] = float(out[m][0]), float(out[m][1])
                            rec[f"synth_logit_{m}"] = float(out[m][0] - out[m][1])
                            rec[f"n_pad_samples_{m}"] = out[f"n_pad_samples_{m}"]
                    except RuntimeError as e:
                        rec["flag"], rec["error"] = "model_error", str(e)[:200]
                    rec["seconds"] = round(time.time() - t1, 4)
                results[r.path] = rec
                n_done += 1
        checkpoint()
        el = time.time() - t0
        rate = el / max(n_done, 1)
        eta = rate * (len(todo) - n_done)
        print(f"  {n_done}/{len(todo)}  {el:.0f}s  {rate:.3f}s/clip  eta {eta / 60:.1f} min", flush=True)
        if args.max_chunks and n_chunks >= args.max_chunks and n_done < len(todo):
            print(f"stopped after {n_chunks} chunk(s) (--max-chunks); partial kept", flush=True)
            return results, 5
    return results, 0


def mode_frame(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One pad mode's rows in the raw-companion layout: base columns, labels, raw columns."""
    synth = raw[f"synth_logit_{mode}"].to_numpy(dtype=np.float64)
    fus = fusion_frame(raw.path, raw.fold, raw.split, synth)
    out = fus.assign(**{c: raw[c].to_numpy() for c in LABEL_COLS})
    out["logit_spoof"] = raw[f"logit_spoof_{mode}"].to_numpy()
    out["logit_bonafide"] = raw[f"logit_bonafide_{mode}"].to_numpy()
    out["synth_logit"] = synth
    out["n_pad_samples"] = raw[f"n_pad_samples_{mode}"].to_numpy()
    for c in ("n_windows", "n_samples", "crop_s", "peak", "flag", "seconds"):
        out[c] = raw[c].to_numpy()
    return out[BASE_COLS + LABEL_COLS + RAW_COLS]


# --- readouts -------------------------------------------------------------------------------


def _y(frame: pd.DataFrame) -> np.ndarray:
    return (frame.label == "spoof").to_numpy(dtype=int)


def failure_table(raw: pd.DataFrame) -> dict:
    out = {"rows": len(raw), "ok": int((raw.flag == "").sum()),
           "decode_error": int((raw.flag == "decode_error").sum()),
           "model_error": int((raw.flag == "model_error").sum()), "by_split": {}}  # fmt: skip
    for split, g in raw.groupby("split"):
        out["by_split"][split] = {"n": len(g), "not_ok": int((g.flag != "").sum()),
                                  "rate": round(float((g.flag != "").mean()), 4)}  # fmt: skip
    lab = raw[raw.label.isin(["spoof", "bonafide"])]
    if len(lab):
        out["fold_rows_not_ok_by_class"] = {k: int(v) for k, v in
                                            (lab.flag != "").groupby(lab.label).sum().items()}  # fmt: skip
        out["fold_rows_not_ok_by_generator"] = {k: int(v) for k, v in
                                                (lab.flag != "").groupby(lab.generator).sum().items()
                                                if v}  # fmt: skip
    return out


def split_readouts(sc: pd.DataFrame) -> dict:
    """Holdout, inner (diagnostic), test, direction, thresholds, Platt, level for one mode."""
    inner, hold, test = (sc[sc.split == s] for s in ("inner_oof", "holdout", "test"))
    out = {}
    if len(hold):
        out["holdout"] = masked_report(_y(hold), hold.synth_logit, hold[["generator", "source"]])
    if len(inner):
        out["inner"] = masked_report(_y(inner), inner.synth_logit, inner[["generator", "source"]])
        out["direction"] = direction_check(_y(inner), inner.synth_logit)
        out["thresholds"] = sweep_thresholds(_y(inner), inner.synth_logit)
        out["platt_diagnostic"] = platt_diagnostic(_y(inner), inner.synth_logit)
        out["level_shortcut"] = level_shortcut(inner.peak, inner.synth_logit, _y(inner))
    if len(test):
        s = test.synth_logit.to_numpy(dtype=np.float64)
        lb = test.logit_bonafide.to_numpy(dtype=np.float64)
        m = np.isfinite(s)
        t = {"n": len(test), "n_used": int(m.sum()),
             "frac_above_half": round(float(np.mean(s[m] > 0)), 4) if m.any() else None,
             "frac_bonafide_le_card_threshold": round(float(np.mean(lb[m] <= -1.0625009)), 4) if m.any() else None,
             "score_mean": round(float(np.mean(test.score[m])), 4) if m.any() else None,
             "score_std": round(float(np.std(test.score[m])), 4) if m.any() else None,
             "distinct_scores": int(pd.Series(test.score[m]).nunique())}  # fmt: skip
        pl = out.get("platt_diagnostic", {})
        if "a" in pl and m.any():
            t["frac_above_half_platt_diagnostic"] = round(float(np.mean(pl["a"] * s[m] + pl["b"] > 0)), 4)
        out["test"] = t
    return out


def pick_mode(table: dict, tie_band: float = 0.02, fa_flag: float = 0.6) -> dict:
    """inner minDCF, then inner AUC, then `zero`; a test share above `fa_flag` is flagged and
    the next candidate preferred (both recorded)."""
    scored = {m: v for m, v in table.items() if v.get("inner", {}).get("status") == "ok"}
    if not scored:
        return {"pick": None, "reason": "no mode has a two-class inner readout"}
    best = min(v["inner"]["min_dcf"] for v in scored.values())
    cands = [m for m, v in scored.items() if v["inner"]["min_dcf"] <= best + tie_band]
    order = sorted(cands, key=lambda m: (-scored[m]["inner"]["auc"], m != "zero"))
    flagged = [m for m in order if (scored[m].get("test", {}).get("frac_above_half") or 0) > fa_flag]
    clean = [m for m in order if m not in flagged]
    pick = (clean or order)[0]
    return {"pick": pick, "candidates_within_tie_band": cands, "order": order,
            "flagged_test_share_above": fa_flag, "flagged": flagged,
            "rule": "inner minDCF, ties within 0.02 by inner AUC, then zero"}  # fmt: skip


def do_readout(args, root: Path) -> int:
    kv = dict(s.split("=", 1) for s in args.readout)
    if "inner" not in kv:
        _die("--readout needs inner=<scores.csv>")
    inner = pd.read_csv(_under(root, Path(kv["inner"])))
    inner = inner[inner.split == "inner_oof"] if "split" in inner.columns else inner
    out = {"inner_scores": kv["inner"], "n_inner": len(inner), "stress": {}}
    for k, v in kv.items():
        if k != "stress":
            continue
        name, path = v.split("=", 1)
        st = pd.read_csv(_under(root, Path(path)))
        out["stress"][name] = stress_readout(_y(inner), inner.synth_logit, _y(st), st.synth_logit,
                                             st[["generator", "source"]],
                                             provisional=not args.final)  # fmt: skip
    if "pilot" in kv:  # the pad-mode pilot's table and selection, carried into the full run's meta
        pm = json.loads(_under(root, Path(kv["pilot"])).read_text())
        out["pilot"] = {"run": kv["pilot"], "table": pm.get("pilot"), "selection": pm.get("selection")}
    if "m1_tsv" in kv and "test" in kv:
        out["m1_disagreement"] = m1_disagreement(_under(root, Path(kv["test"])), _under(root, Path(kv["m1_tsv"])))
    out["caveats"] = CAVEATS
    text = json.dumps(out, indent=2)
    if args.out:
        p = _under(root, args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(p, text)
    if args.merge_into:
        mp = _under(root, args.merge_into)
        meta = json.loads(mp.read_text())
        meta["readout"] = out
        meta["caveats"] = CAVEATS
        _atomic_write_text(mp, json.dumps(meta, indent=2))
    print(text)
    return 0


def m1_disagreement(test_scores: Path, m1_tsv: Path) -> dict:
    """Where M3 and an M1 submission disagree on the NSA test set: Spearman of the two score
    columns (rank-based, so the M1 probability and the Spectra margin compare directly) and the
    files one calls synthetic at its own 0.5 / zero-margin point and the other does not."""
    from scipy.stats import spearmanr

    t = pd.read_csv(test_scores)
    t = t[t.split == "test"].copy() if "split" in t.columns else t
    t["filename"] = [Path(p).name for p in t.path]
    m1 = pd.read_csv(m1_tsv, sep="\t")
    j = t.merge(m1, on="filename", how="inner")
    m = np.isfinite(j.synth_logit.to_numpy()) & np.isfinite(j["cm-score"].to_numpy())
    j = j[m]
    m3_syn, m1_syn = j.synth_logit > 0, j["cm-score"] > 0.5
    return {"m1_tsv": str(m1_tsv), "n_joined": len(j),
            "spearman": round(float(spearmanr(j.synth_logit, j["cm-score"]).statistic), 4),
            "m3_synthetic_share": round(float(m3_syn.mean()), 4),
            "m1_synthetic_share": round(float(m1_syn.mean()), 4),
            "both_synthetic": int((m3_syn & m1_syn).sum()), "both_real": int((~m3_syn & ~m1_syn).sum()),
            "m3_only_synthetic": int((m3_syn & ~m1_syn).sum()),
            "m1_only_synthetic": int((~m3_syn & m1_syn).sum()),
            "note": "0.5 on the M1 probability, zero margin on Spectra; neither is a fusion threshold"}  # fmt: skip


# --- main -----------------------------------------------------------------------------------


def run(argv=None, loader=load_spectra) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    if args.readout:
        return do_readout(args, root)
    args.device = args.device or default_device()
    rows, mode = build_rows(args, root)
    pad_modes = tuple(args.pad_modes.split(",")) if (mode == "pilot" and args.pad_modes) else (args.pad_mode,)
    for m in pad_modes:
        if m not in PAD_MODES:
            _die(f"unknown pad mode {m!r}")
    chash = config_hash(args, root, mode, pad_modes)
    stamp = _stamp()

    if mode == "full":
        run_dir = root / "models" / f"m3_spectra_{stamp}"
        k = 2
        while run_dir.exists():  # two runs inside the same minute must not share a run dir
            run_dir = root / "models" / f"m3_spectra_{stamp}-{k}"
            k += 1
        partial = root / "outputs" / "spectra" / f"{args.out_name}_partial.csv"
    elif mode == "pilot":
        run_dir = root / "outputs" / "spectra" / "pilot" / f"{stamp}_{args.name or 'pilot'}"
        k = 2
        while run_dir.exists():
            run_dir = root / "outputs" / "spectra" / "pilot" / f"{stamp}_{args.name or 'pilot'}-{k}"
            k += 1
        partial = run_dir / "partial.csv"
    else:
        run_dir = root / "outputs" / "spectra" / (args.name or args.manifest.stem)
        partial = run_dir / "partial.csv"
    sidecar = partial.with_suffix(".json")

    done: dict[str, dict] = {}
    if args.resume:
        if not partial.exists() or not sidecar.exists():
            _die(f"nothing to resume: {partial} not found")
        if json.loads(sidecar.read_text()).get("config_hash") != chash:
            _die(f"stale partial {partial}: configuration changed; delete it or drop --resume")
        old = pd.read_csv(partial, float_precision="round_trip")
        old["flag"] = old["flag"].fillna("")
        keep = old[old.flag.isin(["", "decode_error"])]
        done = {r["path"]: r for r in keep.to_dict("records")}
        print(f"resuming: {len(done)} rows kept from {partial}", flush=True)
    elif partial.exists() and mode == "full":
        _die(f"{partial} exists from an earlier run; pass --resume or delete it")

    run_dir.mkdir(parents=True, exist_ok=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(sidecar, json.dumps({"config_hash": chash, "mode": mode, "stamp": stamp}))

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    t_load = time.time()
    model = loader(args.device)
    print(f"model on {args.device} in {time.time() - t_load:.0f}s; {len(rows)} rows, "
          f"{len(rows) - len(done)} to score, modes {pad_modes}", flush=True)  # fmt: skip
    t0 = time.time()
    results, code = score_rows(model, rows, pad_modes, args, partial, done)
    if code:
        return code
    raw = pd.DataFrame([results[p] for p in rows.path])
    raw["flag"] = raw["flag"].fillna("")
    seconds = round(time.time() - t0)

    fails = failure_table(raw)
    meta = {"rung": "M3", "mode": mode, "stamp": stamp, "config_hash": chash, "git_sha": git_sha(root),
            "caveats": CAVEATS,
            "config": {"pad_modes": list(pad_modes), "band_match": args.band_match, "peak_norm": args.peak_norm,
                       "max_windows": args.max_windows, "seed": args.seed, "stress_seed": args.stress_seed,
                       "device": args.device, "torch": torch.__version__,
                       "crop_policy": "fold/stress rows: test-duration draws in manifest order, "
                                      "offset seed = seed + row; test rows whole; cap 8 s"},
            "failures": fails, "seconds": seconds,
            "sec_per_clip": round(seconds / max(1, len(rows) - len(done)), 4)}  # fmt: skip
    over = {s: v for s, v in fails["by_split"].items() if v["rate"] > FAIL_RATE}
    if over:
        (raw if len(pad_modes) > 1 else mode_frame(raw, pad_modes[0])).to_csv(run_dir / "raw.csv", index=False)
        meta["gate"] = {"failure_rate": "failed", "splits_over_5pct": over}
        _atomic_write_text(run_dir / "meta.json", json.dumps(meta, indent=2))
        print(f"FAILURE GATE: splits over {FAIL_RATE:.0%} not-ok: {over}; nothing published", flush=True)
        return 3

    if mode == "pilot":
        table = {}
        for m in pad_modes:
            sc = mode_frame(raw, m)
            sc.to_csv(run_dir / f"scores_{m}.csv", index=False)
            table[m] = split_readouts(sc)
            short = sc[sc.n_samples < 64_600]
            table[m]["short_clips"] = len(short)
            table[m]["padded_clips"] = int((short.n_pad_samples > 0).sum())
        meta["pilot"] = table
        meta["selection"] = pick_mode(table)
        _atomic_write_text(run_dir / "meta.json", json.dumps(meta, indent=2))
        print(json.dumps({"selection": meta["selection"],
                          "inner": {m: table[m].get("inner", {}).get("min_dcf") for m in pad_modes},
                          "test_frac_above_half": {m: table[m].get("test", {}).get("frac_above_half") for m in pad_modes}},
                         indent=2))  # fmt: skip
        print(f"wrote {run_dir}")
        return 0

    sc = mode_frame(raw, pad_modes[0])
    if mode == "manifest":
        sc.to_csv(run_dir / "scores.csv", index=False)
        meta["stress"] = masked_report(_y(sc), sc.synth_logit, sc[["generator", "source"]])
        meta["direction"] = direction_check(_y(sc), sc.synth_logit)
        _atomic_write_text(run_dir / "meta.json", json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("stress", "direction")}, indent=2))
        print(f"wrote {run_dir}")
        return 0

    # full run: raw -> gates -> fusion + meta in the run dir -> publish
    sc.to_csv(run_dir / "raw.csv", index=False)
    meta.update(split_readouts(sc))
    direction = meta.get("direction", {})
    if direction.get("ok") is not True:
        meta["gate"] = {"direction": "failed", "detail": direction}
        _atomic_write_text(run_dir / "meta.json", json.dumps(meta, indent=2))
        print(f"DIRECTION GATE: {direction}; raw scores kept in {run_dir}, nothing published", flush=True)
        return 4
    for item in args.stress:
        name, path = item.split("=", 1)
        st = pd.read_csv(_under(root, Path(path)))
        inner = sc[sc.split == "inner_oof"]
        meta.setdefault("stress", {})[name] = stress_readout(
            _y(inner), inner.synth_logit, _y(st), st.synth_logit, st[["generator", "source"]])
    fus = sc[BASE_COLS]
    fus.to_csv(run_dir / "fusion.csv", index=False)
    _atomic_write_text(run_dir / "meta.json", json.dumps(meta, indent=2))

    out_dir = root / "outputs" / "detector_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    published = out_dir / f"{args.out_name}.csv"
    _atomic_copy(run_dir / "fusion.csv", published)
    summary = {"stamp": stamp, "run_dir": str(run_dir.relative_to(root)), "config_hash": chash,
               "git_sha": meta["git_sha"], "rows": fails["rows"], "ok": fails["ok"],
               "pad_mode": pad_modes[0], "peak_norm": args.peak_norm,
               "holdout_min_dcf": meta.get("holdout", {}).get("min_dcf"),
               "published_at": datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    _atomic_write_text(out_dir / f"{args.out_name}_summary.json", json.dumps(summary, indent=2))
    _atomic_copy(run_dir / "raw.csv", root / "outputs" / "spectra" / f"{args.out_name}_raw.csv")
    for p in (partial, sidecar):
        if p.exists():
            p.unlink()
    print(json.dumps({k: meta.get(k) for k in ("holdout", "test", "direction")}, indent=2))
    print(f"published {published} ({fails['rows']} rows, {fails['ok']} ok); run dir {run_dir}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
