"""Drop-time inventory (plan.md 0:20-0:40): N, formats, sample rate and duration by class,
class balance, decode failures, and the hour-one shortcut checks.

Shortcut checks: AUC of clip duration, leading/trailing silence, and peak level vs. label. Any
|AUC - 0.5| > 0.15 means a trivial feature predicts the label and the split/model must not lean
on it.

Usage:
  uv run python scripts/inventory.py --root data/nsa/train --labels data/nsa/train.csv \
      --id-col file --label-col label --name nsa_train [--decode-sample 400]
  uv run python scripts/inventory.py --root data/nsa/test --name nsa_test
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hearsay import SR
from hearsay.audio import DecodeError, load_audio, probe_audio
from hearsay.submission import AUDIO_EXT

REPO = Path(__file__).resolve().parents[1]


def decode_stats(p: str) -> dict:
    """Silence/level facts from the real loader (-40 dBFS frame threshold, 20 ms frames)."""
    try:
        x = load_audio(p)
    except DecodeError as e:
        return {"decode_ok": False, "decode_err": str(e)[-120:]}
    f = SR // 50
    n = max(1, x.size // f)
    rms = np.sqrt(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    voiced = np.where(20 * np.log10(rms) > -40)[0]
    lead = voiced[0] * 0.02 if voiced.size else x.size / SR
    trail = (n - 1 - voiced[-1]) * 0.02 if voiced.size else x.size / SR
    return {
        "decode_ok": True, "decoded_s": x.size / SR, "peak": float(np.abs(x).max()),
        "lead_sil_s": lead, "trail_sil_s": trail,
        "voiced_frac": voiced.size / n, "nonfinite_or_empty": bool(x.size == 0),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--id-col", default="filename")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--name", required=True)
    ap.add_argument("--decode-sample", type=int, default=400, help="stratified; 0 = all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    files = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    others = [p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() not in AUDIO_EXT]
    df = pd.DataFrame({"id": [str(p.relative_to(args.root)) for p in files],
                       "path": [str(p) for p in files]})  # fmt: skip
    print(f"{len(df)} audio files under {args.root}; {len(others)} other files "
          f"(ext: {sorted({p.suffix for p in others})[:10]})")

    if args.labels:
        lab = pd.read_csv(args.labels)
        print(f"labels: {args.labels} columns={list(lab.columns)} rows={len(lab)}")
        lab = lab.rename(columns={args.id_col: "id", args.label_col: "label"})
        if not lab.id.isin(df.id).all():  # labels may use basenames
            df["id"] = [Path(i).name for i in df.id]
        miss = (~lab.id.isin(df.id)).sum()
        df = df.merge(lab, on="id", how="left")
        print(f"label rows without a file: {miss}; files without a label: {df.label.isna().sum()}")
    else:
        df["label"] = "unlabeled"

    with ProcessPoolExecutor(args.workers) as ex:
        meta = list(ex.map(probe_audio, df.path, chunksize=32))
    df = pd.concat([df, pd.DataFrame(meta)], axis=1)

    per = max(1, args.decode_sample // max(1, df.label.nunique()))
    samp = df if args.decode_sample == 0 else pd.concat(
        g.sample(min(len(g), per), random_state=0) for _, g in df.groupby("label", dropna=False)
    )
    with ProcessPoolExecutor(args.workers) as ex:
        dec = list(ex.map(decode_stats, samp.path, chunksize=16))
    samp = pd.concat([samp.reset_index(drop=True), pd.DataFrame(dec)], axis=1)

    out = REPO / "outputs" / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.name}_probe.csv", index=False)
    samp.to_csv(out / f"{args.name}_decode_sample.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n== class balance\n", df.label.value_counts(dropna=False).to_string())
    for col in ("format", "codec", "sample_rate", "channels"):
        print(f"\n== {col} by class\n", pd.crosstab(df[col].fillna("?"), df.label).to_string())
    print("\n== duration_s by class\n", df.groupby("label").duration_s.describe().to_string())
    print(f"\n== probe errors: {df['error'].notna().sum() if 'error' in df else 0}")
    print(f"== decode sample: {len(samp)} files, failures {(~samp.decode_ok).sum()}")
    print(samp.groupby("label")[["decoded_s", "lead_sil_s", "trail_sil_s", "voiced_frac",
                                 "peak"]].median().to_string())  # fmt: skip

    labs = sorted(set(df.label.dropna()) - {"unlabeled"})
    if len(labs) == 2:
        print(f"\n== shortcut AUCs (positive = {labs[1]!r}); flag if |AUC-0.5| > 0.15")
        ok = samp[samp.decode_ok]
        y = (ok.label == labs[1]).to_numpy(int)
        for col in ("duration_s", "decoded_s", "lead_sil_s", "trail_sil_s", "voiced_frac", "peak",
                    "sample_rate", "bit_rate"):  # fmt: skip
            v = pd.to_numeric(ok[col], errors="coerce")
            if v.notna().sum() > 10 and v.nunique() > 1:
                a = roc_auc_score(y[v.notna()], v[v.notna()])
                print(f"  {col:12s} AUC {a:.3f}{'  <-- SHORTCUT' if abs(a - 0.5) > 0.15 else ''}")
    print(f"\nwrote {out}/{args.name}_*.csv")


if __name__ == "__main__":
    main()
