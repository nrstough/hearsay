"""Learned submission from an M1 probe: load_audio -> embed_clip -> probe LLR -> probability.

The NSA metric is MinDCF (threshold swept), so only ranking counts; we submit the calibrated
posterior at the test prior, sigmoid(LLR + logit(pi_synth)), a probability as the sponsor asked
(no LLRs). --cost-shift additionally subtracts ln 4 so a fixed 50% cut-off would make the 4:1
Bayes decision (only useful if the sponsor ever thresholds instead of sweeping).

Writes the NSA TSV under submissions/ (logged) and a sidecar under outputs/ with the
prior-neutral LLR and per-file flags.

Usage:
  uv run python scripts/make_probe_csv.py --probe models/m1_... --test-dir data/nsa/test \
      --pi-synth 0.3 --val-mindcf 0.21 --val-eer 0.034 --notes "val = NSA grouped split"
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.embed import embed_clip, load_backbone
from hearsay.metrics import C_FA, C_MISS, sigmoid
from hearsay.probe import Probe
from hearsay.submission import (
    REPO,
    append_log,
    list_test_files,
    preflight,
    score_files,
    write_submission,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--id-col", default="filename")
    ap.add_argument("--pi-synth", type=float, default=0.3, help="NSA test prior (70/30 real/synth)")
    ap.add_argument("--cost-shift", action="store_true", help="subtract ln(C_FA/C_MISS)")
    ap.add_argument("--val-mindcf", default="", help="val normalized minDCF -> validation_score")
    ap.add_argument("--val-eer", default="", help="val EER, logged in notes")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    probe = Probe.load(args.probe)
    model = load_backbone(probe.backbone)
    shift = math.log(args.pi_synth / (1 - args.pi_synth))
    if args.cost_shift:
        shift -= math.log(C_FA / C_MISS)
    raw: list[float] = []

    def score(x: np.ndarray) -> float:
        e = embed_clip(model, x, probe.win_s, probe.max_windows)
        llr = float(probe.llr(e)[0])
        raw.append(llr)  # last statement before return: a raise above leaves no entry
        return float(sigmoid(llr + shift))

    items = list_test_files(args.test_dir, args.manifest, args.id_col)
    if not items:
        sys.exit(f"no audio files found under {args.test_dir}")
    tmp = REPO / "outputs" / "preflight"
    tmp.mkdir(parents=True, exist_ok=True)
    fb = float(sigmoid(shift))  # undecodable -> the prior-only posterior
    preflight(score, fb, tmp)
    raw.clear()

    res = score_files([p for _, p in items], score, fallback=fb)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    name = f"{stamp}_M1_{args.probe.name}"
    out = REPO / "submissions" / f"{name}.tsv"
    ids = [i for i, _ in items]
    write_submission(ids, res.scores, out, id_col=args.id_col)

    raw_iter = iter(raw)
    raw_col = [next(raw_iter) if f in ("", "nonfinite_score") else np.nan for f in res.flags]
    side = REPO / "outputs" / "submissions" / f"{name}.sidecar.csv"
    side.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "p": res.scores, "llr": raw_col, "flag": res.flags}).to_csv(
        side, index=False
    )

    notes = (f"minDCF={args.val_mindcf} EER={args.val_eer}; probe {args.probe.name}; "
             f"pi_synth={args.pi_synth} cost_shift={args.cost_shift}; "
             f"{len(ids)} rows; {res.n_flagged} flagged")  # fmt: skip
    append_log(
        "M1", out, validation_score=args.val_mindcf, notes=f"{notes}. {args.notes}".strip(". ")
    )
    print(f"wrote {out} ({len(ids)} rows, {res.n_flagged} flagged); sidecar {side}")


if __name__ == "__main__":
    main()
