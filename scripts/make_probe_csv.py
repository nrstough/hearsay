"""Learned CSV from an M1 probe: load_audio -> embed_clip -> probe.p_synthetic.

Writes the calibrated submission under submissions/ (logged), plus a sidecar under outputs/
with the raw decision value (rank-preserving fallback if the metric punishes calibration) and
per-file flags.

Usage:
  uv run python scripts/make_probe_csv.py --probe models/m1_... --test-dir data/nsa/test \
      --val-score 0.034 --notes "val = NSA grouped split EER"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.embed import embed_clip, load_backbone
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
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--score-max", type=float, default=1.0)
    ap.add_argument("--fallback", type=float, default=0.5, help="P for undecodable files")
    ap.add_argument("--val-score", default="", help="validation metric for log.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    probe = Probe.load(args.probe)
    model = load_backbone(probe.backbone)
    raw: list[float] = []

    def score(x: np.ndarray) -> float:
        e = embed_clip(model, x, probe.win_s, probe.max_windows)
        raw.append(float(probe.clf.decision_function(e[None, probe.layer, :])[0]))
        return float(probe.p_synthetic(e)[0]) * args.score_max

    items = list_test_files(args.test_dir, args.manifest, args.id_col)
    if not items:
        sys.exit(f"no audio files found under {args.test_dir}")
    tmp = REPO / "outputs" / "preflight"
    tmp.mkdir(parents=True, exist_ok=True)
    fb = args.fallback * args.score_max
    preflight(score, fb, tmp)
    raw.clear()

    res = score_files([p for _, p in items], score, fallback=fb)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    name = f"{stamp}_M1_{args.probe.name}"
    out = REPO / "submissions" / f"{name}.csv"
    ids = [i for i, _ in items]
    write_submission(
        ids, res.scores, out, id_col=args.id_col, score_col=args.score_col,
        score_max=args.score_max,
    )  # fmt: skip

    # Sidecar: raw decision values in row order (NaN where the file fell back).
    raw_iter = iter(raw)
    raw_col = [next(raw_iter) if f in ("", "nonfinite_score") else np.nan for f in res.flags]
    side = REPO / "outputs" / "submissions" / f"{name}.sidecar.csv"
    side.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "p": res.scores, "raw": raw_col, "flag": res.flags}).to_csv(
        side, index=False
    )

    notes = f"probe {args.probe.name}; {len(ids)} rows; {res.n_flagged} flagged"
    append_log("M1", out, validation_score=args.val_score, notes=f"{notes}. {args.notes}".strip(". "))
    print(f"wrote {out} ({len(ids)} rows, {res.n_flagged} flagged); sidecar {side}")


if __name__ == "__main__":
    main()
