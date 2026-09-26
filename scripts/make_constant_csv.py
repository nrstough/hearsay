"""M0 rollback artifact: constant submission through the real test loader.

Every test file is decoded by hearsay.audio.load_audio (so decode failures surface now, not at
the deadline) and gets the same hard decision. Under the NSA 4:1 false-alarm cost the best
constant is "always real" (0.0) unless pi_synth > 0.8, then "always synthetic" (1.0). Its
normalized minDCF is 1.0 by definition; every model must beat it.

Writes the NSA TSV (filename<TAB>cm-score) under submissions/ and appends a log.csv row.
Row order: the manifest's order if --manifest is given, else sorted relative paths.

Usage:
  uv run python scripts/make_constant_csv.py --test-dir data/nsa/test --pi-synth 0.5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from hearsay.metrics import C_FA, C_MISS
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
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--id-col", default="filename", help="manifest column AND output id column")
    ap.add_argument("--pi-synth", type=float, required=True, help="NSA train P(synthetic)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    pi = args.pi_synth
    synth = C_FA * (1 - pi) < C_MISS * pi  # always-synthetic is cheaper only when pi > 0.8
    score = 1.0 if synth else 0.0
    verdict = "always synthetic" if synth else "always real"
    print(f"pi_synth={pi} -> {verdict} ({score})")

    items = list_test_files(args.test_dir, args.manifest, args.id_col)
    if not items:
        sys.exit(f"no audio files found under {args.test_dir}")
    tmp = REPO / "outputs" / "preflight"
    tmp.mkdir(parents=True, exist_ok=True)
    preflight(lambda x: score, score, tmp)

    res = score_files([p for _, p in items], lambda x: score, fallback=score)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = args.out or REPO / "submissions" / f"{stamp}_M0_constant.tsv"
    write_submission([i for i, _ in items], res.scores, out, id_col=args.id_col)

    flagged = [(i, f) for (i, _), f in zip(items, res.flags, strict=True) if f]
    notes = (f"minDCF=1.0 EER=0.5; constant {verdict} (pi_synth={pi}); "
             f"{len(items)} rows; {len(flagged)} decode flags")  # fmt: skip
    append_log("M0", out, validation_score=1.0, notes=f"{notes}. {args.notes}".strip(". "))
    print(f"wrote {out} ({len(items)} rows); flagged {len(flagged)}")
    for i, f in flagged[:20]:
        print(f"  {f}: {i}")


if __name__ == "__main__":
    main()
