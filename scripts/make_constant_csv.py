"""M0 rollback artifact: constant-prior CSV through the real test loader.

Every test file is decoded by hearsay.audio.load_audio (so decode failures surface now, not at
the deadline) and gets the same score: the training prior P(synthetic). Writes a new file under
submissions/ and appends a row to submissions/log.csv.

Row order: the manifest's order if --manifest is given (use the sponsor's file if they ship
one), otherwise sorted relative paths under --test-dir.

Usage:
  uv run python scripts/make_constant_csv.py --test-dir data/nsa/test --prior 0.5
  uv run python scripts/make_constant_csv.py --manifest data/nsa/test.csv --id-col file \
      --test-dir data/nsa/test --prior 0.9 --score-max 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from hearsay import SR
from hearsay.submission import REPO, append_log, score_files, write_submission

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".amr", ".mp4"}


def list_test_files(test_dir: Path, manifest: Path | None, id_col: str) -> list[tuple[str, Path]]:
    if manifest:
        with manifest.open(newline="") as f:
            ids = [row[id_col] for row in csv.DictReader(f)]
        return [(i, test_dir / i) for i in ids]
    files = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    return [(str(p.relative_to(test_dir)), p) for p in files]


def preflight(score_fn, fallback: float, tmp: Path) -> None:
    """plan.md: silence + music through the loader before every CSV; finite and reproducible."""
    import soundfile as sf

    t = np.arange(4 * SR) / SR
    chord = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    sf.write(tmp / "silence.wav", np.zeros(4 * SR, dtype=np.float32), SR)
    sf.write(tmp / "music.wav", chord.astype(np.float32), SR)
    paths = [tmp / "silence.wav", tmp / "music.wav"]
    a = score_files(paths, score_fn, fallback)
    b = score_files(paths, score_fn, fallback)
    assert a.scores == b.scores, f"not reproducible: {a.scores} vs {b.scores}"
    assert all(np.isfinite(a.scores)), a.scores
    print(f"preflight: silence={a.scores[0]:.4f} music={a.scores[1]:.4f} flags={a.flags}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--id-col", default="filename", help="manifest column AND output id column")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--prior", type=float, required=True, help="train P(synthetic), in [0, 1]")
    ap.add_argument("--score-max", type=float, default=1.0, help="1 for probability, 100 for %%")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    score = args.prior * args.score_max
    items = list_test_files(args.test_dir, args.manifest, args.id_col)
    if not items:
        sys.exit(f"no audio files found under {args.test_dir}")

    tmp = REPO / "outputs" / "preflight"
    tmp.mkdir(parents=True, exist_ok=True)
    preflight(lambda x: score, score, tmp)

    res = score_files([p for _, p in items], lambda x: score, fallback=score)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = args.out or REPO / "submissions" / f"{stamp}_M0_constprior.csv"
    write_submission(
        [i for i, _ in items], res.scores, out,
        id_col=args.id_col, score_col=args.score_col, score_max=args.score_max,
    )  # fmt: skip

    flagged = [(i, f) for (i, _), f in zip(items, res.flags, strict=True) if f]
    notes = f"constant prior {args.prior}; {len(items)} rows; {len(flagged)} decode flags"
    append_log("M0", out, notes=f"{notes}. {args.notes}".strip(". "))
    print(f"wrote {out} ({len(items)} rows, score={score}); flagged {len(flagged)}")
    for i, f in flagged[:20]:
        print(f"  {f}: {i}")


if __name__ == "__main__":
    main()
