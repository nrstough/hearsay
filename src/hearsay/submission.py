"""Submission files: score every test file in order, validate, write once, log.

NSA format (kickoff slides, Fri Sep 25): tab-separated `TeamName_predictions.tsv`, header
`filename<TAB>cm-score`, score = probability in [0, 1] (0.0 real/bona fide, 1.0 synthetic), no
log-likelihood ratios, every test file present.

Rules enforced here (plan.md, CLAUDE.md): every file yields exactly one finite score in exact
input order; a submission is never overwritten; every one gets a row in submissions/log.csv.
The score must increase with synthetic likelihood.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from hearsay import SR
from hearsay.audio import DecodeError, load_audio

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / "submissions" / "log.csv"
LOG_COLUMNS = [
    "timestamp", "rung", "validation_score", "validation_score_clean_only", "csv_path", "notes",
]  # fmt: skip


@dataclass
class ScoredFiles:
    scores: list[float]
    flags: list[str] = field(default_factory=list)  # "" when the file scored normally

    @property
    def n_flagged(self) -> int:
        return sum(bool(f) for f in self.flags)


def score_files(
    paths: Sequence[str | Path],
    score_fn: Callable[[np.ndarray], float],
    fallback: float,
) -> ScoredFiles:
    """Score each file in order. Any failure yields `fallback` plus a flag, never a missing row."""
    scores, flags = [], []
    for p in paths:
        try:
            x = load_audio(p)
        except DecodeError:
            scores.append(float(fallback))
            flags.append("decode_error")
            continue
        try:
            s = float(score_fn(x))
        except Exception as e:  # noqa: BLE001 - one bad clip must not lose the CSV
            scores.append(float(fallback))
            flags.append(f"score_error:{type(e).__name__}")
            continue
        if not math.isfinite(s):
            scores.append(float(fallback))
            flags.append("nonfinite_score")
            continue
        scores.append(s)
        flags.append("")
    return ScoredFiles(scores, flags)


def write_submission(
    ids: Sequence[str],
    scores: Sequence[float],
    out_path: str | Path,
    *,
    id_col: str = "filename",
    score_col: str = "cm-score",
    score_max: float = 1.0,
    sep: str = "\t",
) -> Path:
    """Validate and write a submission (TSV by default); refuses to overwrite; reads it back."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; every submission gets a new file")
    if len(ids) != len(scores):
        raise ValueError(f"{len(ids)} ids vs {len(scores)} scores")
    if len(ids) == 0:
        raise ValueError("empty submission")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{int((~np.isfinite(arr)).sum())} non-finite scores")
    if arr.min() < 0 or arr.max() > score_max:
        raise ValueError(f"scores outside [0, {score_max}]: [{arr.min()}, {arr.max()}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", newline="") as f:
        w = csv.writer(f, delimiter=sep, lineterminator="\n")
        w.writerow([id_col, score_col])
        w.writerows((i, repr(float(s))) for i, s in zip(ids, arr, strict=True))

    with out_path.open(newline="") as f:
        rows = list(csv.reader(f, delimiter=sep))
    assert rows[0] == [id_col, score_col]
    assert [r[0] for r in rows[1:]] == list(ids), "row order changed on write"
    assert np.allclose([float(r[1]) for r in rows[1:]], arr, rtol=0, atol=0)
    return out_path


def append_log(
    rung: str,
    csv_path: str | Path,
    *,
    validation_score: float | str = "",
    validation_score_clean_only: float | str | None = None,
    notes: str = "",
    log_path: Path = LOG_PATH,
) -> None:
    """Append one row to submissions/log.csv. Clean-only defaults to the main score."""
    if validation_score_clean_only is None:
        validation_score_clean_only = validation_score
    try:
        csv_rel = Path(csv_path).resolve().relative_to(REPO)
    except ValueError:
        csv_rel = Path(csv_path)
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().astimezone().isoformat(timespec="seconds"), rung, validation_score,
            validation_score_clean_only, str(csv_rel), notes,
        ])  # fmt: skip


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
